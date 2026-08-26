"""
This module implements an `AbstractVector` whose underlying data is stored in an HDF5 file.

It generally can store vectors of elements with fixed sizes, where that element is one of:

* HDF5-compatible scalar type (booleans, signed/unsigned integers, and floats)
* Enum
* SVector, SMatrix, SArray, and NTuple values with supported element types
* general composite types with supported fields
* Vector, Matrix, and Array values with supported element types and declared dimensions
* String
* Symbol
* Char
* reconstructible singleton types

Further, it can serialize types to bytes or strings and store those in the HDF5 file. This
allows it to store:

* Custom types explicitly assigned a serialization storage style
* Vector, Matrix, and Array values whose dimensions are not declared or vary between
  elements

It supports common `AbstractVector` operations and can grow via `push!`. Direct iteration
reads elements individually from HDF5. `iterable(arr)` first loads the full vector into
memory and is generally faster when the vector fits in RAM.
"""
module HDF5Vectors

export AbstractHDF5Vector, create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector, iterable

using HDF5
using StaticArrays: StaticArray, SVector

# See https://juliaio.github.io/HDF5.jl/stable/#Supported-data-types
const hdf5_scalar_types = Union{Bool, UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64}

##################
# Storage Styles #
##################

"""
An abstract type intended as a parent for all HDF5 vector storage styles. Custom backends
define a subtype and select it with [`storage_style`](@ref).
"""
abstract type AbstractHDF5VectorStorageStyle end

"""
Used when each value can be represented in one HDF5 dataset, including:

* Int8, Int16, Int32, and Int64 (and unsigned forms)
* Float32 and Float64
* Enum and Char values represented by HDF5-native integers
* Bits-type structs when non-portable storage is requested
* String
"""
struct ElementalStorageStyle{HT} <: AbstractHDF5VectorStorageStyle
    datatype::Type{HT}
end

"""
Used to store composite structured information by placing each field in its own HDF5
vector, including:

* Heterogeneous tuples
* Named tuples
* General structs with supported field types
"""
struct CompositeStorageStyle <: AbstractHDF5VectorStorageStyle end

#############
# Interface #
#############

function singleton_value(::Type{T}) where {T}

    if !applicable(T)
        throw(ArgumentError(
            "The singleton type $T does not have a zero-argument constructor and cannot " *
            "use SingletonStorageStyle.",
        ))
    end

    value = T()
    if !(value isa T)
        throw(ArgumentError(
            "The zero-argument constructor for singleton type $T did not return a $T.",
        ))
    end
    return value

end

singleton_value(::Type{NamedTuple{(), Tuple{}}}) = (;)

function unsupported_element_type(el_type, reason)
    throw(ArgumentError(
        "HDF5Vectors does not support the element type $el_type: $reason " *
        "Define a storage_style method for this type to select an explicit storage format.",
    ))
end

# Types that aren't native HDF5 scalars but that are bits-types can still be stored using
# the elemental storage type, but that's not portable, so this function considers
# portability before deciding to store non-native types as elemental or composite.

"""
    storage_style(el_type::Type; dims = nothing, portable = true, kwargs...)

Returns the storage style used for vectors with the declared element type `el_type`.
Built-in styles include:

* `ElementalStorageStyle` for scalars or non-portable bits-type structs
* `SingletonStorageStyle` for types that have exactly one possible value
* `ArrayStorageStyle` for arrays of known, consistent dimensions holding elemental types
* `CompositeStorageStyle` for field-oriented structs and heterogeneous tuples
* `ByteArrayStorageStyle` for Julia serialization
* `JSONStorageStyle` for JSON3 serialization

`dims` declares the fixed dimensions of dynamically sized array elements. `portable`
selects field-oriented storage rather than a native HDF5 datatype for bits-type composite
elements.

A more-specific method can select a representation for a custom element type. Style
selection occurs both when a vector is created and when it is loaded, so a custom method
must return the same style from the element type and stored options.

For example, this selects Julia byte serialization for `MyType`:

```
HDF5Vectors.storage_style(::Type{MyType}; kwargs...) = HDF5Vectors.ByteArrayStorageStyle()
```

[Supported Element Types and Creation Options](@ref) and [Custom Element Types](@ref)
describe the selection rules and provide customization examples.
"""
function storage_style(el_type::Type; portable = true, kwargs...)

    # HDF5-native types have more-specific methods. This fallback selects a safe
    # representation for other types.

    # We can figure out the fields of concrete types.
    if isconcretetype(el_type)

        if Base.issingletontype(el_type)

            # Singleton values need no per-element storage, but we must be able to construct
            # the value through a public interface when loading the vector.
            singleton_value(el_type)
            return SingletonStorageStyle()

        elseif isprimitivetype(el_type)

            # All HDF5-native primitive types have more-specific storage_style methods. An
            # unknown primitive type has no fields to store as a composite and must not be
            # mistaken for a singleton.
            return unsupported_element_type(
                el_type,
                "HDF5.jl does not provide a native datatype for it.",
            )

        elseif isempty(fieldnames(el_type))

            # Mutable zero-field structs have distinct identities despite carrying no
            # fields. There is no value representation that this package can preserve.
            return unsupported_element_type(
                el_type,
                "it has no fields but is not a singleton type.",
            )

        elseif isbitstype(el_type) && !portable

            # HDF5 can store a bits type as one custom datatype, but the resulting layout is
            # more difficult to interpret outside Julia.
            return ElementalStorageStyle(el_type)

        else

            # Store other concrete types field-by-field.
            return CompositeStorageStyle()

        end

    else

        # Otherwise, we don't know the structure of this type and have no other way to log
        # it, so we'll serialize it as a fallback.
        return ByteArrayStorageStyle()

    end

end

"""
    construct(type::Type, el)

Creates the appropriate Julia value from the raw element `el` retrieved from the
HDF5 dataset. The behaviour is determined by the storage style associated with `type`.
No generic implementation is provided; storage backends define methods for their
particular container types and element representations.
"""
function construct end

"""
    deconstruct(type::Type, el)

Converts the Julia value `el` into the representation stored in the HDF5 file. The
storage style associated with `type` chooses how the conversion is performed. No generic
implementation is provided; storage backends define methods for their particular
container types and element representations. For composite storage, the result contains
one stored value for each declared field, in field order.
"""
function deconstruct end

######################
# AbstractHDF5Vector #
######################

"""
An abstract type intended as the parent for all types of HDF5 vectors. Subtypes should have
a corresponding storage style and implement [`create_hdf5_vector`](@ref), and at least these
parts of the AbstractArray interface: `length`, `push!`, `getindex`, and `collect`.
`setindex!` is optional for storage that supports in-place replacement.
"""
abstract type AbstractHDF5Vector{T} <: AbstractVector{T} end

Base.eltype(::Type{<:AbstractHDF5Vector{ET}}) where {ET} = ET
Base.size(arr::AbstractHDF5Vector) = (length(arr),)
Base.similar(::AbstractHDF5Vector{T}, ::Type{T}, dims::Dims) where {T} = Vector{T}(undef, dims)
Base.IndexStyle(::Type{<:AbstractHDF5Vector}) = IndexLinear()
Base.broadcastable(arr::AbstractHDF5Vector) = collect(arr)

# Composite storage checks this before replacing any field, so append-only children can
# reject the operation before another child is changed.
"""
    supports_setindex(vector::AbstractHDF5Vector)

Returns whether `vector` supports replacing existing elements with `setindex!`. The default
is `false`. A custom backend that implements replacement and can be nested in composite
storage should define this to return `true`.
"""
supports_setindex(::AbstractHDF5Vector) = false

function validate_chunk_length(chunk_length)
    if !(chunk_length isa Integer) || chunk_length isa Bool || chunk_length <= 0
        throw(ArgumentError("chunk_length must be a positive integer; got $chunk_length."))
    end
    return Int64(chunk_length)
end

function validate_dims(dims, expected_ndims)

    if isnothing(dims)
        return nothing
    elseif !(dims isa Tuple)
        throw(ArgumentError("dims must be a tuple of positive integers; got $dims."))
    elseif length(dims) != expected_ndims
        throw(DimensionMismatch(
            "Expected $expected_ndims dimensions, but dims contains $(length(dims)).",
        ))
    elseif any(dim -> !(dim isa Integer) || dim isa Bool || dim <= 0, dims)
        throw(ArgumentError("dims must be a tuple of positive integers; got $dims."))
    end

    return Tuple(Int64(dim) for dim in dims)

end

function validate_fixed_dims(dims, expected_dims)

    if isnothing(dims)
        return expected_dims
    elseif !(dims isa Tuple) || any(
        dim -> !(dim isa Integer) || dim isa Bool || dim < 0,
        dims,
    )
        throw(ArgumentError("dims must be a tuple of nonnegative integers; got $dims."))
    end

    dims = Tuple(Int64(dim) for dim in dims)
    if dims != expected_dims
        throw(DimensionMismatch(
            "The provided dimensions $dims do not match the type dimensions " *
            "$expected_dims.",
        ))
    end
    return expected_dims

end

# Route reductions through one bulk read rather than scalar HDF5 reads.
function Base.mapreduce(f, op, arr::AbstractHDF5Vector; kwargs...)
    return mapreduce(f, op, iterable(arr); kwargs...)
end
# Use the iterable form rather than trying to iterate via getindex.
Base.map(f, arr::AbstractHDF5Vector) = map(f, iterable(arr))

# Range and integer-vector indexing should behave like any other AbstractVector and return a
# plain Julia Vector. We construct these from scalar indexing so all storage backends share
# one consistent behavior.
function Base.getindex(arr::AbstractHDF5Vector, k::AbstractRange{<:Integer})
    return [arr[j] for j in k]
end

# Logical indexing examines the whole mask, so one bulk read is generally much faster than
# a separate HDF5 read for every selected position.
function Base.getindex(arr::AbstractHDF5Vector, mask::AbstractVector{Bool})
    checkbounds(arr, mask)
    return collect(arr)[mask]
end
function Base.getindex(arr::AbstractHDF5Vector, k::AbstractVector{<:Integer})
    return [arr[j] for j in k]
end
Base.getindex(arr::AbstractHDF5Vector, ::Colon) = collect(arr)

abstract type AbstractHDF5VectorIterator{T} end

# This loads all data up front so iteration does not perform one HDF5 read per element.
struct HDF5VectorIterator{T} <: AbstractHDF5VectorIterator{T}
    data::Vector{T}
    count::Int64
end

struct HDF5VectorIteratorState
    index::Int64
end

"""
    iterable(arr::AbstractHDF5Vector)

Loads `arr` into memory and returns an iterator over the collected values. This is generally
much faster than reading one element from HDF5 on each iteration, but requires enough memory
to hold the entire vector.
"""
iterable(arr::AbstractHDF5Vector) = HDF5VectorIterator(collect(arr), length(arr))

Base.eltype(::Type{<:AbstractHDF5VectorIterator{T}}) where {T} = T
Base.length(itr::HDF5VectorIterator) = itr.count

function Base.iterate(itr::HDF5VectorIterator, state = HDF5VectorIteratorState(1))
    if state.index > itr.count
        return nothing
    end
    (el, next_data_itr_state) = iterate(itr.data, state.index)
    return (el, HDF5VectorIteratorState(next_data_itr_state))
end

function serialize_to_byte_array(x)
    io = IOBuffer() # Will use UInt8 by default.
    Serialization.serialize(io, x)
    return take!(io)
end

function deserialize_from_byte_array(x)
    io = IOBuffer(x)
    return Serialization.deserialize(io)
end

"""
    store_metadata(
        style::AbstractHDF5VectorStorageStyle,
        group::HDF5.Group,
        el_type;
        dims = nothing,
        portable,
    )

Stores the common metadata required by `load_hdf5_vector` in a newly created HDF5 vector
group. This is an implementation hook for custom storage backends; ordinary callers do not
need to call it.
"""
function store_metadata(
    style::AbstractHDF5VectorStorageStyle,
    group::HDF5.Group,
    el_type;
    dims = nothing,
    portable,
)
    metadata_group = HDF5.create_group(group, "metadata")
    metadata_group["type"] = string(el_type)
    metadata_group["serialized_type"] = serialize_to_byte_array(el_type)
    metadata_group["dimensions_are_constant"] = !isnothing(dims)
    metadata_group["dimensions"] = isnothing(dims) ? Int64[] : Int64[dims...,]
    metadata_group["portable"] = portable
    return metadata_group
end

"""
    create_hdf5_vector(
        group::HDF5.Group,
        name::AbstractString,
        el_type;
        dims = nothing,
        chunk_length = 1000,
        portable = true,
    )

Creates the appropriate HDF5 vector type for the given element type, storing the vector in
the given HDF5 `group` under `name`.

Optional keyword arguments:

* `dims = nothing`: Dimensions of each dynamically sized array element. This must be a
  tuple of positive integers whose length matches the array rank. It enables efficient
  array-like storage for arrays of elemental values; dimensions of tuples and static
  arrays are inferred from their types.
* `chunk_length = 1000`: Positive integer chunk length for the underlying HDF5 datasets. It
  affects storage layout and I/O performance but does not limit the vector length.
* `portable = true`: When true, use field-oriented storage for composite bits types. When
  false, permit a faster native HDF5 datatype where available. It is ignored for types with
  only one supported representation and does not make Julia-serialized data portable.
"""
function create_hdf5_vector(
    group::HDF5.Group, name::AbstractString, el_type;
    dims = nothing, chunk_length = 1000, portable = true,
)
    chunk_length = validate_chunk_length(chunk_length)
    return create_hdf5_vector(
        storage_style(el_type; dims, portable),
        group, name, el_type;
        dims, chunk_length, portable,
    )
end

function read_storage_options(metadata_group::HDF5.Group)
    dimensions_are_constant = read(metadata_group["dimensions_are_constant"])
    dims = dimensions_are_constant ? (read(metadata_group["dimensions"])...,) : nothing
    portable = read(metadata_group["portable"])
    return (; dims, portable)
end

"""
    load_hdf5_vector(group::HDF5.Group)

Reconstructs an HDF5 vector from a group created by `create_hdf5_vector`. The metadata stored
in the group (type, dimensions, portability) is used to determine which specific vector
implementation to instantiate. This form takes only the `group` and pulls the element type
from the metadata.
"""
function load_hdf5_vector(group::HDF5.Group)
    metadata_group = group["metadata"]
    el_type = deserialize_from_byte_array(read(metadata_group["serialized_type"]))
    options = read_storage_options(metadata_group)
    return load_hdf5_vector(
        storage_style(el_type; options...),
        group,
        el_type;
        options...,
    )
end

"""
    load_hdf5_vector(group::HDF5.Group, el_type)

Reconstructs an HDF5 vector when the caller already knows the element type. When loading a
group created by `create_hdf5_vector`, its stored dimensions and portability setting are
used to select the storage representation.
"""
function load_hdf5_vector(group::HDF5.Group, el_type)
    options = read_storage_options(group["metadata"])
    return load_hdf5_vector(
        storage_style(el_type; options...),
        group,
        el_type;
        options...,
    )
end

"""
    copy_to_hdf5_vector(
        group::HDF5.Group,
        name::AbstractString,
        collection;
        dims = nothing,
        chunk_length = 1000,
        portable = true,
    )

Creates an HDF5 vector using `eltype(collection)` and copies the collection into it. The
copy uses specialized bulk writes where the selected storage format supports them.

Optional keyword arguments:

* `dims = nothing`: Dimensions of each dynamically sized array element. This must be a
  tuple of positive integers whose length matches the array rank. It enables efficient
  array-like storage for arrays of elemental values; dimensions of tuples and static
  arrays are inferred from their types. Dimensions are not inferred by inspecting the
  collection.
* `chunk_length = 1000`: Positive integer chunk length for the underlying HDF5 datasets. It
  affects storage layout and I/O performance but does not limit the vector length.
* `portable = true`: When true, use field-oriented storage for composite bits types. When
  false, permit a faster native HDF5 datatype where available. It is ignored for types with
  only one supported representation and does not make Julia-serialized data portable.
"""
function copy_to_hdf5_vector(
    group::HDF5.Group, name::AbstractString, collection;
    dims = nothing, chunk_length = 1000, portable = true,
)
    chunk_length = validate_chunk_length(chunk_length)
    return copy_to_hdf5_vector(
        storage_style(eltype(collection); dims, portable),
        group, name, collection;
        dims, chunk_length, portable,
    )
end

# The generic implementation creates the vector and fills it one element at a time. Storage
# backends can specialize this method when their representation supports a bulk write.
function copy_to_hdf5_vector(
    style::AbstractHDF5VectorStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    kwargs...,
)
    v = create_hdf5_vector(style, group, name, eltype(collection); kwargs...)
    for el in collection
        push!(v, el)
    end
    return v
end

"""
    create_hdf5_vector(
        style::AbstractHDF5VectorStorageStyle,
        group::HDF5.Group,
        name::AbstractString,
        el_type;
        kwargs...,
    )

Creates the appropriate HDF5 vector type for the given storage style and element type,
storing the vector in the given HDF5 `group` under `name`. This is an
implementation hook for storage backends; users should select a style by defining
[`storage_style`](@ref) for their element type and call the overload without a style.
"""
function create_hdf5_vector(
    style::AbstractHDF5VectorStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type;
    kwargs...,
)
    error("There is no implementation of `create_hdf5_vector` for the $(typeof(style)) storage style used for name = $name with el_type = $el_type.")
end

##############################
# HDF5VectorOfElementalTypes #
##############################

# `construct` and `deconstruct` allow multiple Julia element types to share this dataset
# implementation while using different HDF5 representations.
mutable struct HDF5VectorOfElementalTypes{T, DT} <: AbstractHDF5Vector{T}
    dataset::HDF5.Dataset
    datatype::Type{DT}
    count::Int64
end

function create_hdf5_vector(
    style::ElementalStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type;
    chunk_length,
    portable,
    kwargs...,
)
    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, el_type; portable)
    datatype = style.datatype
    vector_dims = (0,) # Last dimension is 0 until we start writing to it.
    max_dims = (-1,) # Last dimension can grow forever.
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", datatype, dataspace; chunk = (chunk_length,))
    return HDF5VectorOfElementalTypes{el_type, datatype}(dataset, datatype, 0)
end

# We customize this for speed. It's much faster to save an array all at once rather than
# saving it element-by-element.
function copy_to_hdf5_vector(
    style::ElementalStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    chunk_length,
    portable,
    kwargs...,
)

    # Deconstruct the full collection before changing the HDF5 file. A conversion error
    # should not leave a partially created vector behind.
    el_type = eltype(collection)
    datatype = style.datatype
    type = HDF5VectorOfElementalTypes{el_type, datatype}
    array = datatype[deconstruct(type, el) for el in collection]
    n = length(array)

    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, el_type; portable)

    # Use an extensible dataspace so the copied vector can continue to grow.
    vector_dims = (n,)
    max_dims = (-1,) # This can grow forever.
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", datatype, dataspace; chunk = (chunk_length,))

    this_group["data"][:] = array

    return HDF5VectorOfElementalTypes{el_type, datatype}(dataset, datatype, n)

end

function load_hdf5_vector(
    style::ElementalStorageStyle,
    group::HDF5.Group,
    el_type;
    kwargs...,
)
    dataset = group["data"]
    datatype = style.datatype
    count = size(dataset)[end]
    return HDF5VectorOfElementalTypes{el_type, datatype}(dataset, datatype, count)
end

Base.length(arr::HDF5VectorOfElementalTypes) = arr.count # Common with HDF5VectorOfArrayishTypes
supports_setindex(::HDF5VectorOfElementalTypes) = true

function Base.setindex!(arr::HDF5VectorOfElementalTypes{T}, el::T, k::Int) where {T}
    arr.dataset[k] = deconstruct(typeof(arr), el)
    return el
end
function Base.getindex(arr::HDF5VectorOfElementalTypes{T, DT}, k::Int) where {T, DT}
    construct(typeof(arr), read(arr.dataset, DT, k))
end
function Base.collect(arr::HDF5VectorOfElementalTypes{T, DT}) where {T, DT}
    data = read(arr.dataset, DT, 1:arr.count)
    return [construct(typeof(arr), el) for el in data]
end

function Base.push!(arr::HDF5VectorOfElementalTypes{T}, el::T) where {T}
    next_count = arr.count + 1
    HDF5.set_extent_dims(arr.dataset, (next_count,))
    arr.dataset[next_count] = deconstruct(typeof(arr), el)
    arr.count = next_count
    return arr
end

function array_element_datatype(type; kwargs...)
    style = storage_style(type; kwargs...)
    return style isa ElementalStorageStyle ? style.datatype : nothing
end

##############################
# HDF5VectorOfSingletonTypes #
##############################

# Singleton types have exactly one possible value, so each vector element is already known
# from the element type. We store only the vector length and reconstruct that value for every
# valid index.

"""
Used for singleton types whose value can be reconstructed from the element type. Only the
vector length is stored because the element type determines every value.
"""
struct SingletonStorageStyle <: AbstractHDF5VectorStorageStyle end

mutable struct HDF5VectorOfSingletonTypes{T} <: AbstractHDF5Vector{T}
    dataset::HDF5.Dataset
    count::Int64
end

function create_hdf5_vector(
    style::SingletonStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type;
    chunk_length,
    portable,
    kwargs...,
)

    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, el_type; portable)

    # The single integer in the dataset is the vector length.
    vector_dims = (1,)
    max_dims = (1,)
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", Int64, dataspace)
    dataset[1] = 0
    return HDF5VectorOfSingletonTypes{el_type}(dataset, 0)

end

function copy_to_hdf5_vector(
    style::SingletonStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    chunk_length,
    portable,
    kwargs...,
)

    el_type = eltype(collection)
    n = length(collection)
    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, el_type; portable)

    # The single integer in the dataset is the vector length.
    vector_dims = (1,)
    max_dims = (1,)
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", Int64, dataspace)
    dataset[1] = n
    return HDF5VectorOfSingletonTypes{el_type}(dataset, n)

end

function load_hdf5_vector(
    style::SingletonStorageStyle,
    group::HDF5.Group,
    el_type;
    kwargs...,
)
    dataset = group["data"]
    count = dataset[1]
    return HDF5VectorOfSingletonTypes{el_type}(dataset, count)
end

Base.length(arr::HDF5VectorOfSingletonTypes) = arr.count
supports_setindex(::HDF5VectorOfSingletonTypes) = true

function Base.setindex!(arr::HDF5VectorOfSingletonTypes{T}, el::T, k::Int) where {T}
    checkbounds(arr, k)
    return el
end
function Base.getindex(arr::HDF5VectorOfSingletonTypes{T}, k::Int) where {T}
    if k <= 0 || k > arr.count
        error("Index $k was out of bounds: [1, $(arr.count)].")
    end
    return singleton_value(T)
end
function Base.collect(arr::HDF5VectorOfSingletonTypes{T}) where {T}
    return fill(singleton_value(T), arr.count)
end

function Base.push!(arr::HDF5VectorOfSingletonTypes{T}, el::T) where {T}
    next_count = arr.count + 1
    arr.dataset[1] = next_count # We just store the length.
    arr.count = next_count
    return arr
end

#############################
# HDF5VectorOfArrayishTypes #
#############################

"""
Used to stack fixed-size array-like values in one multidimensional HDF5 dataset. `datatype`
is the scalar HDF5 representation and `dims` contains the dimensions of one vector element.
"""
struct ArrayStorageStyle{HT, ND} <: AbstractHDF5VectorStorageStyle
    datatype::Type{HT}
    dims::NTuple{ND, Int64}
end

function fixed_array_storage_style(element_type, dims; kwargs...)
    datatype = array_element_datatype(element_type; kwargs...)
    if isnothing(datatype)
        return CompositeStorageStyle()
    end
    return ArrayStorageStyle(datatype, dims)
end

# Here, T is the type of each array-like element. The field types of D encode the element
# dimensions, as in Tuple{D1, D2, D3}. DT is the HDF5 datatype used for storage.
mutable struct HDF5VectorOfArrayishTypes{T, D, DT} <: AbstractHDF5Vector{T}
    dataset::HDF5.Dataset
    datatype::Type{DT}
    count::Int64
end

function create_hdf5_vector(
    style::ArrayStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    arrayish_el_type;
    chunk_length,
    portable,
    kwargs...,
)
    el_dims = style.dims
    datatype = style.datatype
    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, arrayish_el_type; dims = el_dims, portable)
    vector_dims = (el_dims..., 0) # Last dimension is 0 until we start writing to it.
    max_dims = (el_dims..., -1,) # Last dimension can grow forever.
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", datatype, dataspace; chunk = (el_dims..., chunk_length,))
    return HDF5VectorOfArrayishTypes{arrayish_el_type, Tuple{el_dims...,}, datatype}(dataset, datatype, 0)
end

function copy_to_hdf5_vector(
    style::ArrayStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    chunk_length,
    portable,
    kwargs...,
)

    # Make a big array with the deconstructed values from the collection before changing
    # the HDF5 file. A conversion or dimension error should not leave a partially created
    # vector behind.
    arrayish_el_type = eltype(collection) # like Vector{Int64} or SVector{3, Float64}
    el_dims = style.dims
    datatype = style.datatype # Like Int64 or Float64
    n = length(collection)
    type = HDF5VectorOfArrayishTypes{arrayish_el_type, Tuple{el_dims...,}, datatype}
    big_array = Array{datatype}(undef, (el_dims..., n))
    for (k, el) in enumerate(collection)
        validate_arrayish_element(el_dims, el)
        big_array[(Colon() for _ in el_dims)..., k] .= deconstruct(type, el)
    end

    # Set up the group and dataset with the current size and the ability to grow.
    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, arrayish_el_type; dims = el_dims, portable)
    vector_dims = (el_dims..., n)
    max_dims = (el_dims..., -1,) # Last dimension can grow forever.
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", datatype, dataspace; chunk = (el_dims..., chunk_length,))

    # Add the data.
    this_group["data"][(Colon() for _ in el_dims)..., :] = big_array

    return type(dataset, datatype, n)

end

function load_hdf5_vector(
    style::ArrayStorageStyle,
    group::HDF5.Group,
    el_type;
    kwargs...,
)
    dataset = group["data"]
    datatype = style.datatype
    el_dims = style.dims # size(dataset)[1:end-1]
    count = size(dataset)[end]
    return HDF5VectorOfArrayishTypes{el_type, Tuple{el_dims...,}, datatype}(dataset, datatype, count)
end

@inline colons(D) = Tuple(Colon() for _ in fieldtypes(D))

Base.length(arr::HDF5VectorOfArrayishTypes) = arr.count

supports_setindex(::HDF5VectorOfArrayishTypes) = true

# ArrayStorageStyle places every vector element in a fixed-size frame of one HDF5 dataset.
# Tuple and StaticArray dimensions are fixed by their types, but separate Array values with
# the same type can have different dimensions. Those dynamic arrays therefore need an exact
# size check before HDF5Vectors copies their values into a frame. In particular, this check
# prevents Julia's broadcasting rules from silently expanding a smaller array during a bulk
# copy.

# Bulk copies use the expected-dimensions form while preparing values, before an HDF5 vector
# or destination group exists. Other array-like types need no runtime validation because
# their dimensions cannot vary without changing their types.
validate_arrayish_element(::Tuple, el) = el

function validate_arrayish_element(expected_dims::Tuple, el::Array)

    actual_dims = size(el)
    if actual_dims != expected_dims
        throw(DimensionMismatch(
            "Expected an element with dimensions $expected_dims, but got $actual_dims.",
        ))
    end
    return el

end

# push! and setindex! already have an HDF5 vector. Its `D` parameter records the dimensions
# of one frame, so this form retrieves those dimensions and shares the check used by bulk
# copies.
function validate_arrayish_element(
    arr::HDF5VectorOfArrayishTypes{T, D},
    el::T,
) where {T, D}
    return validate_arrayish_element(fieldtypes(D), el)
end

@inline function construct_arrayish_elements(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    elements,
) where {T, D, DT}
    element_type = eltype(T)
    if element_type === DT
        return elements
    end
    elemental_vector_type = HDF5VectorOfElementalTypes{element_type, DT}
    return map(el -> construct(elemental_vector_type, el), elements)
end

@inline function deconstruct_arrayish_elements(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    elements,
) where {T, D, DT}
    element_type = eltype(T)
    if element_type === DT
        return elements
    end
    elemental_vector_type = HDF5VectorOfElementalTypes{element_type, DT}
    return map(el -> deconstruct(elemental_vector_type, el), elements)
end

function Base.setindex!(arr::HDF5VectorOfArrayishTypes{T, D}, el::T, k::Int) where {T, D}
    validate_arrayish_element(arr, el)
    arr.dataset[colons(D)..., k] = deconstruct(typeof(arr), el)
    return el
end

function Base.getindex(arr::HDF5VectorOfArrayishTypes{T, D, DT}, k::Int) where {T, D, DT}
    construct(typeof(arr), read(arr.dataset, DT, colons(D)..., k))
end

function copy_each_frame_and_construct!(arr, collected::Vector{T}, data::Array{ET, N}, n) where {T, ET, N}
    for k in 1:n
        v = view(data, (Colon() for _ in 1:N-1)..., k)
        collected[k] = construct(typeof(arr), v)
    end
end

function Base.collect(arr::HDF5VectorOfArrayishTypes{T, D, DT}) where {T, D, DT}
    data = read(arr.dataset, DT, colons(D)..., 1:arr.count)
    collected = Vector{T}(undef, arr.count)
    copy_each_frame_and_construct!(arr, collected, data, arr.count)
    return collected
end

function Base.push!(arr::HDF5VectorOfArrayishTypes{T, D}, el::T) where {T, D}
    validate_arrayish_element(arr, el)
    next_count = arr.count + 1
    HDF5.set_extent_dims(arr.dataset, (fieldtypes(D)..., next_count,))
    arr.dataset[colons(D)..., next_count] = deconstruct(typeof(arr), el)
    arr.count = next_count
    return arr
end

##################################
# HDF5VectorWithByteArrayStorage #
##################################

import Serialization

"""
Used to store each vector element with Julia's `Serialization` format. The serialized bytes
are Julia-specific, and this representation does not support replacing existing elements.
"""
struct ByteArrayStorageStyle <: AbstractHDF5VectorStorageStyle end

# Serialized values are concatenated in `storage`; `stops` records the cumulative ending
# byte position of each value.
mutable struct HDF5VectorWithByteArrayStorage{T} <: AbstractHDF5Vector{T}
    storage::HDF5VectorOfElementalTypes{UInt8, UInt8}
    stops::HDF5VectorOfElementalTypes{Int64, Int64}
end

function create_hdf5_vector(
    style::ByteArrayStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type;
    portable,
    kwargs...,
)
    this_group = create_group(group, name)
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")
    return HDF5VectorWithByteArrayStorage{el_type}(
        create_hdf5_vector(data_group, "bytes", UInt8; kwargs...),
        create_hdf5_vector(data_group, "stops", Int64; kwargs...),
    )
end

function copy_to_hdf5_vector(
    style::ByteArrayStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    chunk_length,
    portable,
    kwargs...,
)

    # Serialize the full collection before changing the HDF5 file. Each call to serialize
    # writes an independently deserializable value into the shared byte buffer.
    io = IOBuffer()
    stops = Int64[]
    sizehint!(stops, length(collection))
    for el in collection
        Serialization.serialize(io, el)
        push!(stops, Int64(position(io)))
    end
    bytes = take!(io)

    # Store the concatenated bytes and cumulative end positions using the elemental bulk
    # copy path rather than pushing each byte and stop individually.
    el_type = eltype(collection)
    this_group = create_group(group, name)
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")
    return HDF5VectorWithByteArrayStorage{el_type}(
        copy_to_hdf5_vector(data_group, "bytes", bytes; chunk_length),
        copy_to_hdf5_vector(data_group, "stops", stops; chunk_length),
    )

end

function load_hdf5_vector(
    ::ByteArrayStorageStyle,
    group::HDF5.Group,
    el_type;
    kwargs...,
)
    storage = load_hdf5_vector(group["data"]["bytes"], UInt8)
    stops = load_hdf5_vector(group["data"]["stops"], Int64)
    byte_count = length(storage)
    if isempty(stops)
        if !iszero(byte_count)
            throw(DimensionMismatch(
                "Serialized storage contains $byte_count bytes but no element stops.",
            ))
        end
    else
        final_stop = stops[end]
        if final_stop != byte_count
            throw(DimensionMismatch(
                "The final serialized element stop is $final_stop, but the byte storage " *
                "contains $byte_count bytes.",
            ))
        end
    end
    return HDF5VectorWithByteArrayStorage{el_type}(storage, stops)
end

Base.length(arr::HDF5VectorWithByteArrayStorage) = length(arr.stops)

function Base.push!(arr::HDF5VectorWithByteArrayStorage{T}, el::T) where {T}
    io = IOBuffer()
    Serialization.serialize(io, el)
    count = 0
    seekstart(io)
    while !eof(io)
        push!(arr.storage, read(io, UInt8))
        count += 1
    end
    stop = length(arr.stops) == 0 ? count : arr.stops[end] + count
    push!(arr.stops, stop)
    return arr
end

function Base.setindex!(arr::HDF5VectorWithByteArrayStorage, el, k)
    # Replacing one value could change its byte length and require rewriting all later stop
    # positions, so byte-array storage is append-only.
    error("setindex! is not supported for HDF5VectorWithByteArrayStorage.")
end

function deserialize_from_vector!(io, byte_array::Vector{UInt8}, start, stop)
    seekstart(io)
    for k in start : stop
        write(io, byte_array[k])
    end
    seekstart(io)
    return Serialization.deserialize(io)
end

function Base.getindex(arr::HDF5VectorWithByteArrayStorage{T}, k::Int) where {T}
    stop = arr.stops[k]
    start = k == 1 ? 1 : arr.stops[k-1] + 1
    range = Int64(start) : Int64(stop)
    return Serialization.deserialize(IOBuffer(read(arr.storage.dataset, UInt8, range)))
end

function Base.collect(arr::HDF5VectorWithByteArrayStorage{T}) where {T}
    data = collect(arr.storage)
    stops = collect(arr.stops)
    io = IOBuffer()
    return T[
        deserialize_from_vector!(io, data, (k == 1 ? 1 : stops[k-1]+1), stops[k])
        for k in eachindex(stops)
    ]
end

##############################
# HDF5VectorOfCompositeTypes #
##############################

mutable struct HDF5VectorOfCompositeTypes{T} <: AbstractHDF5Vector{T}
    arrays::Vector{AbstractHDF5Vector}
    count::Int64
end

function create_hdf5_vector(
    style::CompositeStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type::Type{T};
    chunk_length,
    portable,
    kwargs...,
) where {T}
    this_group = create_group(group, name)
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")
    return HDF5VectorOfCompositeTypes{T}(
        [
            create_hdf5_vector(
                data_group,
                string(fn),
                ft;
                chunk_length,
                portable,
            ) for (fn, ft) in zip(fieldnames(T), fieldtypes(T))
        ],
        0,
    )
end

function copy_to_hdf5_vector(
    style::CompositeStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    chunk_length,
    portable,
    kwargs...,
)

    el_type = eltype(collection)
    n = length(collection)

    # Composite deconstruct methods return the stored value for each declared field, in
    # field order. The default method simply reads the fields, but a custom method can
    # transform or derive them. Calling deconstruct exactly once for each element keeps
    # bulk copying consistent with push! and avoids repeating work or observable side
    # effects for every field.
    vector_type = HDF5VectorOfCompositeTypes{el_type}
    deconstructed_values = [deconstruct(vector_type, el) for el in collection]

    this_group = create_group(group, name)
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")

    # The deconstructed values form rows: one row for each element and one entry for each
    # field. Each child HDF5 vector instead needs a column containing one field from every
    # element. The field index performs that row-to-column rearrangement.
    #
    # Construct each column with its declared field type. This matters for abstract fields:
    # using only the runtime values could select a narrower storage style that would differ
    # from the declared style selected again when the parent vector is loaded.
    return HDF5VectorOfCompositeTypes{el_type}(
        [
            copy_to_hdf5_vector(
                data_group,
                string(field_name),
                field_type[
                    values[field_index] for values in deconstructed_values
                ];
                chunk_length,
                portable,
            ) for (field_index, (field_name, field_type)) in enumerate(
                zip(fieldnames(el_type), fieldtypes(el_type)),
            )
        ],
        n,
    )

end

function load_hdf5_vector(
    ::CompositeStorageStyle,
    group::HDF5.Group,
    el_type;
    kwargs...,
)
    arrays = [
        load_hdf5_vector(group["data"][string(fn)], ft)
        for (fn, ft) in zip(fieldnames(el_type), fieldtypes(el_type))
    ]
    count = isempty(arrays) ? 0 : length(first(arrays))
    for (field_name, array) in zip(fieldnames(el_type), arrays)
        field_count = length(array)
        if field_count != count
            throw(DimensionMismatch(
                "Composite storage for $el_type contains $field_count values for field " *
                "$field_name, but there are only arrays for $count values.",
            ))
        end
    end
    return HDF5VectorOfCompositeTypes{el_type}(arrays, count)
end

Base.length(arr::HDF5VectorOfCompositeTypes) = arr.count

function supports_setindex(arr::HDF5VectorOfCompositeTypes)
    return all(supports_setindex, arr.arrays)
end

function Base.setindex!(arr::HDF5VectorOfCompositeTypes{T}, el::T, k::Int) where {T}

    checkbounds(arr, k)
    if !supports_setindex(arr)
        throw(ArgumentError(
            "setindex! is not supported because at least one field uses " *
            "append-only storage.",
        ))
    end

    # Replacement must use the same representation as push! and bulk copy. Reading fields
    # with getproperty here would bypass a custom deconstruct method and could store a value
    # that construct later interprets incorrectly. Deconstruct once before the first child
    # write so the transformation itself cannot fail after an earlier field was replaced.
    values = deconstruct(typeof(arr), el)
    for (sub_array, value) in zip(arr.arrays, values)
        setindex!(sub_array, value, k)
    end
    return el

end

function Base.push!(arr::HDF5VectorOfCompositeTypes{T}, el::T) where {T}

    # Each field has its own child HDF5 vector. deconstruct supplies the values for those
    # children in field order and allows custom types to transform their stored fields.
    values = deconstruct(typeof(arr), el)
    for (sub_array, value) in zip(arr.arrays, values)
        push!(sub_array, value)
    end
    arr.count += 1
    return arr

end

# Default composite reconstruction calls the element type with its field values. Types that
# require a different constructor can define a more specific `construct` method.
function Base.getindex(arr::HDF5VectorOfCompositeTypes{T}, k::Int) where {T}
    return construct(typeof(arr), ((getindex(sub_array, k) for sub_array in arr.arrays)...,))
end

function Base.collect(arr::HDF5VectorOfCompositeTypes{T}) where {T}
    collected_arrays = map(collect, arr.arrays)
    return [construct(typeof(arr), els) for els in zip(collected_arrays...)]
end

# The `el` here will always be a tuple of the values for the fields.
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T} = T(el...,)
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: Tuple} = el
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: NamedTuple} = T(el)

# Similarly, to deconstruct, get a tuple of the values.
function deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T}
    return Tuple(getproperty(el, fn) for fn in fieldnames(T))
end

######################
# JSON Serialization #
######################

"""
Used to store each vector element as a JSON string through JSON3. JSON3 needs to be loaded
before a vector with this style is created or loaded.
"""
struct JSONStorageStyle <: AbstractHDF5VectorStorageStyle end

###############
# Basic Types #
###############

storage_style(el_type::Type{<:hdf5_scalar_types}; kw...) = ElementalStorageStyle(el_type)
construct(::Type{HDF5VectorOfElementalTypes{T, DT}}, el::T) where {T, DT} = el
deconstruct(::Type{HDF5VectorOfElementalTypes{T, DT}}, el::T) where {T, DT} = el

#########################
# Other Elemental Types #
#########################


##########
# String #
##########

# Strings use ElementalStorageStyle as scalar values, but this package does not store them
# as scalar elements of multidimensional array-like datasets.
array_element_datatype(::Type{String}; kwargs...) = nothing
storage_style(el_type::Type{String}; kwargs...) = ElementalStorageStyle(el_type)
construct(::Type{HDF5VectorOfElementalTypes{String, DT}}, el::String) where {DT} = el
deconstruct(::Type{HDF5VectorOfElementalTypes{String, DT}}, el::String) where {DT} = el

########
# Char #
########

# Int32 is an HDF5-native, fixed-width type large enough for every Unicode code point.
storage_style(el_type::Type{<:Char}; kwargs...) = ElementalStorageStyle(Int32)
construct(::Type{HDF5VectorOfElementalTypes{Char, DT}}, el::Int32) where {DT} = Char(el)
deconstruct(::Type{HDF5VectorOfElementalTypes{Char, DT}}, el::Char) where {DT} = Int32(el)

##########
# Symbol #
##########

array_element_datatype(::Type{Symbol}; kwargs...) = nothing # Same as for strings.
storage_style(el_type::Type{Symbol}; kwargs...) = ElementalStorageStyle(String)
construct(::Type{HDF5VectorOfElementalTypes{Symbol, DT}}, el::String) where {DT} = Symbol(el)
deconstruct(::Type{HDF5VectorOfElementalTypes{Symbol, DT}}, el::Symbol) where {DT} = string(el)

########
# Enum #
########

function storage_style(::Type{<:Enum{BT}}; kwargs...) where {BT <: hdf5_scalar_types}
    return ElementalStorageStyle(BT)
end
function construct(
    ::Type{HDF5VectorOfElementalTypes{T, BT}},
    el::BT,
) where {BT, T <: Enum{BT}}
    return T(el)
end
function deconstruct(
    ::Type{HDF5VectorOfElementalTypes{T, BT}},
    el::T,
) where {BT, T <: Enum{BT}}
    return BT(el)
end

##########
# NTuple #
##########

function storage_style(t::Type{Tuple{}}; dims = nothing, kwargs...)
    validate_fixed_dims(dims, (0,))
    return SingletonStorageStyle()
end
function storage_style(t::Type{NTuple{N, T}}; dims = nothing, kwargs...) where {N, T}
    validate_fixed_dims(dims, (N,))
    return fixed_array_storage_style(T, (N,); kwargs...)
end

# By constructing explicitly from 1:N, this is all known at compile time, so this doesn't
# allocate.
function construct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, elements) where {T <: NTuple, D, DT}
    N = fieldtype(D, 1)
    converted_elements = construct_arrayish_elements(
        HDF5VectorOfArrayishTypes{T, D, DT},
        elements,
    )
    return ((converted_elements[i] for i in 1:N)...,)
end
function construct(::Type{HDF5VectorOfCompositeTypes{T}}, elements) where {T <: NTuple}
    return elements # Already a tuple.
end

# We use an SVector here because the HDF5 library doesn't know how to take an NTuple as if
# it were a vector.
function deconstruct(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    el::T,
) where {N, ET, T <: NTuple{N, ET}, D, DT}
    elements = deconstruct_arrayish_elements(type, el)
    return SVector{N, DT}(elements)
end
function deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: NTuple}
    return el # Already a tuple.
end

#########
# Array #
#########

# A single element of the vector we're setting up will be an Array. If that array's
# dimensions are known and it stores elemental types, then we can use our efficient
# ArrayStorageStyle. Check whether the element type has an elemental representation.
function storage_style(::Type{<:Array{T, N}}; dims = nothing, kwargs...) where {T, N}
    dims = validate_dims(dims, N)
    if !isnothing(dims)
        datatype = array_element_datatype(T; kwargs...)
        if !isnothing(datatype)
            return ArrayStorageStyle(datatype, dims)
        end
    end
    return ByteArrayStorageStyle() # We don't otherwise know how to store this.
end

function construct(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    elements,
) where {T <: Array, D, DT}
    elements = construct_arrayish_elements(type, elements)
    return elements isa T ? elements : collect(elements)
end
function deconstruct(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    el::T,
) where {T <: Array, D, DT}
    return deconstruct_arrayish_elements(type, el)
end

# Static arrays share their array-like and composite conversions. Their storage-style
# methods remain separate because their dimensions have different type representations.
function construct(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    elements,
) where {T <: StaticArray, D, DT}
    return T(construct_arrayish_elements(type, elements))
end
function deconstruct(
    type::Type{HDF5VectorOfArrayishTypes{T, D, DT}},
    el::T,
) where {T <: StaticArray, D, DT}
    return deconstruct_arrayish_elements(type, el)
end
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: StaticArray} = T(el...)
deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: StaticArray} = (el.data,)

###########
# SVector #
###########

function storage_style(t::Type{SVector{N, T}}; dims = nothing, kwargs...) where {N, T}
    validate_fixed_dims(dims, (N,))
    if N == 0
        return SingletonStorageStyle()
    end
    return fixed_array_storage_style(T, (N,); kwargs...)
end

###########
# SMatrix #
###########

using StaticArrays: SMatrix

function storage_style(::Type{SMatrix{M, N, T, L}}; dims = nothing, kwargs...) where {M, N, T, L}
    validate_fixed_dims(dims, (M, N))
    if L == 0
        return SingletonStorageStyle()
    end
    return fixed_array_storage_style(T, (M, N); kwargs...)
end

##########
# SArray #
##########

using StaticArrays: SArray

function storage_style(::Type{SArray{S, T, D, L}}; dims = nothing, kwargs...) where {S, T, D, L}
    dims = validate_fixed_dims(dims, fieldtypes(S))
    if L == 0
        return SingletonStorageStyle()
    end
    return fixed_array_storage_style(T, dims; kwargs...)
end

end # module HDF5Vectors
