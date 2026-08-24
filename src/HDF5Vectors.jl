"""
This module implements an AbstractVector whose underlying data is stored in an HDF5 file.

It generally can store vectors of elements with fixed sizes, where that element is one of:

* HDF5-compatible numeric type (signed/unsigned ints and floats)
* Enum
* SVector, SMatrix, and SArray of numeric types
* Tuple of numeric types
* bits-type consisting of any types on this list
* general composite type consisting of any types on this list
* Vector, Matrix, and Array of any types on this list, as long as their dimensions are
  always the same
* String
* reconstructible singleton types

Further, it can serialize types to bytes or strings and store those in the HDF5 file. This
allows it to store:

* Custom types explicitly assigned a serialization storage style
* Vector, Matrix, and Array values whose dimensions are not declared or vary between
  elements

It fulfills the general AbstractVector interface. Note, however, that iterating directly is
slow; for far better speed, iterate on `iterable(arr)`.
"""
module HDF5Vectors

export AbstractHDF5Vector, create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector, iterable

using HDF5
using StaticArrays: SVector

# See https://juliaio.github.io/HDF5.jl/stable/#Supported-data-types
const hdf5_scalar_types = Union{Bool, UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64}

##################
# Storage Styles #
##################

"""
An abstract type intended as a parent for all HDF5 vector storage styles.
"""
abstract type AbstractHDF5VectorStorageStyle end

"""
Used to store "elemental" types -- types that HDF5 can natively understand, including:

* Int8, Int16, Int32, and Int64 (and unsigned forms)
* Float32 and Float64
* Enum
* Char
* Bits-type structs
* String
"""
struct ElementalStorageStyle{HT} <: AbstractHDF5VectorStorageStyle
    datatype::Type{HT}
end

"""
Used to store composite structured information, like:

* General tuple of types on this list
* General named tuple of types on this list
* General struct of types on this list
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

# These are the only functions types will have to implement to use ElementalStorageStyle
# or ArrayStorageStyle

# Types that aren't native HDF5 scalars but that are bits-types can still be stored using
# the elemental storage type, but that's not portable, so this function considers
# portability before deciding to store non-native types as elemental or composite.
"""
    storage_style(el_type::Type; kwargs...)

Returns the storage style intended for this type. Available styles include:

* `ElementalStorageStyle` for scalars or non-portable bits-type structs
* `SingletonStorageStyle` for types that have exactly one possible value
* `ArrayStorageStyle` for arrays of known, consistent dimensions holding elemental types
* `CompositeStorageStyle` for general structs
* `ByteArrayStorageStyle` for Julia serialization
* `JSONStorageStyle` for serializing types to JSON strings

The default storage style for scalars and "non-portable" bits-type structs (more
below) is `ElementalStorageStyle`. For vectors with known dimensions, `ArrayStorageStyle`
is the default. Singleton types use `SingletonStorageStyle`. For other structs (either
non-bits-types or "portable"), the default is `CompositeStorageStyle`. Nonconcrete types and
arrays without known dimensions default to `ByteArrayStorageStyle`. Unsupported primitive
types produce an error unless the user explicitly defines another storage style.

The storage style is selected again from the element type and stored options when a vector
is loaded. A custom `storage_style` method must therefore make a consistent choice from
those inputs. The style-taking storage methods are implementation hooks, not per-vector
overrides for selecting a different representation.

Array storage results in HDF5 files where the dataset has the dimensions of each element,
plus one added dimension. For instance, if each element to be stored is an m-by-n array,
then the HDF5 file will contain an m-by-n-by-p array, where element `k` is `[:, :, k]`.

Structs can be stored in a "portable" way. For the a struct defined as:

```
struct MyType
    a::Int64
    b::Float64
end
```

the resulting HDF5 file would look like so:

```
/my_group/my_vector/arrays/a # a 1D array of Int64
/my_group/my_vector/arrays/b # a 1D array of Float64
```

This format is called "portable" because it is easy to interpret this dataset outside of
Julia.

"Portability" is controlled by the `portable` keyword argument. When this is false, the
above struct would be stored as:

```
/my_group/my_vector # a 1D array of custom type inferred from MyType
```

This uses the HDF5 type system via the HDF5.jl package to encode the type. The underlying
data can still be interpreted outside of Julia, but it requires substantially more code to
interpret the type information in a useful way. If you are _only_ interested in loading
the HDF5 in Julia, use `portable = false`, and the resulting storage will be faster. (Note
that non-bits types cannot use the HDF5 type system and hence will always use the portable
form.)

When the elements to be stored are themselves vectors, matrices, or arrays of known
dimension, the user should provide those dimensions via the `dims` keyword argument.
Otherwise, since the dimensions of an array are not known from its type, and it's not known
if the user _intends_ for dimensions to be consistent over time or not,

Keyword arguments:

* `portable`: When true (the default), composite types like structs will be stored in a
  slower but more portable way. (For other types, this argument is ignored.)
* `dims`: Sets the dimensions of Array types (otherwise, ignored), such as (3, 4) when each
  element is a 3-by-4 matrix.

Users can add a `storage_style` method for their custom types to allow them to express how
their types out to be stored. E.g., if a type should always be serialized, then this would
instruct Julia to use serialization to a byte array for the give type:

```
HDF5Vectors.storage_style(::Type{MyType}; kwargs...) = HDF5Vectors.ByteArrayStorageStyle()
```
"""
function storage_style(el_type::Type; portable = true, kwargs...)

    # The HDF5-native types don't even get here. They have their own storage_style. If we're
    # here, then we have no other specified storage style to use, and we need to figure out
    # a safe one.

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

            # We can log bits types as elementals, but that's obviously not portable, so
            # only do this if the user doesn't care to have a portable HDF5 file.
            return ElementalStorageStyle(el_type)

        else

            # If it's not one of those special cases, we can likely log it as a composite
            # style.
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

Create the appropriate Julia value from the raw element `el` retrieved from the
HDF5 dataset. The behaviour is determined by the storage style associated with `type`.
This generic definition is a placeholder; concrete storage implementations overload
`construct` for their particular container types and element representations.
"""
function construct end

"""
    deconstruct(type::Type, el)

Convert the Julia value `el` into the representation stored in the HDF5 file. The
storage style associated with `type` chooses how the conversion is performed. Like
`construct`, this generic definition is a no-op; storage backends provide concrete
methods.
"""
function deconstruct end

######################
# AbstractHDF5Vector #
######################

"""
An abstract type intended as the parent for all type of HDF5 vectors. Subtypes should have
a corresponding storage style and implement [`create_hdf5_vector`](@ref), and at least these
parts of the AbstractArray interface: `length`, `setindex!`, `push!`, `getindex`, and
`collect`.
"""
abstract type AbstractHDF5Vector{T} <: AbstractVector{T} end

Base.eltype(::Type{<:AbstractHDF5Vector{ET}}) where {ET} = ET
Base.size(arr::AbstractHDF5Vector) = (length(arr),)
Base.similar(::AbstractHDF5Vector{T}, ::Type{T}, dims::Dims) where {T} = Vector{T}(undef, dims)
Base.IndexStyle(::Type{<:AbstractHDF5Vector}) = IndexLinear()
Base.broadcastable(arr::AbstractHDF5Vector) = collect(arr)
# Base.BroadcastStyle(::Type{SrcType}) = SrcStyle()
# Base.similar(bc::Broadcasted{DestStyle}, ::Type{ElType})

# This should take care of operations like `sum` and `mean`.
function Base.mapreduce(f, op, arr::AbstractHDF5Vector; kwargs...)
    return mapreduce(f, op, iterable(arr); kwargs...)
end
# Some things that don't use mapreduce: findmax/min, argmax/min, any, all, count

# Use the iterable form rather than trying to iterate via getindex.
Base.map(f, arr::AbstractHDF5Vector) = map(f, iterable(arr))

# Range and vector indexing should behave like any other AbstractVector and return a plain
# Julia Vector. We intentionally construct this from scalar indexing so all storage backends
# (elemental, array-ish, composite, serialized) share one consistent behavior.
function Base.getindex(arr::AbstractHDF5Vector, k::AbstractRange{<:Integer})
    return [arr[j] for j in k]
end
function Base.getindex(arr::AbstractHDF5Vector, k::AbstractVector{<:Integer})
    return [arr[j] for j in k]
end
Base.getindex(arr::AbstractHDF5Vector, ::Colon) = collect(arr)

abstract type AbstractHDF5VectorIterator{T} end

# This loads all of the data up front and then iterates over it, but we could make a
# different kind of iterator later that loads chunks and reads incrementally.
struct HDF5VectorIterator{T} <: AbstractHDF5VectorIterator{T}
    data::Vector{T} # Implementation detail (not for public consumption)
    count::Int64
end

# This could store what chunk number we're on, etc.
struct HDF5VectorIteratorState
    index::Int64
end

"""
    iterable(arr::AbstractHDF5Vector)

Returns an iterable type corresponding to the given HDF5 vector. This is generally much
faster than iterating on the vector directly. That is, instead of `[f(el) for el in arr]`,
it is much faster to use `[f(el) for el in iterable(arr)]`.
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

# If we just let the HDF5Arrays have a cache, then iteration (with a mutable iterator!)
# works efficiently. But if we don't want to fill up RAM with all of the things we've cached
# then we'll need to clear the cache, which is an extra step. I'm really not sure we want
# cache.

# If we want a fallback `iterate` behavior...

# # This allocates like crazy because it's a non-bits-type, so the creation of these
# # requires allocation. It's not _that_ slow, but iterating on these arrays is the slowest
# # way to work with them. Iterate over the result of `iterable` instead.
# struct HDF5ArrayIteratorState{T}
#     data::Vector{T}
#     index::Int64
# end
# function Base.iterate(arr::HDF5VectorOfHDF5NativeType{T}) where {T}
#     data = collect(arr)
#     el, internal_state = iterate(data)
#     return (el, HDF5ArrayIteratorState{T}(data, internal_state))
# end
# function Base.iterate(arr::HDF5VectorOfHDF5NativeType{T}, state::HDF5ArrayIteratorState{T})::Union{Nothing, Tuple{T, HDF5ArrayIteratorState{T}}} where {T}
#     if state.index > arr.count
#         return nothing
#     end
#     itr_out = iterate(state.data, state.index)
#     return (itr_out[1], HDF5ArrayIteratorState{T}(state.data, itr_out[2]))
# end

# I don't think we need these unless the iterator itself is stateful.
# Base.isdone(arr::HDF5VectorOfHDF5NativeType) = arr.count == 0
# Base.isdone(::HDF5VectorOfHDF5NativeType, ::Nothing) = true
# Base.isdone(::HDF5VectorOfHDF5NativeType, state::HDF5ArrayIteratorState) = isdone(state.data, state.index)

# This seems inefficient, but this is used rarely.
function serialize_to_byte_array(x)
    io = IOBuffer() # Will use UInt8 by default.
    Serialization.serialize(io, x)
    return take!(io)
end

function deserialize_from_byte_array(x)
    io = IOBuffer(x)
    return Serialization.deserialize(io)
end

# function get_storage_dimensions(style::AbstractHDF5VectorStorageStyle)
#     return ()
# end

function store_metadata(style::AbstractHDF5VectorStorageStyle, group, el_type; dims = nothing, portable)
    metadata_group = HDF5.create_group(group, "metadata")
    metadata_group["type"] = string(el_type)
    metadata_group["serialized_type"] = serialize_to_byte_array(el_type)
    metadata_group["dimensions_are_constant"] = !isnothing(dims)
    metadata_group["dimensions"] = isnothing(dims) ? Int64[] : Int64[dims...,]
    metadata_group["portable"] = portable
    return metadata_group
end

"""
    create_hdf5_vector(group, name, el_type; kwargs...)

Creates the appropriate HDF5 vector type for the given element type, storing the vector in
the given HDF5 `group`` in a new group/dataset, `name`.

Optional keyword arguments:

* `dims`: Tuple of the dimensions to use for a Vector, Matrix, or Array
* `chunk_length`: Length of chunk to use (default 1000)
* `portable`: True to maximize how "portable" the storage is (default true)
"""
function create_hdf5_vector(
    group, name, el_type;
    dims = nothing, chunk_length = 1000, portable = true,
)
    return create_hdf5_vector(
        storage_style(el_type; dims, portable),
        group, name, el_type;
        dims, chunk_length, portable,
    )
end

"""
    load_hdf5_vector(group; kwargs...)

Reconstruct an HDF5 vector from a group created by `create_hdf5_vector`.  The
metadata stored in the group (type, dimensions, portability) is used to determine
which specific vector implementation to instantiate.  This form takes only the
`group` and pulls the element type from the metadata; the optional keyword
arguments are forwarded to `storage_style` and to the underlying loader.
"""
function load_hdf5_vector(group; kwargs...)
    metadata_group = group["metadata"]
    el_type = deserialize_from_byte_array(read(metadata_group["serialized_type"]))
    dimensions_are_constant = read(metadata_group["dimensions_are_constant"])
    dims = dimensions_are_constant ? (read(metadata_group["dimensions"])...,) : nothing
    portable = read(metadata_group["portable"])
    return load_hdf5_vector(storage_style(el_type; dims, portable, kwargs...), group, el_type; dims, portable, kwargs...)
end

"""
    load_hdf5_vector(group_or_dataset, el_type; kwargs...)

Reconstruct an HDF5 vector when the caller already knows the element type.
This overload is primarily used when loading a dataset directly (rather than the
parent group) or when the metadata does not reside in the expectation of the
vector type.  The element type is passed explicitly and used to select the
storage style; the remainder of the arguments is forwarded.
"""
function load_hdf5_vector(group_or_dataset, el_type; kwargs...)
    return load_hdf5_vector(storage_style(el_type; kwargs...), group_or_dataset, el_type; kwargs...)
end

function copy_to_hdf5_vector(
    group, name, collection;
    dims = nothing, chunk_length = 1000, portable = true,
)
    return copy_to_hdf5_vector(
        storage_style(eltype(collection); dims, portable),
        group, name, collection;
        dims, chunk_length, portable,
    )
end

# The generic implementation of this just creates the vector and then fills it in one by
# one. This is slow, but it works, and some things like serialization can't really be made
# faster anyway, so it's a useful default method.
function copy_to_hdf5_vector(
    style::AbstractHDF5VectorStorageStyle,
    group,
    name,
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
    create_hdf5_vector(style, group, name, el_type; kwargs...)

Creates the appropriate HDF5 vector type for the given storage style and element type,
storing the vector in the given HDF5 `group`` in a new group/dataset, `name`. This is an
implementation hook for storage backends; users should select a style by defining
[`storage_style`](@ref) for their element type and call the overload without a style.

Optional keyword arguments:

* `dims`: Tuple of the dimensions to use for a Vector, Matrix, or Array
* `chunk_length`: Length of chunk to use (default 1000)
* `portable`: True to maximize how "portable" the storage is (default true)
"""
function create_hdf5_vector(style::AbstractHDF5VectorStorageStyle, group, name, el_type; kwargs...)
    error("There is no implementation of `create_hdf5_vector` for the $(typeof(style)) storage style used for name = $name with el_type = $el_type.")
end

##############################
# HDF5VectorOfElementalTypes #
##############################

# We can implement this set of behavior and use it across a variety of types by exposing
# a few functions that specify how an element type becomes an HDF5 array.

# We could potentially make other structs like this to specialize on scalar types vs vectors
# types, but it's not clear that we need to do that.
mutable struct HDF5VectorOfElementalTypes{T, DT} <: AbstractHDF5Vector{T}
    dataset::HDF5.Dataset
    datatype::Type{DT}
    count::Int64
end

function create_hdf5_vector(style::ElementalStorageStyle, group, name, el_type; chunk_length, portable, kwargs...)
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
function copy_to_hdf5_vector(style::ElementalStorageStyle, group, name, collection; chunk_length, portable, kwargs...)

    # This is basically the same as create_hdf5_vector.
    el_type = eltype(collection)
    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, el_type; portable)
    datatype = style.datatype

    # Here, however, we have the length. Nonetheless, we want this to be able to grow, so we
    # will create it with a dataspace explicitly (rather than just setting
    # this_group["data"] to array).
    n = length(collection)
    vector_dims = (n,)
    max_dims = (-1,) # This can grow forever.
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", datatype, dataspace; chunk = (chunk_length,))

    # To deconstruct, we need to know what type we're deconstructing _for_.
    type = HDF5VectorOfElementalTypes{el_type, datatype}

    # Now deconstruct every element in RAM.
    array = [deconstruct(type, el) for el in collection]

    # Now that we have the array, assign it to the HDF5 file.
    this_group["data"][:] = array

    # Now we have a normal HDF5 vector.
    return HDF5VectorOfElementalTypes{el_type, datatype}(dataset, datatype, n)

end

function load_hdf5_vector(style::ElementalStorageStyle, group, el_type; kwargs...)
    dataset = group["data"]
    datatype = style.datatype # eltype(dataset)
    count = size(dataset)[end]
    return HDF5VectorOfElementalTypes{el_type, datatype}(dataset, datatype, count)
end

Base.length(arr::HDF5VectorOfElementalTypes) = arr.count # Common with HDF5VectorOfArrayishTypes
function Base.setindex!(arr::HDF5VectorOfElementalTypes{T, DT}, el, k) where {T, DT}
    arr.dataset[k] = deconstruct(typeof(arr), el)
end
function Base.getindex(arr::HDF5VectorOfElementalTypes{T, DT}, k::Int) where {T, DT}
    construct(typeof(arr), read(arr.dataset, DT, k))
end
function Base.collect(arr::HDF5VectorOfElementalTypes{T, DT}) where {T, DT}
    data = read(arr.dataset, DT, 1:arr.count)
    return [construct(typeof(arr), el) for el in data]
end

function Base.push!(arr::HDF5VectorOfElementalTypes, el)
    arr.count += 1
    HDF5.set_extent_dims(arr.dataset, (arr.count,))
    arr[arr.count] = el
    return arr
end

function is_elemental(type; kwargs...)
    return isa(storage_style(type; kwargs...), ElementalStorageStyle)
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
    group,
    name,
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
    group,
    name,
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

function load_hdf5_vector(style::SingletonStorageStyle, group, el_type; kwargs...)
    dataset = group["data"]
    count = dataset[1]
    return HDF5VectorOfSingletonTypes{el_type}(dataset, count)
end

Base.length(arr::HDF5VectorOfSingletonTypes) = arr.count
function Base.setindex!(arr::HDF5VectorOfSingletonTypes{T}, el, k::Int) where {T}
    if k <= 0 || k > arr.count
        error("Index $k was out of bounds: [1, $(arr.count)].")
    end
    convert(T, el)
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

function Base.push!(arr::HDF5VectorOfSingletonTypes{T}, el) where {T}
    convert(T, el)
    arr.count += 1
    arr.dataset[1] = arr.count # We just store the length.
    return arr
end

#############################
# HDF5VectorOfArrayishTypes #
#############################

# There are only two differences between the elemental and array types: the array uses
# `colons`, and it constructs from a view into the matrix.

# Potentially, the style itself could encode dimensions and eltype.
struct ArrayStorageStyle{HT, ND} <: AbstractHDF5VectorStorageStyle
    datatype::Type{HT}
    dims::NTuple{ND, Int64}
end

# N is Tuple{D1, D2, D3...}, a Tuple type whose type parameters are "value types".
mutable struct HDF5VectorOfArrayishTypes{T, D, DT} <: AbstractHDF5Vector{T}
    dataset::HDF5.Dataset
    datatype::Type{DT}
    count::Int64
end

function create_hdf5_vector(style::ArrayStorageStyle, group, name, arrayish_el_type; chunk_length, portable, kwargs...)
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

function copy_to_hdf5_vector(style::ArrayStorageStyle, group, name, collection; chunk_length, portable, kwargs...)

    # Set up the group and metadata just like for create_hdf5_vector.
    arrayish_el_type = eltype(collection) # like Vector{Int64} or SVector{3, Float64}
    el_dims = style.dims
    datatype = style.datatype # Like Int64 or Float64
    this_group = HDF5.create_group(group, name)
    store_metadata(style, this_group, arrayish_el_type; dims = el_dims, portable)

    # Set up the dataset with the current size and the ability to grow.
    n = length(collection)
    vector_dims = (el_dims..., n)
    max_dims = (el_dims..., -1,) # Last dimension can grow forever.
    dataspace = HDF5.dataspace(vector_dims, max_dims)
    dataset = create_dataset(this_group, "data", datatype, dataspace; chunk = (el_dims..., chunk_length,))

    # Make a big array with the deconstructed values from the collection.
    type = HDF5VectorOfArrayishTypes{arrayish_el_type, Tuple{el_dims...,}, datatype}
    big_array = Array{style.datatype}(undef, (el_dims..., n))
    for k in eachindex(collection)
        big_array[(Colon() for _ in el_dims)..., k] .= deconstruct(type, collection[k])
    end

    # Add the data.
    this_group["data"][(Colon() for _ in el_dims)..., :] = big_array

    return type(dataset, datatype, n)

end

function load_hdf5_vector(style::ArrayStorageStyle, group, el_type; kwargs...)
    dataset = group["data"]
    datatype = style.datatype
    el_dims = style.dims # size(dataset)[1:end-1]
    count = size(dataset)[end]
    return HDF5VectorOfArrayishTypes{el_type, Tuple{el_dims...,}, datatype}(dataset, datatype, count)
end

@inline colons(D) = Tuple(Colon() for _ in fieldtypes(D))

Base.length(arr::HDF5VectorOfArrayishTypes) = arr.count

function Base.setindex!(arr::HDF5VectorOfArrayishTypes{T, D, DT}, el, k) where {T, D, DT}
    arr.dataset[colons(D)..., k] = deconstruct(typeof(arr), el)
end
function Base.getindex(arr::HDF5VectorOfArrayishTypes{T, D, DT}, k::Int) where {T, D, DT}
    construct(typeof(arr), read(arr.dataset, DT, colons(D)..., k))
end
function copy_each_frame_and_construct!(arr, collected::Vector{T}, data::Array{ET, N}, n) where {T, ET, N}
    for k in 1:n
        v = view(data, (Colon() for _ in 1:N-1)..., k) # view seems to allocate for matrices and above.
        collected[k] = construct(typeof(arr), v)
    end
end
function Base.collect(arr::HDF5VectorOfArrayishTypes{T, D, DT}) where {T, D, DT}
    data = read(arr.dataset, DT, colons(D)..., 1:arr.count)
    collected = Vector{T}(undef, arr.count)
    copy_each_frame_and_construct!(arr, collected, data, arr.count)
    return collected
end

function Base.push!(arr::HDF5VectorOfArrayishTypes{T, D, DT}, el) where {T, D, DT}
    arr.count += 1
    HDF5.set_extent_dims(arr.dataset, (fieldtypes(D)..., arr.count,))
    arr[arr.count] = el
    return arr
end

##################################
# HDF5VectorWithByteArrayStorage #
##################################

import Serialization

# Make a style so that users can apply the style trait to their custom types.
struct ByteArrayStorageStyle <: AbstractHDF5VectorStorageStyle end

# Create a type to handle anything that needs to go to/from JSON. We'll just store a single-
# dataset HDF5 vector of strings inside.
mutable struct HDF5VectorWithByteArrayStorage{T} <: AbstractHDF5Vector{T}
    storage::HDF5VectorOfElementalTypes{UInt8, UInt8}
    stops::HDF5VectorOfElementalTypes{Int64, Int64}
    # We could add the IOBuffer here and always use the same one.
end
function create_hdf5_vector(style::ByteArrayStorageStyle, group, name, el_type; portable, kwargs...)
    this_group = create_group(group, string(name))
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")
    return HDF5VectorWithByteArrayStorage{el_type}(
        create_hdf5_vector(data_group, "bytes", UInt8; kwargs...),
        create_hdf5_vector(data_group, "stops", Int64; kwargs...),
    )
end
# We'll use the generic copy_to_hdf5_vector.
function load_hdf5_vector(style::ByteArrayStorageStyle, group_or_dataset, el_type; kwargs...)
    this_group = group_or_dataset
    return HDF5VectorWithByteArrayStorage{el_type}(
        load_hdf5_vector(this_group["data"]["bytes"], UInt8; kwargs...),
        load_hdf5_vector(this_group["data"]["stops"], Int64; kwargs...),
    )
end
Base.length(arr::HDF5VectorWithByteArrayStorage) = length(arr.stops)
function Base.push!(arr::HDF5VectorWithByteArrayStorage, el)
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
    # To implement this, we'd need to completely redo the byte array and all of the stops.
    # Let's just not support this for serialized types.
    error("setindex! is not supported for HDF5VectorWithByteArrayStorage.")
end
function deserialize_from_vector!(io, byte_array::Vector{UInt8}, start, stop)
    seekstart(io)
    for k in start : stop
        write(io, byte_array[k])
    end
    seekstart(io)
    Serialization.deserialize(io) # This reads everything, resetting the buffer.
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
    arrays::Vector{AbstractHDF5Vector} # Use a Tuple to zip it?
    count::Int64
end

function create_hdf5_vector(style::CompositeStorageStyle, group, name, el_type::Type{T}; chunk_length, portable, kwargs...) where {T}
    this_group = create_group(group, string(name))
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")
    return HDF5VectorOfCompositeTypes{T}(
        [
            create_hdf5_vector(
                data_group,
                string(fn),
                ft; # Since this comes from the _type_, abstract types like NamedTuple may fail here.
                chunk_length,
                portable,
            ) for (fn, ft) in zip(fieldnames(T), fieldtypes(T))
        ],
        0,
    )
end

function copy_to_hdf5_vector(
    style::CompositeStorageStyle,
    group,
    name,
    collection;
    chunk_length,
    portable,
    kwargs...,
)

    el_type = eltype(collection)
    n = length(collection)
    this_group = create_group(group, string(name))
    store_metadata(style, this_group, el_type; portable)
    data_group = create_group(this_group, "data")

    # Use each declared field type for its collection. In particular, an abstract field's
    # runtime values must not narrow the storage style away from the one used when loading.
    return HDF5VectorOfCompositeTypes{el_type}(
        [
            copy_to_hdf5_vector(
                data_group,
                string(fn),
                field_type[getproperty(el, fn) for el in collection];
                chunk_length,
                portable,
            ) for (fn, field_type) in zip(fieldnames(el_type), fieldtypes(el_type))
        ],
        n,
    )

end

function load_hdf5_vector(style::CompositeStorageStyle, group_or_dataset, el_type; kwargs...)
    this_group = group_or_dataset
    arrays = [
        load_hdf5_vector(this_group["data"][string(fn)], ft; kwargs...)
        for (fn, ft) in zip(fieldnames(el_type), fieldtypes(el_type))
    ]
    if isempty(arrays)
        @show el_type
        return HDF5VectorOfCompositeTypes{el_type}(arrays, 0)
    else
        return HDF5VectorOfCompositeTypes{el_type}(arrays, length(first(arrays)))
    end
end

Base.length(arr::HDF5VectorOfCompositeTypes) = arr.count

function Base.setindex!(arr::HDF5VectorOfCompositeTypes{T}, el, k) where {T}
    for (sub_array, fn) in zip(arr.arrays, fieldnames(T))
        setindex!(sub_array, getproperty(el, fn), k)
    end
    return el
end

function Base.push!(arr::HDF5VectorOfCompositeTypes{T}, el) where {T}
    for (sub_array, value) in zip(arr.arrays, deconstruct(typeof(arr), el))
        push!(sub_array, value)
    end
    arr.count += 1
    return arr
end

# This assumes the struct can be created with its individual fields, which isn't perfectly
# general, but what else can we do? Something with StructTypes?
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

# We define the storage style here, but its implementation is in HDF5VectorsJSON3Ext, which
# is only loaded if JSON3 is loaded.
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

# Strings use the ElementalStorageStyle under the hood, but they aren't really "elemental"
# types (you can make an array of tuples of them), so we have to call that out directly
# here.
is_elemental(type::Type{String}; kwargs...) = false
storage_style(el_type::Type{String}; kwargs...) = ElementalStorageStyle(el_type)
construct(::Type{HDF5VectorOfElementalTypes{String, DT}}, el::String) where {DT} = el
deconstruct(::Type{HDF5VectorOfElementalTypes{String, DT}}, el::String) where {DT} = el

########
# Char #
########

storage_style(el_type::Type{<:Char}; kwargs...) = ElementalStorageStyle(Int32) # I don't know why these are Int32 instead of Int.
construct(::Type{HDF5VectorOfElementalTypes{Char, DT}}, el::Int32) where {DT} = Char(el)
deconstruct(::Type{HDF5VectorOfElementalTypes{Char, DT}}, el::Char) where {DT} = Int32(el)

##########
# Symbol #
##########

is_elemental(type::Type{Symbol}; kwargs...) = false # Same as for strings.
storage_style(el_type::Type{Symbol}; kwargs...) = ElementalStorageStyle(String)
construct(::Type{HDF5VectorOfElementalTypes{Symbol, DT}}, el::String) where {DT} = Symbol(el)
deconstruct(::Type{HDF5VectorOfElementalTypes{Symbol, DT}}, el::Symbol) where {DT} = string(el)

########
# Enum #
########

storage_style(el_type::Type{<:Enum}; kwargs...) = ElementalStorageStyle(Int32) # I don't know why these are Int32 instead of Int.
construct(::Type{HDF5VectorOfElementalTypes{T, DT}}, el::Int32) where {T <: Enum, DT} = T(el)
deconstruct(::Type{HDF5VectorOfElementalTypes{T, DT}}, el::Enum) where {T <: Enum, DT} = Int32(el)

##########
# NTuple #
##########

function storage_style(t::Type{Tuple{}}; dims = nothing, kwargs...)
    return SingletonStorageStyle()
end
function storage_style(t::Type{NTuple{N, T}}; dims = nothing, kwargs...) where {N, T}
    if is_elemental(T; kwargs...)
        @assert isnothing(dims) || dims == (N,) "The dimensions of the NTuple ($N) don't match the provided `dims` keyword argument, $dims."
        return ArrayStorageStyle(T, (N,))
    else
        return CompositeStorageStyle()
    end
end

# By constructing explicitly from 1:N, this is all known at compile time, so this doesn't
# allocate.
function construct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, elements) where {T <: NTuple, D, DT}
    N = fieldtype(D, 1)
    return ((elements[i] for i in 1:N)...,)
end
function construct(::Type{HDF5VectorOfCompositeTypes{T}}, elements) where {T <: NTuple}
    return elements # Already a tuple.
end

# We use an SVector here because the HDF5 library doesn't know how to take an NTuple as if
# it were a vector.
function deconstruct(::Type{<:HDF5VectorOfArrayishTypes}, el::NTuple{N, ET}) where {N, ET}
    return SVector{N, ET}(el...,)
end
function deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: NTuple}
    return el # Already a tuple.
end

#########
# Array #
#########

# A single element of the vector we're setting up will be an Array. If that array's
# dimensions are known and it stores elemental types, then we can use our efficient
# ArrayStorageStyle. See if the eltype of the Array shold use the elemental style.
function storage_style(::Type{<:Array{T, N}}; dims = nothing, kwargs...) where {T, N}
    if !isnothing(dims) && is_elemental(T; kwargs...)
        return ArrayStorageStyle(T, dims)
    else
        return ByteArrayStorageStyle() # We don't otherwise know how to store this.
    end
end

construct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, el) where {T <: Array, D, DT} = collect(el)
deconstruct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, el) where {T <: Array, D, DT} = el

###########
# SVector #
###########

function storage_style(t::Type{SVector{N, T}}; dims = nothing, kwargs...) where {N, T}
    if N == 0
        return SingletonStorageStyle()
    elseif is_elemental(T; kwargs...)
        return ArrayStorageStyle(T, (N,))
    else
        return CompositeStorageStyle()
    end
end

construct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, el) where {T <: SVector, D, DT} = T(el)
deconstruct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, el::SVector) where {T <: SVector, D, DT} = el

# When these are composite, we treat them like normal composite types. They have a `data`
# field, and we log that one field, letting the type of the `data` field break down like
# any other composite type.
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: SVector} = T(el...)
deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: SVector} = (el.data,)

###########
# SMatrix #
###########

using StaticArrays: SMatrix

function storage_style(::Type{SMatrix{M, N, T, L}}; dims = nothing, kwargs...) where {M, N, T, L}
    if L == 0
        return SingletonStorageStyle()
    elseif is_elemental(T; kwargs...)
        @assert isnothing(dims) || dims == (M, N) "The dimensions of the SMatrix ($M, $N) don't match the provided `dims` keyword argument, $dims."
        return ArrayStorageStyle(T, (M, N,))
    else
        return CompositeStorageStyle()
    end
end

construct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, el) where {T <: SMatrix, D, DT} = T(el)
deconstruct(::Type{<:HDF5VectorOfArrayishTypes}, el::SMatrix) = el

# When these are composite, we treat them like normal composite types. They have a `data`
# field, and we log that one field, letting the type of the `data` field break down like
# any other composite type.
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: SMatrix} = T(el...)
deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: SMatrix} = (el.data,)

##########
# SArray #
##########

using StaticArrays: SArray

function storage_style(::Type{SArray{S, T, D, L}}; dims = nothing, kwargs...) where {S, T, D, L}
    if L == 0
        return SingletonStorageStyle()
    elseif is_elemental(T; kwargs...)
        @assert isnothing(dims) || dims == fieldtypes(S) "The dimensions of the SArray $(fieldtypes(S))  don't match the provided `dims` keyword argument, $dims."
        dims = fieldtypes(S) # This returns a tuple of numbers because S uses "value types".
        return ArrayStorageStyle(T, dims)
    else
        return CompositeStorageStyle()
    end
end

construct(::Type{HDF5VectorOfArrayishTypes{T, D, DT}}, el) where {T <: SArray, D, DT} = T(el)
deconstruct(::Type{<:HDF5VectorOfArrayishTypes}, el::SArray) = el

# When these are composite, we treat them like normal composite types. They have a `data`
# field, and we log that one field, letting the type of the `data` field break down like
# any other composite type.
construct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: SArray} = T(el...)
deconstruct(::Type{HDF5VectorOfCompositeTypes{T}}, el) where {T <: SArray} = (el.data,)

end # module HDF5Vectors
