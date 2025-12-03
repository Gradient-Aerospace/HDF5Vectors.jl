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

Futher, it can serialize types to bytes or strings and store those in the HDF5 file. This
allows it to store:

* Any type that serializes
* Vector, Matrix, and Array that are different lengths from element to element (via serialization)

It fulfills the general AbstractVector interface. Note, however, that iterating directly is
slow; for far better speed, iterate on `iterable(arr)`.
"""
module HDF5Vectors

export AbstractHDF5Vector, create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector, iterable

using HDF5
using StaticArrays: SVector

# See https://juliaio.github.io/HDF5.jl/stable/#Supported-data-types
const hdf5_scalar_types = Union{UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64}

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

# These are the only functions types will have to implement to use ElementalStorageStyle
# or ArrayStorageStyle

# Types that aren't native HDF5 scalars but that are bits-types can still be stored using
# the elemental storage type, but that's not portable, so this function considers 
# portability before deciding to store non-native types as elemental or composite.
"""
    storage_style(el_type::Type; kwargs...)

Returns the storage style intended for this type. Available styles include:

* `ElementalStorageStyle` for scalars or non-portable bits-type structs
* `ArrayStorageStyle` for arrays of known, consistent dimensions holding elemental types
* `CompositeStorageStyle` for general structs
* `ByteArrayStorageStyle` for arrays of inconsistent dimensions
* `JSONStorageStyle` for serializing types to JSON strings

The default storage style for scalars and "non-portable" bits-type structs (more
below) is `ElementalStorageStyle`. For vectors with known dimensions, `ArrayStorageStyle`
is the default. For other structs (either non-bits-types or "portable"), the default is
`CompositeStorageStyle`. For all other types, `ByteArrayStorageStyle` is the default storage
style.

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
    if isbitstype(el_type) && !portable
        return ElementalStorageStyle(el_type) # Use the Julia type as the HDF5 type.
    else
        return CompositeStorageStyle()
    end
end

"""
    construct(type::Type, el)
TODO: Update.
Given the `el` "element", as stored in the HDF5 file, the constructs the given `type`. What
the element is depends on the storage style associated with the type.
"""
function construct end

"""
    deconstruct(type::Type, el)

TODO: Update.

Decomposes the given `el` element into the datatype used for storage in the HDF5 file. This
depends on the storage style associated with the type of the element.
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
function create_hdf5_vector(group, name, el_type; dims = nothing, chunk_length = 1000, portable = true)
    return create_hdf5_vector(
        storage_style(el_type; dims, portable), 
        group, name, el_type; 
        dims, chunk_length, portable,
    )
end

"""
TODO
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
TODO
"""
function load_hdf5_vector(group_or_dataset, el_type; kwargs...)
    return load_hdf5_vector(storage_style(el_type; kwargs...), group_or_dataset, el_type; kwargs...)
end

function copy_to_hdf5_vector(group, name, collection; kwargs...)
    v = create_hdf5_vector(group, name, eltype(collection); kwargs...)
    for el in collection
        push!(v, el)
    end
    return v
end

"""
    create_hdf5_vector(style, group, name, el_type; kwargs...)

Creates the appropriate HDF5 vector type for the given storage style and element type,
storing the vector in the given HDF5 `group`` in a new group/dataset, `name`.

Optional keyword arguments:

* `dims`: Tuple of the dimensions to use for a Vector, Matrix, or Array
* `chunk_length`: Length of chunk to use (default 1000)
* `portable`: True to maximize how "portable" the storage is (default true)
"""
function create_hdf5_vector(style::AbstractHDF5VectorStorageStyle, group, name, el_type; kwargs...)
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

function load_hdf5_vector(style::ElementalStorageStyle, group, el_type; kwargs...)
    dataset = group["data"]
    datatype = style.datatype # eltype(dataset)
    count = size(dataset)[end]
    return HDF5VectorOfElementalTypes{el_type, datatype}(dataset, datatype, count)
end

Base.length(arr::HDF5VectorOfElementalTypes) = arr.count # Common with HDF5VectorOfArrayishTypes
function Base.setindex!(arr::HDF5VectorOfElementalTypes{T, DT}, el, k) where {T, DT}
    arr.dataset[k] = deconstruct(arr, el)
end
function Base.getindex(arr::HDF5VectorOfElementalTypes{T, DT}, k) where {T, DT}
    # construct(T, read(arr.dataset, DT, k))
    construct(arr, read(arr.dataset, DT, k))
end
function Base.collect(arr::HDF5VectorOfElementalTypes{T, DT}) where {T, DT}
    data = read(arr.dataset, DT, 1:arr.count)
    # return [construct(T, el) for el in data]
    return [construct(arr, el) for el in data]
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
    arr.dataset[colons(D)..., k] = deconstruct(arr, el)
end
function Base.getindex(arr::HDF5VectorOfArrayishTypes{T, D, DT}, k) where {T, D, DT}
    # construct(T, read(arr.dataset, DT, colons(D)..., k))
    construct(arr, read(arr.dataset, DT, colons(D)..., k))
end
function copy_each_frame_and_construct!(arr, collected::Vector{T}, data::Array{ET, N}, n) where {T, ET, N}
    for k in 1:n
        v = view(data, (Colon() for _ in 1:N-1)..., k) # view seems to allocate for matrices and above.
        # collected[k] = construct(T, v)
        collected[k] = construct(arr, v)
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
function Base.getindex(arr::HDF5VectorWithByteArrayStorage{T}, k) where {T}
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
                ft;
                chunk_length,
                portable,
            ) for (fn, ft) in zip(fieldnames(T), fieldtypes(T))
        ],
        0,
    )
end

function load_hdf5_vector(style::CompositeStorageStyle, group_or_dataset, el_type; kwargs...)
    this_group = group_or_dataset
    arrays = [
        load_hdf5_vector(this_group["data"][string(fn)], ft; kwargs...)
        for (fn, ft) in zip(fieldnames(el_type), fieldtypes(el_type))
    ]
    return HDF5VectorOfCompositeTypes{el_type}(arrays, length(first(arrays)))
end

Base.length(arr::HDF5VectorOfCompositeTypes) = arr.count

function Base.setindex!(arr::HDF5VectorOfCompositeTypes{T}, el, k) where {T}
    for (sub_array, fn) in zip(arr.arrays, fieldnames(T))
        setindex!(sub_array, getfield(el, fn), k)
    end
    return el
end

function Base.push!(arr::HDF5VectorOfCompositeTypes{T}, el) where {T}
    for (sub_array, value) in zip(arr.arrays, deconstruct(arr, el))
        push!(sub_array, value)
    end
    arr.count += 1
    return arr
end

# This assumes the struct can be created with its individual fields, which isn't perfectly
# general, but what else can we do? Something with StructTypes?
function Base.getindex(arr::HDF5VectorOfCompositeTypes{T}, k) where {T}
    return construct(arr, ((getindex(sub_array, k) for sub_array in arr.arrays)...,))
end

function Base.collect(arr::HDF5VectorOfCompositeTypes{T}) where {T}
    collected_arrays = map(collect, arr.arrays)
    return [construct(arr, els) for els in zip(collected_arrays...)]
end

# The `el` here will always be a tuple of the values for the fields.
construct(::HDF5VectorOfCompositeTypes{T}, el) where {T} = T(el...,)
construct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: Tuple} = el
construct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: NamedTuple} = T(el)

# Similarly, to deconstruct, get a tuple of the values.
function deconstruct(::HDF5VectorOfCompositeTypes{T}, el) where {T}
    return Tuple(getfield(el, fn) for fn in fieldnames(T))
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
construct(::HDF5VectorOfElementalTypes{T, DT}, el::T) where {T, DT} = el
deconstruct(::HDF5VectorOfElementalTypes{T, DT}, el::T) where {T, DT} = el

#########################
# Other Elemental Types #
#########################


##########
# String #
##########

storage_style(el_type::Type{String}; kwargs...) = ElementalStorageStyle(el_type)
construct(::HDF5VectorOfElementalTypes{String, DT}, el::String) where {DT} = el
deconstruct(::HDF5VectorOfElementalTypes{String, DT}, el::String) where {DT} = el

########
# Char #
########

storage_style(el_type::Type{<:Char}; kwargs...) = ElementalStorageStyle(Int32) # I don't know why these are Int32 instead of Int.
construct(::HDF5VectorOfElementalTypes{Char, DT}, el::Int32) where {DT} = Char(el)
deconstruct(::HDF5VectorOfElementalTypes{Char, DT}, el::Char) where {DT} = Int32(el)

########
# Enum #
########

storage_style(el_type::Type{<:Enum}; kwargs...) = ElementalStorageStyle(Int32) # I don't know why these are Int32 instead of Int.
construct(::HDF5VectorOfElementalTypes{T, DT}, el::Int32) where {T <: Enum, DT} = T(el)
deconstruct(::HDF5VectorOfElementalTypes{T, DT}, el::Enum) where {T <: Enum, DT} = Int32(el)

##########
# NTuple #
##########

# When using the array storage style, construct will be given a view into the data.
# When using the composite storage style, construct will be given a tuple of elements.
# These are fundamentally different things, so we need the construction to know what style
# of construction to use. Some options:
#
# 1. Make different construct methods, like construct_from_array, construct_from_element.
# 2. Make construct(style, el), deconstruct(style, el) or construct(arr, el) since the
#    vector type has all of the necessary information. The former might be most consistent
#    though. The vector type could store the style.
function storage_style(::Type{<:NTuple{N, T}}; dims = nothing, kwargs...) where {N, T}
    if is_elemental(T; kwargs...)
        @assert isnothing(dims) || dims == (N,) "The dimensions of the NTuple ($N) don't match the provided `dims` keyword argument, $dims."
        return ArrayStorageStyle(T, (N,))
    else
        return CompositeStorageStyle()
    end
end

# By constructing explicitly from 1:N, this is all known at compile time, so this doesn't
# allocate.
function construct(::HDF5VectorOfArrayishTypes{T, D, DT}, elements) where {T <: NTuple, D, DT}
    N = fieldtype(D, 1)
    return ((elements[i] for i in 1:N)...,)
end
function construct(::HDF5VectorOfCompositeTypes{T}, elements) where {T <: NTuple}
    return elements # Already a tuple.
end

# We use an SVector here because the HDF5 library doesn't know how to take an NTuple as if
# it were a vector.
function deconstruct(::HDF5VectorOfArrayishTypes, el::NTuple{N, ET}) where {N, ET}
    return SVector{N, ET}(el...,)
end
function deconstruct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: NTuple}
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

construct(::HDF5VectorOfArrayishTypes{T, D, DT}, el) where {T <: Array, D, DT} = collect(el)
deconstruct(::HDF5VectorOfArrayishTypes{T, D, DT}, el) where {T <: Array, D, DT} = el

###########
# SVector #
###########

function storage_style(::Type{<:SVector{N, T}}; dims = nothing, kwargs...) where {N, T}
    if is_elemental(T; kwargs...)
        return ArrayStorageStyle(T, (N,))
    else
        return CompositeStorageStyle()
    end
end

construct(::HDF5VectorOfArrayishTypes{T, D, DT}, el) where {T <: SVector, D, DT} = T(el)
deconstruct(::HDF5VectorOfArrayishTypes{T, D, DT}, el::SVector) where {T <: SVector, D, DT} = el

# When these are composite, we treat them like normal composite types. They have a `data`
# field, and we log that one field, letting the type of the `data` field break down like
# any other composite type.
construct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: SVector} = T(el...)
deconstruct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: SVector} = (el.data,)

###########
# SMatrix #
###########

using StaticArrays: SMatrix

function storage_style(::Type{SMatrix{M, N, T, L}}; dims = nothing, kwargs...) where {M, N, T, L}
    if is_elemental(T; kwargs...)
        @assert isnothing(dims) || dims == (M, N) "The dimensions of the SMatrix ($M, $N) don't match the provided `dims` keyword argument, $dims."
        return ArrayStorageStyle(T, (M, N,))
    else
        return CompositeStorageStyle()
    end
end

construct(::HDF5VectorOfArrayishTypes{T, D, DT}, el) where {T <: SMatrix, D, DT} = T(el)
deconstruct(::HDF5VectorOfArrayishTypes, el::SMatrix) = el

# When these are composite, we treat them like normal composite types. They have a `data`
# field, and we log that one field, letting the type of the `data` field break down like
# any other composite type.
# TODO: Test these.
construct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: SMatrix} = T(el...)
deconstruct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: SMatrix} = (el.data,)

##########
# SArray #
##########

using StaticArrays: SArray

function storage_style(::Type{SArray{S, T, D, L}}; dims = nothing, kwargs...) where {S, T, D, L}
    if is_elemental(T; kwargs...)
        @assert isnothing(dims) || dims == fieldtypes(S) "The dimensions of the SArray $(fieldtypes(S))  don't match the provided `dims` keyword argument, $dims."
        dims = fieldtypes(S) # This returns a tuple of numbers because S uses "value types".
        return ArrayStorageStyle(T, dims)
    else
        return CompositeStorageStyle()
    end
end

construct(::HDF5VectorOfArrayishTypes{T, D, DT}, el) where {T <: SArray, D, DT} = T(el)
deconstruct(::HDF5VectorOfArrayishTypes, el::SArray) = el

# When these are composite, we treat them like normal composite types. They have a `data`
# field, and we log that one field, letting the type of the `data` field break down like
# any other composite type.
# TODO: Test these.
construct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: SArray} = T(el...)
deconstruct(::HDF5VectorOfCompositeTypes{T}, el) where {T <: SArray} = (el.data,)

end # module HDF5Vectors
