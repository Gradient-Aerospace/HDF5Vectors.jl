# Custom HDF5 Vector Types

```@meta
CurrentModule = HDF5Vectors
```

Most custom element types should reuse an existing representation as described in [Custom Element Types](custom_element_types.md). A new HDF5 vector backend is needed only for an entirely new on-disk representation.

## Backend Components

A backend consists of:

1. A subtype of [`AbstractHDF5VectorStorageStyle`](@ref) that selects the representation.
2. A subtype of [`AbstractHDF5Vector`](@ref) that holds the open HDF5 objects and any in-memory state.
3. A [`storage_style`](@ref) method for element types that use the backend.
4. Style-taking creation and loading hooks.
5. The required vector operations.

Style-taking methods are implementation hooks. Application code should continue to call [`create_hdf5_vector`](@ref), [`load_hdf5_vector`](@ref), and [`copy_to_hdf5_vector`](@ref) without passing a style directly.

## Creating and Loading Storage

A new style needs creation and loading methods with these signatures:

```julia
HDF5Vectors.create_hdf5_vector(
    style::MyStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type;
    dims = nothing,
    chunk_length,
    portable,
    kwargs...,
)

HDF5Vectors.load_hdf5_vector(
    style::MyStorageStyle,
    group::HDF5.Group,
    el_type;
    dims,
    portable,
    kwargs...,
)
```

The creation hook should create one child group named `name` inside the supplied parent group. It can then call [`HDF5Vectors.store_metadata`](@ref) on that new group so the ordinary loading functions can recover the element type, `dims`, and `portable` setting. Backend-specific datasets and metadata should also be stored under the same new group.

The loading hook receives the HDF5 vector group itself rather than its parent. It should open the backend's datasets, validate the stored layout sufficiently to construct a usable vector, and return the custom `AbstractHDF5Vector` subtype.

The generic [`copy_to_hdf5_vector`](@ref) implementation calls the creation hook and then `push!` for every value. A backend can add a style-taking bulk-copy method when its representation supports a more efficient write:

```julia
HDF5Vectors.copy_to_hdf5_vector(
    style::MyStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    collection;
    dims = nothing,
    chunk_length,
    portable,
    kwargs...,
)
```

Where practical, values that can fail conversion or validation should be prepared before the destination group is created. This allows ordinary input errors to be reported without leaving an incomplete destination behind.

## Required Vector Operations

The custom vector type must implement:

* `Base.length(vector)`
* `Base.push!(vector, value)`
* `Base.getindex(vector, index::Int)`
* `Base.collect(vector)` with an efficient whole-vector read

Values accepted by `push!` should already have the vector's declared element type. Any in-memory length should be updated only after the corresponding HDF5 write succeeds.

[`AbstractHDF5Vector`](@ref) supplies `eltype`, `size`, linear index style, range and logical indexing, broadcasting, `map`, reductions, iteration, and [`iterable`](@ref) in terms of the required methods.

`Base.setindex!(vector, value, index::Int)` is optional because some representations are append-only. A backend that supports replacement and may be nested inside composite storage should also define [`HDF5Vectors.supports_setindex`](@ref) to return `true`.

## A Complete Wrapping Backend

The built-in JSON extension is a useful model because it implements a new representation by wrapping an ordinary HDF5 vector of strings. The following simplified backend has the same structure:

```julia
module ExampleJSONBackend

import HDF5
import HDF5Vectors
import JSON3

struct ExampleJSONStorageStyle <: HDF5Vectors.AbstractHDF5VectorStorageStyle end

mutable struct ExampleJSONVector{T} <: HDF5Vectors.AbstractHDF5Vector{T}
    storage::HDF5Vectors.AbstractHDF5Vector{String}
end

function HDF5Vectors.create_hdf5_vector(
    style::ExampleJSONStorageStyle,
    group::HDF5.Group,
    name::AbstractString,
    el_type;
    dims = nothing,
    portable,
    kwargs...,
)
    vector_group = HDF5.create_group(group, name)
    HDF5Vectors.store_metadata(style, vector_group, el_type; dims, portable)
    data_group = HDF5.create_group(vector_group, "data")
    storage = HDF5Vectors.create_hdf5_vector(data_group, "json", String; kwargs...)
    return ExampleJSONVector{el_type}(storage)
end

function HDF5Vectors.load_hdf5_vector(
    ::ExampleJSONStorageStyle,
    group::HDF5.Group,
    el_type;
    kwargs...,
)
    storage = HDF5Vectors.load_hdf5_vector(group["data/json"], String)
    return ExampleJSONVector{el_type}(storage)
end

Base.length(vector::ExampleJSONVector) = length(vector.storage)

function Base.push!(vector::ExampleJSONVector{T}, value::T) where {T}
    push!(vector.storage, JSON3.write(value))
    return vector
end

function Base.getindex(vector::ExampleJSONVector{T}, index::Int) where {T}
    return JSON3.read(vector.storage[index], T)
end

function Base.collect(vector::ExampleJSONVector{T}) where {T}
    return T[JSON3.read(value, T) for value in collect(vector.storage)]
end

end
```

An application or package can select this backend for one of its types:

```julia
import HDF5Vectors

struct MyType
    name::String
    value::Float64
end

function HDF5Vectors.storage_style(::Type{MyType}; kwargs...)
    return ExampleJSONBackend.ExampleJSONStorageStyle()
end
```

The new representation is then selected through the ordinary public interface:

```julia
import HDF5

HDF5.h5open("custom_backend.h5", "w") do file
    values = HDF5Vectors.create_hdf5_vector(file["/"], "values", MyType)
    push!(values, MyType("example", 1.0))
    @show collect(values)
end
```

This example intentionally relies on the generic element-by-element copy and does not implement `setindex!`. A production backend can add either optimization independently.

## Testing a Backend

A useful minimum test set covers creating, pushing, scalar indexing, collecting, closing and reloading, and copying both empty and nonempty collections. Tests should exercise both loading forms—with the stored element type and with an explicit element type—and verify that the physical HDF5 datasets contain the intended representation. If replacement is supported, it is helpful to test it both directly and as a field of a composite element.
