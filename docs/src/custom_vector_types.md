# Custom HDF5 Vector Types

```@meta
CurrentModule = HDF5Vectors
```

To create a new type of HDF5 vector, define a new storage style type (`<:AbstractHDF5VectorStorageStyle`), create the corresponding vector type (`<:AbstractHDF5Vector`), and define [`storage_style`](@ref) for the element types that use it. The style selection must be reproducible from the element type and stored options because it is performed again when loading a vector.

Then implement the following HDF5Vectors storage hooks:

* `create_hdf5_vector(style::MyNewHDF5VectorStorageStyle, group, name, el_type; kwargs...)`
* `load_hdf5_vector(style::MyNewHDF5VectorStorageStyle, group_or_dataset, el_type; kwargs...)`

The generic `copy_to_hdf5_vector` implementation creates the selected vector and calls `push!` for each element. A storage backend may also implement a specialized bulk-copy method when it can do so more efficiently:

* `copy_to_hdf5_vector(style::MyNewHDF5VectorStorageStyle, group, name, collection; kwargs...)`

These style-taking methods are implementation hooks. Application code should call `create_hdf5_vector` or `copy_to_hdf5_vector` without passing a style directly.

The vector type must also implement the following parts of the AbstractArray interface:

* `Base.length(v)`
* `Base.setindex!(v, el, k)`
* `Base.push!(v, el)`
* `Base.getindex(v, k)`
* `Base.collect(v)`

These have definitions for `AbstractHDF5Vector` and likely don't need custom implementations:

* [`iterable`](@ref)
* `Base.eltype(v)`
* `Base.size(v)`
* `Base.similar(v, ...)`
* `Base.broadcastable(v)`
* `Base.map(f, v)`
* `Base.mapreduce(f, op, v; kwargs...)`
* `Base.iterate(v)` and `Base.iterate(v, state)`

If your type uses `construct` and `deconstruct`, you'll need methods for those:

* `construct(::Type{MyHDF5Vector}, el)`
* `deconstruct(::Type{MyHDF5Vector}, el)`

For an example of implementing a new type of storage, see the source for `HDF5VectorOfCompositeTypes`. It is fairly short, despite that it is used for the storage of all composite types in this package.
