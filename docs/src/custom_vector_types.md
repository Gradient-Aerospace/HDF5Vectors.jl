# Custom HDF5 Vector Types

```@meta
CurrentModule = HDF5Vectors
```

To create a new type of HDF5 vector, you will need to define a new storage style type (`<:AbstractHDF5VectorStorageStyle`), create your type (`<:AbstractHDF5Vector`), and then implement following HDF5Vectors functions:

* [`create_hdf5_vector(style::AbstractHDF5VectorStorageStyle, group, name, el_type; kwargs...)`](@ref)

as well as the AbstractArray interface:

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

For an example of implementing a new type of storage, see the source for `HDF5VectorOfCompositeTypes`. It is fairly short, despite that it is used for the storage of all composite types in this package.
