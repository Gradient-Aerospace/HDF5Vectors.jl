# HDF5Vectors

```@meta
CurrentModule = HDF5Vectors
```

HDF5Vectors provides append-only vectors whose values live in an HDF5 file rather than in RAM. An `HDF5Vector` supports familiar `AbstractVector` reads and grows with `push!`, which makes it useful for logging long simulations or other streams of data. Existing Julia vectors can also be copied efficiently into HDF5 layouts designed to remain understandable outside Julia.

## Installation

HDF5Vectors can be installed from the Julia package prompt:

```
pkg> add https://github.com/Gradient-Aerospace/HDF5Vectors.jl
```

## Creating a Vector

An HDF5 vector belongs to an open HDF5 group. The following example creates `/values` and appends three `Float64` values:

```julia
import HDF5
using HDF5Vectors

HDF5.h5open("storage.h5", "w") do file
    values = create_hdf5_vector(file["/"], "values", Float64)
    push!(values, 1.0)
    push!(values, 2.0)
    push!(values, 3.0)
end
```

The `do` block closes the file when it finishes. Because an `HDF5Vector` holds open HDF5 objects, it can be used only while its file remains open.

The declared element type is a strict part of the interface. A vector created for `Float64` accepts `1.0`, but `push!(values, 1)` throws a `MethodError` rather than converting the integer.

## Loading and Reading

The vector can be loaded again by opening its group:

```julia
HDF5.h5open("storage.h5", "r") do file
    values = load_hdf5_vector(file["values"])
    @show length(values)
    @show values[1]
    @show values[2:3]
    ordinary_vector = collect(values)
end
```

The element type and complete storage schema are recorded when the vector is created, so the usual loading form needs only the HDF5 group. When a vector was created from a type and that type is already known, `load_hdf5_vector(file["values"], Float64)` repeats schema inference from the stored options and validates it against the file without deserializing the stored schema.

Scalar and range indexing follow normal Julia vector behavior. Integer-vector and logical indexing are also supported, and `collect` reads all values into an ordinary Julia `Vector`. Iteration works through the standard `AbstractVector` interface, although `collect` or range indexing is usually more efficient when many consecutive values are needed.

HDF5 vectors are append-only. Existing elements cannot be replaced or removed.

## Copying an Existing Vector

When the values already exist in Julia, [`copy_to_hdf5_vector`](@ref) creates and fills the HDF5 vector with one recursive bulk write:

```julia
source = Float64[1, 2, 3, 4]

HDF5.h5open("copied_values.h5", "w") do file
    values = copy_to_hdf5_vector(file["/"], "values", source)
    @show collect(values)
end
```

The declared element type is `eltype(source)`. The input must currently be an `AbstractVector`; support for general iterables may be added in a future release.

## Appending After Loading

An existing file can be opened with `"r+"` when more values need to be appended:

```julia
HDF5.h5open("storage.h5", "r+") do file
    values = load_hdf5_vector(file["values"])
    push!(values, 4.0)
end
```

## Choosing a Representation

For common Julia types, HDF5Vectors infers a complete storage schema automatically. Scalars use one HDF5 dataset, fixed-size arrays are stacked along a new dimension, and structs are usually split into named field datasets. Dynamically sized arrays and nonconcrete declared types can use Julia serialization when a directly readable layout cannot be inferred.

The remaining guides add detail:

* [Supported Element Types and Creation Options](supported_types.md) describes automatic schema inference and the available creation options.
* [HDF5 Storage Layout](storage_layout.md) shows the exact groups and datasets seen by readers in Python, MATLAB, C++, and other environments.
* [Custom Element Types](custom_element_types.md) shows how a type can select an existing physical representation with a small codec.
* [Custom Schemas](custom_schemas.md) describes the advanced interface for defining an entirely new physical representation.
* [API Reference](api.md) collects the ordinary interface and extension points.
* [When Writing to HDF5 Fails](write_failures.md) explains the limits of recovery after interrupted or unsuccessful writes.
