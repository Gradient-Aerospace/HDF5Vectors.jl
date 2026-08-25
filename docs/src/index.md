# HDF5Vectors

```@meta
CurrentModule = HDF5Vectors
```

HDF5Vectors provides vectors whose underlying values live in an HDF5 file rather than in RAM. They support familiar `AbstractVector` operations and can grow over time with `push!`, making them useful for incrementally logging more data than will fit in memory. Existing Julia collections can also be copied into HDF5 layouts that are straightforward to read from other languages.

## Installation

Install HDF5Vectors from the Julia package prompt:

```
pkg> add https://github.com/Gradient-Aerospace/HDF5Vectors.jl
```

## Getting Started

Create an HDF5 vector by opening an HDF5 file, selecting the group that will contain it, and specifying its name and element type. This example creates `/x` and appends 100 `Float64` values:

```julia
import HDF5
using HDF5Vectors

HDF5.h5open("storage.h5", "w") do file
    x = create_hdf5_vector(file["/"], "x", Float64)
    for value in 1.0 : 100.0
        push!(x, value)
    end
end
```

The `do` block closes the HDF5 file when the block finishes. An HDF5 vector uses objects owned by its open file, so use the vector only while that file remains open.

Open the file again and load the vector from its HDF5 group:

```julia
HDF5.h5open("storage.h5", "r") do file
    x = load_hdf5_vector(file["/x"])
    @show length(x)
    @show x[1]
    @show x[end]
    values = collect(x)
end
```

The element type and creation options are stored in the HDF5 vector's metadata, so callers normally need to provide only the group. If the element type is already known, `load_hdf5_vector(file["/x"], Float64)` can be used instead.

## Copying an Existing Collection

When all the values already exist in Julia, use [`copy_to_hdf5_vector`](@ref). Supported storage styles use bulk writes where possible, making this more efficient than calling `push!` for every value.

```julia
source = Float64[1, 2, 3, 4]

HDF5.h5open("copied_values.h5", "w") do file
    x = copy_to_hdf5_vector(file["/"], "x", source)
    @show collect(x)
end
```

The copied vector uses `eltype(source)` as its declared element type.

## Continuing to Add Values

Open an existing file with write access to continue appending to a stored vector. HDF5 uses the mode `"r+"` for opening an existing file for both reading and writing.

```julia
HDF5.h5open("storage.h5", "r+") do file
    x = load_hdf5_vector(file["/x"])
    push!(x, 101.0)
end
```

## Common Vector Operations

### Adding Elements

Values passed to `push!` must already be instances of the HDF5 vector's declared element type; HDF5Vectors does not convert them to that type. For example, a vector declared with element type `Float64` accepts `1.0`, but not the integer `1`.

### Reading and Iterating

Scalar, range, integer-vector, logical, and colon indexing follow normal Julia vector behavior. Non-scalar indexing returns an ordinary Julia `Vector`, and `collect` reads all values into a Julia `Vector`.

```julia
HDF5.h5open("storage.h5", "r") do file
    x = load_hdf5_vector(file["/x"])
    first_value = x[1]
    first_ten = x[1:10]
    selected = x[[1, 10, 20]]
    all_values = collect(x)
end
```

Direct iteration reads each element individually from HDF5. When the entire vector fits in memory, it is generally much faster to call [`iterable`](@ref) and iterate over its result:

```julia
HDF5.h5open("storage.h5", "r") do file
    x = load_hdf5_vector(file["/x"])
    result = [value^2 for value in iterable(x)]
end
```

Currently, [`iterable`](@ref) loads the entire HDF5 vector into a Julia `Vector` before iteration. This avoids a separate HDF5 read for every element, but it requires enough memory to hold the full vector.

### Replacing Elements

Some storage representations support replacing an existing value with `setindex!`:

```julia
HDF5.h5open("storage.h5", "r+") do file
    x = load_hdf5_vector(file["/x"])
    x[10] = 42.0
end
```

Byte-array serialization is append-only and does not support replacement. A composite vector supports replacement only when the storage representation of every field supports it. See [Supported Element Types and Creation Options](supported_types.md) for the available representations.

## Choosing Types and Understanding the HDF5 File

The next guides describe the available element representations and the resulting on-disk format:

* [Supported Element Types and Creation Options](supported_types.md) explains which Julia types can be stored and how `dims`, `chunk_length`, and `portable` affect them.
* [HDF5 Storage Layout](storage_layout.md) documents the datasets and groups that readers in Julia, Python, MATLAB, C++, and other environments will encounter.

## Specifying a Storage Type

Users can specify what "style" of storage should be used for a given type. Storage style is a property of the element type and relevant creation options, not an override for one particular vector. The same [`storage_style`](@ref) method is called again when loading the vector, so it must make a consistent selection from those inputs.

For instance, suppose we had the following type:

```
@enum ServerStatus unknown up down
struct SomeServerDetails
    hostname::String
    status::ServerStatus
end
```

When that's stored in the HDF5 file, let's make it serialize to JSON. To do this, we can apply the following trait:

```
import HDF5Vectors: storage_style, JSONStorageStyle
storage_style(::Type{SomeServerDetails}; kwargs...) = JSONStorageStyle()
```

That's it. Now, when we create an array for this type, each element will be serialized to JSON. The resulting HDF5 file will have an array of strings that can be loaded in any other environment. To complete the example:

```
import HDF5
import JSON3
using HDF5Vectors
HDF5.h5open("server_details.h5", "w") do fid
    details = create_hdf5_vector(fid["/"], "details", SomeServerDetails)
    push!(details, SomeServerDetails("localhost", up))
    push!(details, SomeServerDetails("old_pc", down))
    push!(details, SomeServerDetails("phone", unknown))
    @show collect(details)
end
```

The HDF5 file will have the following:

```
/details/data/json/data # An array of JSON strings, one for each of the pushed elements
```

(By default, the JSON is compact in style, not "pretty", to reduce the extra storage burden of all of those spaces. You can "pretty" on extraction from the HDF5 file if a human is supposed to look at it.)
