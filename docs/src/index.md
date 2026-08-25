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

Byte-array serialization (see below) is append-only and does not support replacement. A composite vector supports replacement only when the storage representation of every field supports it.

## Supported Types

This works for storing many common types, including:

Elemental types:

* Int8, Int16, Int32, and Int64 (and unsigned forms)
* Float32 and Float64
* Enum
* Char
* Bits-type structs when created with `portable = false`

String-like types:

* String
* Symbol

Singleton types:

* `Nothing`
* Immutable zero-field marker types with zero-argument constructors
* Empty tuples, named tuples, and static arrays

Array-like types:

* SVector, SMatrix, and SArray of elemental type
* Vector, Matrix, and Array of elemental values when their fixed dimensions are supplied with `dims`
* NTuple of elemental type

Composite types:

* General tuple of types on this list
* General named tuple of types on this list
* General structs whose fields have supported types and which can be reconstructed from their field values

Serialized types:

* Vector, Matrix, and Array of non-elemental type or whose dimensions are not known in advance
* Custom types explicitly assigned `ByteArrayStorageStyle` or `JSONStorageStyle`

Serialization provides a fallback approach for logging to HDF5 when other storage types do not make sense. For instance, a nonconcrete type, whose structure may change from element to element, defaults to serialization. This is slow and is intended only for types selected by the rules above. See "Specifying a Storage Type" below for more.

Primitive types that HDF5.jl does not natively support, including `Float16`, `Int128`, and `UInt128`, are rejected unless the user defines another storage style for them.

## How Data Is Stored

### Elemental Types

The HDF5 vector is a group, and its values are stored in a one-dimensional dataset named `data` inside that group. For example, the values for a vector at `/my_group/my_vector` are stored at `/my_group/my_vector/data`.

### Array-Like Types

When the elements to be stored each have dimensions like (M, N, ...), the HDF5 file will have an array of the appropriate type whose dimensions are (M, N, ..., Z), where Z is the number of elements being stored. This is easy to interpret outside of Julia while also allowing fast access and efficient storage.

When the elements to store are Vector, Matrix, or Array (or any AbstractArray whose dimensions are not known from the type), the `dims = (M, N, ...)` argument must be provided to [`create_hdf5_vector`](@ref) to use array-like storage. Otherwise, arrays cannot generally be stored this way and will instead be serialized to byte arrays, which is far slower and uninterpretable outside Julia.

### Composite Types

There are two ways to store a composite type: "portable" and "non-portable".

"Portable" storage means that an HDF5 group is created for each field of the struct, and inside of each is a group/dataset.

To see how structs can be stored portably, imagine we want a vector of 100 `MyType` elements, where:

```
struct MySubType
    c::Int64                # Elemental
    d::NTuple{2, Float64}   # Array-like
end
struct MyType
    a::Float64              # Elemental
    b::MySubType            # Composite
end
```

If the HDF5 vector were created in the `/my_group` group with the name `my_type`, those would be stored portably like so:

```
/my_group/my_type/               # Group
/my_group/my_type/data/a         # Array of 100 Float64
/my_group/my_type/data/b         # Group
/my_group/my_type/data/b/data/c  # Array of 100 Int64
/my_group/my_type/data/b/data/d  # Array of 2-by-100 Float64
```

For bits-type structs, a user can specify that they want "non-portable" storage. This means that the HDF5.jl package can define a custom HDF5 type to store the struct, and the resulting HDF5 file will look like this:

```
/my_group/my_type/data # An array of the HDF5 custom type
```

This is much faster and more efficient, but accessing it outside of Julia will require substantially more code to work with the HDF5 type definition system.

### Serialized Types

When a type is selected for serialization by the rules above, it can be stored as serialized data. This is far slower than the other storage styles and is intended for supported types that do not have a useful native or structured representation. There are two types of serialization currently provided.

The `ByteArrayStorageStyle` uses Julia's Serialization package to serialize a given type to a byte array. The resulting HDF5 dataset will be uninterpretable outside of Julia.

The `JSONStorageStyle` uses the JSON3 package to serialize a given type to a JSON string. The resulting HDF5 dataset will be an array of JSON strings. See the example below. In order to use the `JSONStorageStyle`, your project will have to import JSON3.

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
