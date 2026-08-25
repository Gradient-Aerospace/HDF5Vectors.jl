# Supported Element Types and Creation Options

```@meta
CurrentModule = HDF5Vectors
```

HDF5Vectors chooses a storage representation from the vector's declared element type and the options passed to [`create_hdf5_vector`](@ref) or [`copy_to_hdf5_vector`](@ref). The common types below work without defining a custom storage style.

The examples on this page assume the following imports:

```julia
import HDF5
using HDF5Vectors
```

## Scalars and String-Like Values

The simplest element types are stored in one HDF5 dataset.

| Julia element type | HDF5 representation |
|:--|:--|
| `Bool` | 8-bit HDF5 bitfield |
| `Int8`, `Int16`, `Int32`, `Int64` and their unsigned forms | Corresponding HDF5 integer |
| `Float32`, `Float64` | Corresponding HDF5 float |
| `String` | HDF5 string |
| `Symbol` | HDF5 string |
| `Char` | `Int32` Unicode code point |
| `Enum` | The enum's integer base type |

Primitive types that HDF5.jl does not natively support, including `Float16`, `Int128`, and `UInt128`, are rejected unless a custom storage style is defined for them.

## Fixed-Size Arrays and Tuples

`SVector`, `SMatrix`, `SArray`, and homogeneous `NTuple` element types carry their dimensions in their Julia types, so no `dims` option is needed. When their values have an elemental representation, all vector elements are stacked in one multidimensional HDF5 dataset.

```julia
using StaticArrays

HDF5.h5open("static_vectors.h5", "w") do file
    positions = create_hdf5_vector(file["/"], "positions", SVector{3, Float64})
    push!(positions, SVector(1.0, 2.0, 3.0))
end
```

Heterogeneous tuples and named tuples use the same field-oriented storage as other composite types.

## Vectors, Matrices, and Arrays

The dimensions of a `Vector`, `Matrix`, or `Array` are not part of its Julia type. Supply `dims` when every element will have the same dimensions and its values can use an elemental HDF5 representation:

```julia
HDF5.h5open("dynamic_vectors.h5", "w") do file
    positions = create_hdf5_vector(file["/"], "positions", Vector{Float64}; dims = (3,))
    push!(positions, [1.0, 2.0, 3.0])
    push!(positions, [4.0, 5.0, 6.0])
end
```

Every added element is checked against the declared dimensions. The dimensions must be a tuple of positive integers whose length matches the array rank.

The same option applies when copying an existing collection:

```julia
source = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

HDF5.h5open("copied_positions.h5", "w") do file
    positions = copy_to_hdf5_vector(file["/"], "positions", source; dims = (3,))
end
```

Without `dims`, these array values use Julia byte serialization. That permits dimensions to vary from element to element, but is slower and cannot be interpreted outside Julia. Supplying `dims` does not force array-like storage when the array's values lack a supported elemental representation.

## Composite Types

Concrete structs, heterogeneous tuples, and named tuples are stored field-by-field by default. Each field must itself have a supported representation, and composite types can be nested.

```julia
using StaticArrays

struct Sample
    time::Float64
    position::SVector{3, Float64}
    label::String
end

HDF5.h5open("samples.h5", "w") do file
    samples = create_hdf5_vector(file["/"], "samples", Sample)
    push!(samples, Sample(0.0, SVector(1.0, 2.0, 3.0), "start"))
end
```

Default reconstruction calls the declared element type with the stored field values in field order. A type whose constructors do not accept those values requires a custom [`construct`](@ref) method.

Bits-type structs can instead use one native HDF5 datatype when created with `portable = false`. See [the `portable` option](#The-portable-Option) and [HDF5 Storage Layout](storage_layout.md) for the tradeoff.

## Singleton Types

Singleton types have only one possible value, so HDF5Vectors stores the vector length rather than repeating that value. Supported examples include `Nothing`, empty tuples and named tuples, empty static arrays, and immutable zero-field marker types with zero-argument constructors.

Mutable zero-field types are not supported because separate instances have distinct identities that cannot be represented by storing only a count.

## Serialized Values

HDF5Vectors uses Julia's `Serialization` format as a fallback for supported nonconcrete element types and for `Vector`, `Matrix`, or `Array` values whose dimensions were not declared. A custom type can also explicitly select `ByteArrayStorageStyle`.

Serialization is not a promise that every Julia value can be stored. It is intended for types that Julia's `Serialization` library can reliably round-trip in the environments where the HDF5 file will be used. The stored bytes are Julia-specific and cannot be interpreted by ordinary HDF5 readers in other languages.

`JSONStorageStyle` is an explicit alternative for values supported by JSON3. Its JSON strings can be read outside Julia. See [Custom Element Types](custom_element_types.md) for selecting either serialization style.

## Creation Options

The following options are accepted by both [`create_hdf5_vector`](@ref) and [`copy_to_hdf5_vector`](@ref). `dims` and `portable` are stored in the vector metadata because they can affect how the vector must later be loaded; `chunk_length` affects only dataset creation.

### The `dims` Option

`dims` declares the fixed dimensions of each dynamically sized array element and enables efficient array-like storage when the element values have a supported elemental representation. Dimensions of tuples and static arrays are inferred from their types.

### The `chunk_length` Option

`chunk_length` is the number of vector elements in each chunk of the underlying extensible HDF5 datasets. It defaults to 1000, affects storage layout and I/O performance, and does not limit the total vector length.

```julia
HDF5.h5open("large_log.h5", "w") do file
    values = create_hdf5_vector(file["/"], "values", Float64; chunk_length = 10_000)
end
```

The default is a reasonable starting point. The best value depends on element size and the application's read and write patterns.

### The `portable` Option

`portable` controls the representation of bits-type composite elements. It defaults to `true`, which stores each field separately in datasets that are straightforward to inspect from other languages. Setting it to `false` permits HDF5.jl to store the entire bits type as one native HDF5 datatype, which is generally faster but requires the external reader to interpret that datatype.

```julia
struct Point
    x::Float64
    y::Float64
end

HDF5.h5open("points.h5", "w") do file
    portable_points = create_hdf5_vector(file["/"], "portable_points", Point)
    native_points = create_hdf5_vector(file["/"], "native_points", Point; portable = false)
end
```

This option is ignored for element types that have only one supported representation. It does not make Julia-serialized values readable outside Julia.
