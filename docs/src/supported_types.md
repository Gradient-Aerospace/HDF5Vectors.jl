# Supported Element Types and Creation Options

```@meta
CurrentModule = HDF5Vectors
```

HDF5Vectors infers a storage schema from the declared element type. A schema separates the logical Julia value from its encoded value and physical HDF5 layout. This lets `Char`, for example, behave as `Char` in Julia while being stored as an `Int32`, and it lets a struct be represented by one recursively stored column per field.

The examples on this page assume these imports:

```julia
import HDF5
using HDF5Vectors
```

## Scalars and String-Like Values

The simplest element types use one one-dimensional HDF5 dataset.

| Julia element type | Encoded HDF5 value |
|:--|:--|
| `Bool` | HDF5 boolean representation provided by HDF5.jl |
| `Int8`, `Int16`, `Int32`, `Int64` and their unsigned forms | Corresponding HDF5 integer |
| `Float32`, `Float64` | Corresponding HDF5 float |
| `String` | HDF5 variable-length string |
| `Symbol` | HDF5 string |
| `Char` | `Int32` Unicode code point |
| `Enum` | The enum's integer base type |

Primitive types without a native HDF5 representation, including `Float16`, `Int128`, and `UInt128`, are rejected by default. A [custom codec](custom_element_types.md#Defining-a-Scalar-Codec) can map such a logical type to a supported encoded type when the application has an appropriate conversion.

## Fixed-Size Arrays and Tuples

Homogeneous `NTuple`, `SVector`, `SMatrix`, and `SArray` types carry their dimensions in their types. When their element type has a scalar encoding, HDF5Vectors stacks their values in one multidimensional dataset.

```julia
using StaticArrays

HDF5.h5open("positions.h5", "w") do file
    positions = create_hdf5_vector(file["/"], "positions", SVector{3, Float64})
    push!(positions, SVector(1.0, 2.0, 3.0))
    push!(positions, SVector(4.0, 5.0, 6.0))
end
```

No `dims` option is needed. Static arrays whose elements do not have scalar encodings are represented as records when possible.

Heterogeneous tuples and named tuples are records rather than dense arrays. Their entries are stored recursively under the names `1`, `2`, and so on for tuples, or their field names for named tuples.

## Dynamically Sized Arrays

The dimensions of a `Vector`, `Matrix`, or `Array` are not part of its type. The `dims` option can declare one fixed shape shared by every element:

```julia
HDF5.h5open("dynamic_positions.h5", "w") do file
    positions = create_hdf5_vector(
        file["/"],
        "positions",
        Vector{Float64};
        dims = (3,),
    )
    push!(positions, [1.0, 2.0, 3.0])
    push!(positions, [4.0, 5.0, 6.0])
end
```

Each value must have exactly the declared dimensions. The dimensions must be a tuple of positive integers whose length matches the array rank.

The same option is available during bulk copy:

```julia
source = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

HDF5.h5open("copied_positions.h5", "w") do file
    positions = copy_to_hdf5_vector(file["/"], "positions", source; dims = (3,))
end
```

Without `dims`, dynamically sized arrays use Julia byte serialization by default. This permits the dimensions to vary between elements, but the stored values cannot be interpreted by ordinary HDF5 readers outside Julia.

## Structs and Other Records

Concrete structs are stored field-by-field by default, and their fields are inferred recursively. This is the normal representation for application data that should remain easy to inspect outside Julia.

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

The default record codec reads fields with `getfield` and reconstructs a value by calling `Sample(time, position, label)`. Nested structs, tuples, named tuples, and supported static arrays are handled recursively. A type that needs a different logical conversion can define a custom codec.

When `portable = false`, a nonzero-size bits type may instead use the single HDF5 datatype selected by HDF5.jl. This can be faster, but field-oriented storage is usually easier to read from another language.

## Constant Values

Types with one reconstructible value need no per-element dataset. HDF5Vectors stores the logical vector length and the schema's constant value. Supported examples include `Nothing`, empty tuples and named tuples, empty static arrays, and immutable zero-field marker types with zero-argument constructors.

A field-bearing singleton without a zero-argument constructor is treated as a record when its fields are supported. Mutable zero-field types are rejected because distinct instances have identities that count-only storage cannot preserve.

## Julia-Serialized Values

Julia serialization is the default fallback in two cases:

* A dynamically sized `Array` type is created without `dims`.
* The declared element type is nonconcrete, such as an abstract type.

Serialization is useful for Julia-specific data, but it is not a promise that every Julia value can be saved reliably. The relevant types and modules must be available when values are loaded, and the byte format is not intended for readers outside Julia.

An application can explicitly choose this representation with [`serialization_schema`](@ref), as shown in [Custom Element Types](custom_element_types.md#Selecting-Julia-Serialization).

## Creation Options

The type-inference forms of [`create_hdf5_vector`](@ref) and [`copy_to_hdf5_vector`](@ref) accept the same options.

### `dims`

`dims` declares the fixed shape of each dynamically sized array element. Dimensions of homogeneous tuples and static arrays are already known from their types.

### `chunk_length`

`chunk_length` controls the last dimension of each extensible HDF5 dataset chunk. It defaults to 1000 and does not limit the total vector length.

```julia
HDF5.h5open("large_log.h5", "w") do file
    values = create_hdf5_vector(file["/"], "values", Float64; chunk_length = 10_000)
end
```

The default is a reasonable starting point. The best choice depends on the encoded value size and the application's read and write patterns.

### `portable`

`portable` defaults to `true`. It gives bits-type records field-oriented storage rather than one native HDF5 datatype. It does not change scalar storage, and it cannot make Julia-serialized values readable outside Julia.

### `serialize_arrays`

`serialize_arrays` defaults to `true`. Setting it to `false` rejects dynamically sized array types when no usable dense schema can be inferred. This can help an application ensure that it never silently chooses Julia-specific storage for arrays.

### `serialize_nonconcrete`

`serialize_nonconcrete` defaults to `true`. Setting it to `false` rejects nonconcrete declared element types instead of selecting Julia serialization.

## Supplying an Explicit Schema

Applications that define codecs can construct a schema directly and pass it to `create_hdf5_vector(group, name, schema; chunk_length)`. In that form, schema-inference options do not apply because the representation has already been selected. [Custom Element Types](custom_element_types.md) begins with the smaller and usually preferable approach of extending [`infer_schema`](@ref) for the application type.
