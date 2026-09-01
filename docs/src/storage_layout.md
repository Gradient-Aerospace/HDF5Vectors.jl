# HDF5 Storage Layout

```@meta
CurrentModule = HDF5Vectors
```

Many HDF5Vectors files are written in Julia and read in Python, MATLAB, C++, or other environments. The paths under `data` are therefore a documented part of the format rather than an incidental implementation detail.

## The Vector Group

Every HDF5 vector occupies one group with two children:

```
/values/
/values/data/                   # Physical value storage
/values/metadata/               # HDF5Vectors schema and bookkeeping
```

The logical length is stored at `/values/metadata/count` as a scalar `Int64`. The most useful additional metadata paths are:

```
/values/metadata/format_name
/values/metadata/format_version
/values/metadata/logical_type
/values/metadata/schema/
/values/metadata/serialized_schema
```

The tree under `metadata/schema` describes the selected representation with ordinary HDF5 strings, integers, and groups. It records schema and codec identifiers, logical and encoded type names, dimensions, record field names, and child schemas as applicable.

`serialized_schema` contains the exact Julia schema object. Ordinary untyped loading deserializes it so application-defined codecs can be recovered without a registry inside HDF5Vectors. Typed loading, such as `load_hdf5_vector(group, MyType)`, can repeat inference from the stored options and validate the result against the ordinary schema tree. An explicitly supplied schema is authoritative and is checked for compatibility with the physical layout rather than exact equality with the stored Julia schema. This permits deliberate migrations after a Julia type or schema implementation changes. Files passed to a loading form that deserializes either values or schemas should be trusted.

External readers normally need only `metadata/count`, the documented schema information they care about, and the paths under `data`.

## Scalar Values

Numbers, booleans, strings, symbols, characters, enums, JSON strings, and application values with scalar codecs use:

```
/values/data/values             # One-dimensional dataset with N encoded values
```

Native integers and floats retain their corresponding HDF5 datatypes. `Symbol` is encoded as a string, `Char` as an `Int32` Unicode code point, and an enum as its integer base type. A custom scalar codec records its encoded type in `/values/metadata/schema/encoded_type`.

A JSON schema also uses `/values/data/values`; each entry is one compact JSON string. No JSON-specific group is added.

## Dense Array-Like Values

A fixed-size array-like element is stacked along one additional Julia dimension. If each Julia element has dimensions `(M, N, ...)` and the vector has length `Z`, HDF5.jl presents the value dataset as:

```
/values/data/values             # Julia shape (M, N, ..., Z)
```

The final Julia dimension selects the HDF5Vector element. This layout is used for supported homogeneous tuples and static arrays, and for dynamic `Array` types created with `dims`.

HDF5.jl reverses multidimensional extents at the HDF5 C boundary to account for Julia's column-major order. A row-major reader such as h5py therefore sees the raw shape `(Z, ..., N, M)`. For example, 100 Julia matrices of size `(2, 3)` appear as `(2, 3, 100)` through HDF5.jl and `(100, 3, 2)` through h5py. In h5py, `dataset[k]` selects one vector element whose axes appear in the reverse order from Julia.

## Field-Oriented Records

Portable records place one recursively encoded column under each field name. Consider:

```julia
using StaticArrays

struct GPSTimeStamp
    weeks::Int32
    microseconds::Int64
end

struct Measurement
    timestamp::GPSTimeStamp
    temperature::Float64
    position::SVector{3, Float64}
end
```

A vector stored at `/measurements` has this physical value layout:

```
/measurements/data/timestamp/weeks/values
/measurements/data/timestamp/microseconds/values
/measurements/data/temperature/values
/measurements/data/position/values
```

The first three datasets have shape `(Z,)` through HDF5.jl. The position dataset has shape `(3, Z)` through HDF5.jl and `(Z, 3)` through a row-major reader such as h5py.

Every nonconstant field column contains the same number of logical values. An external reader reconstructs one `Measurement` by reading the same vector index from each field path. Nested structs continue to use their field names, so the physical paths remain meaningful without Julia.

Tuple fields use the names `1`, `2`, and so on because tuples have positions rather than Julia field names. Named tuples use their names. The ordered field-name vector and recursive child schemas are also recorded under `metadata/schema`; numeric child names there describe schema order, not physical value paths.

## Native Bits-Type Records

When a nonzero-size bits type is created with `portable = false`, it may use one HDF5 datatype instead of field-oriented storage:

```
/measurements/data/values       # Dataset of the HDF5 datatype derived from the Julia type
```

This representation is often faster. External readers must inspect and interpret the resulting HDF5 datatype, so the default field-oriented layout is generally preferable when portability matters.

## Constant Values

A constant schema stores no per-element values. Its data group is empty:

```
/markers/data/                  # Empty group
/markers/metadata/count        # Number of logical marker values
```

The constant itself is contained in the serialized Julia schema. An external reader usually needs application knowledge to assign meaning to the count. When a schema is supplied explicitly during loading, its constant is used without reading or comparing the constant in the stored Julia schema because there are no physical values whose interpretation needs to be validated.

## Julia-Serialized Values

Blob storage concatenates the independently serialized values and records the cumulative ending byte position of each one:

```
/values/data/bytes              # One-dimensional UInt8 dataset
/values/data/stops              # One-dimensional Int64 dataset
```

For a zero-based reader, element `k` begins at byte zero when `k == 0` and otherwise at `stops[k - 1]`; it ends immediately before `stops[k]`. Repeated stops represent empty encoded values.

The default blob codec uses Julia's `Serialization` format. These bytes are not suitable for reconstruction outside Julia, and loading them requires the relevant Julia types and modules to be available.

## Finding the Selected Representation

An external tool that does not already know the application schema can inspect `/values/metadata/schema/kind`. The built-in values are `scalar`, `dense`, `record`, `blob`, and `constant`. Record schema nodes contain `field_names` and a `children` group; dense nodes contain `dimensions`; scalar and dense nodes contain `encoded_type`.

Schema and codec identifiers are descriptive strings. Applications can give their custom implementations stable identifiers with [`schema_identifier`](@ref) and [`codec_identifier`](@ref).
