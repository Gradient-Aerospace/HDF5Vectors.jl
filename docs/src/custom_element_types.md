# Custom Element Types

```@meta
CurrentModule = HDF5Vectors
```

Most concrete structs work without HDF5Vectors-specific methods. Customization is useful when the default field-oriented record is unsuitable or when an application type has a simpler encoded representation.

## Starting With the Default

This type needs no custom schema:

```julia
struct Measurement
    time::Float64
    name::String
    values::Vector{Float64}
end

import HDF5
using HDF5Vectors

HDF5.h5open("measurements.h5", "w") do file
    measurements = create_hdf5_vector(file["/"], "measurements", Measurement)
    push!(measurements, Measurement(0.0, "initial", [1.0, 2.0]))
end
```

The `time` and `name` fields use scalar datasets. Because the dimensions of `values` were not declared separately, that field uses Julia serialization. The complete layout is recursive, so a record can combine portable and Julia-specific fields.

## Defining a Scalar Codec

A codec is a pure conversion between a logical Julia type and an encoded type that HDF5Vectors already knows how to store. The following example stores each `Grade` as one `UInt8`:

```julia
using HDF5Vectors

struct Grade
    label::String
end

struct GradeCodec <: HDF5Vectors.AbstractCodec{Grade, UInt8} end

function HDF5Vectors.encode_value(::GradeCodec, grade::Grade)
    return UInt8(only(grade.label))
end

function HDF5Vectors.decode_value(::GradeCodec, value::UInt8)
    return Grade(string(Char(value)))
end

function HDF5Vectors.infer_schema(::Type{Grade}; kwargs...)
    return HDF5Vectors.ScalarSchema(GradeCodec())
end
```

The same methods work whether `Grade` is the vector's element type, a field of a struct, or the element type of a fixed-size array. Recursive inference always returns through the public [`infer_schema`](@ref) function.

The ordinary vector interface now needs no special handling:

```julia
import HDF5

HDF5.h5open("grades.h5", "w") do file
    grades = create_hdf5_vector(file["/"], "grades", Grade)
    push!(grades, Grade("A"))
    push!(grades, Grade("B"))
    @show read(file["grades/data/values"])
    @show collect(grades)
end
```

The schema stores the concrete `GradeCodec` object, so untyped loading does not depend on a registry of codec names inside HDF5Vectors. The module that defines `GradeCodec` must still be loaded before Julia can deserialize that schema.

An application can specialize [`codec_identifier`](@ref) if its human-readable metadata should remain stable when the Julia codec type is renamed.

## Selecting JSON

JSON storage is another scalar codec. Loading JSON3 activates the HDF5Vectors extension that supplies the JSON conversion methods. The application type must support `JSON3.write` and `JSON3.read(value, Type)`.

```julia
import JSON3
using HDF5Vectors

struct ServerDetails
    hostname::String
    active::Bool
end

JSON3.StructTypes.StructType(::Type{ServerDetails}) = JSON3.StructTypes.Struct()

function HDF5Vectors.infer_schema(::Type{ServerDetails}; kwargs...)
    return HDF5Vectors.json_schema(ServerDetails)
end
```

Each value is stored as one JSON string at `/vector_name/data/values`. This makes the encoded values directly usable by non-Julia HDF5 and JSON libraries.

JSON3 is a weak dependency of HDF5Vectors. Applications that select `json_schema` should include JSON3 in their own dependencies and load it before values are written or read.

## Selecting Julia Serialization

An application can explicitly select Julia byte serialization for a type with [`serialization_schema`](@ref):

```julia
struct Snapshot
    labels::Vector{String}
    values::Dict{String, Float64}
end

function HDF5Vectors.infer_schema(::Type{Snapshot}; kwargs...)
    return HDF5Vectors.serialization_schema(Snapshot)
end
```

No additional codec methods are needed because the built-in serialization codec converts the complete value to and from bytes. This representation is appropriate only when Julia-specific storage is acceptable.

## Defining a Record Codec

The default record codec reads a struct's fields and calls its constructor with those fields in order. A custom record codec can present different logical fields when that interface is unsuitable.

```julia
struct PointFromTuple
    x::Float64
    y::Float64
    PointFromTuple(values::Tuple{Float64, Float64}) = new(values...)
end

struct PointCodec <: HDF5Vectors.AbstractRecordCodec{PointFromTuple} end

function HDF5Vectors.decompose(::PointCodec, point::PointFromTuple)
    return (point.x, point.y)
end

function HDF5Vectors.compose(::PointCodec, fields::Tuple)
    return PointFromTuple((fields[1], fields[2]))
end

function HDF5Vectors.infer_schema(
    ::Type{PointFromTuple};
    dims = nothing,
    policy = HDF5Vectors.SchemaPolicy(),
)
    if !isnothing(dims)
        throw(ArgumentError("PointFromTuple does not accept declared dimensions."))
    end
    children = (
        HDF5Vectors.infer_schema(Float64; policy),
        HDF5Vectors.infer_schema(Float64; policy),
    )
    return HDF5Vectors.RecordSchema(
        PointFromTuple,
        ("x", "y"),
        PointCodec(),
        children,
    )
end
```

`decompose` returns one logical value for each named child schema. Each child then performs its own recursive encoding. `compose` receives the decoded fields in the same order. This separation keeps record structure independent of the physical representation selected for each field.

## Testing a Codec

A codec can first be tested without opening an HDF5 file:

```julia
schema = HDF5Vectors.infer_schema(Grade)
encoded = HDF5Vectors.encode_value(schema, Grade("A"))
@assert encoded == UInt8('A')
@assert HDF5Vectors.decode_value(schema, encoded) == Grade("A")
```

A complete integration test should also create a vector, use both `push!` and `copy_to_hdf5_vector`, inspect the encoded HDF5 dataset, close and reload the file, and verify both typed and untyped loading. If the codec is intended for record fields or fixed-size arrays, exercising that recursive use is valuable as well.
