# This file is included inside each test module. Imports and fixture types therefore remain
# local to that module, allowing every test file to run alone without creating shared test
# state or relying on the order used by runtests.jl.

import EnumX
import HDF5
import HDF5Vectors
import JSON3
import StaticArrays
using Test

using HDF5Vectors:
    AbstractCodec,
    AbstractRecordCodec,
    AbstractSchema,
    ScalarSchema,
    DenseSchema,
    RecordSchema,
    BlobSchema,
    ConstantSchema,
    IdentityCodec,
    CharCodec,
    SymbolCodec,
    EnumCodec,
    JSONCodec,
    SerializationCodec,
    StructCodec,
    TupleCodec,
    NamedTupleCodec,
    StaticArrayCodec,
    ConstantCodec,
    SchemaPolicy,
    infer_schema,
    json_schema,
    serialization_schema,
    logical_type,
    encoded_type,
    encode_value,
    decode_value,
    write_schema,
    read_schema

@enum PrototypeUInt8Enum::UInt8 prototype_zero = 0 prototype_max = 255

struct PrototypePoint
    x::Float64
    y::Int64
end

struct PrototypeSample
    point::PrototypePoint
    label::Symbol
    values::Vector{Float64}
end

function Base.:(==)(first::PrototypeSample, second::PrototypeSample)
    return first.point == second.point &&
        first.label == second.label &&
        first.values == second.values
end

struct PrototypeSingleton1{Value}
end

struct PrototypeSingleton2{Value}
end

struct PrototypeUnconstructibleSingleton
    PrototypeUnconstructibleSingleton(::Nothing) = new()
end

mutable struct PrototypeMutableMarker
end

abstract type PrototypeAbstractValue end

struct PrototypeConcreteValue <: PrototypeAbstractValue
    value::Int64
end

# This application codec is intentionally defined using only the public extension
# interface. It is reused at the root of a vector and recursively inside a record, and its
# schema must survive ordinary untyped loading without any package-owned codec registry.
struct PrototypeGrade
    label::String
end

struct PrototypeGradeCodec <: AbstractCodec{PrototypeGrade, UInt8} end

function HDF5Vectors.encode_value(::PrototypeGradeCodec, grade::PrototypeGrade)
    return UInt8(only(grade.label))
end

function HDF5Vectors.decode_value(::PrototypeGradeCodec, value::UInt8)
    return PrototypeGrade(string(Char(value)))
end

function HDF5Vectors.infer_schema(::Type{PrototypeGrade}; kwargs...)
    return ScalarSchema(PrototypeGradeCodec())
end

# JSON is another logical conversion over scalar storage rather than a separate physical
# schema. Declaring JSON3's struct mapping makes this small type readable in both
# directions; selecting `json_schema` opts only this application type into that codec.
struct PrototypeJSONValue
    name::String
    values::Vector{Int64}
end

function Base.:(==)(first::PrototypeJSONValue, second::PrototypeJSONValue)
    return first.name == second.name && first.values == second.values
end

JSON3.StructTypes.StructType(::Type{PrototypeJSONValue}) = JSON3.StructTypes.Struct()

function HDF5Vectors.infer_schema(::Type{PrototypeJSONValue}; kwargs...)
    return json_schema(PrototypeJSONValue)
end

struct PrototypeGradedValue
    grade::PrototypeGrade
    value::Float64
end

struct PrototypeJSONRecord
    details::PrototypeJSONValue
    value::Float64
end

function Base.:(==)(first::PrototypeJSONRecord, second::PrototypeJSONRecord)
    return first.details == second.details && first.value == second.value
end

function test_schema_round_trip(schema, value)
    encoded = encode_value(schema, value)
    @test decode_value(schema, encoded) == value
end
