import HDF5Vectors
import JSON3
using Test

const HDF5Vectors2 = HDF5Vectors.HDF5Vectors2

using HDF5Vectors.HDF5Vectors2:
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
import EnumX
import HDF5
import StaticArrays

include("schema_and_codecs.jl")
include("format.jl")
include("scalar_and_constant_storage.jl")
include("dense_storage.jl")
include("record_storage.jl")
include("blob_storage.jl")
include("public_vector.jl")
include("implementation_comparison.jl")
