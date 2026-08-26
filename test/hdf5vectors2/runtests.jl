include(joinpath(@__DIR__, "..", "..", "src", "HDF5Vectors2", "HDF5Vectors2.jl"))

using .HDF5Vectors2:
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
    SerializationCodec,
    StructCodec,
    TupleCodec,
    NamedTupleCodec,
    StaticArrayCodec,
    ConstantCodec,
    SchemaPolicy,
    infer_schema,
    serialization_schema,
    logical_type,
    encoded_type,
    encode_value,
    decode_value,
    write_schema,
    read_schema
import EnumX
import HDF5
import HDF5Vectors
import StaticArrays

include("schema_and_codecs.jl")
include("format.jl")
include("scalar_and_constant_storage.jl")
include("dense_storage.jl")
include("record_storage.jl")
include("blob_storage.jl")
include("public_vector.jl")
include("implementation_comparison.jl")
