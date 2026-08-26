module HDF5Vectors2

using Serialization
import StaticArrays

include("codecs.jl")
include("schemas.jl")
include("inference.jl")

export AbstractCodec, AbstractRecordCodec, AbstractSchema
export ScalarSchema, DenseSchema, RecordSchema, BlobSchema, ConstantSchema
export IdentityCodec, CharCodec, SymbolCodec, EnumCodec, SerializationCodec
export StructCodec, TupleCodec, NamedTupleCodec, StaticArrayCodec, ConstantCodec
export SchemaPolicy, infer_schema, serialization_schema
export logical_type, encoded_type, encode_value, decode_value

end # module HDF5Vectors2
