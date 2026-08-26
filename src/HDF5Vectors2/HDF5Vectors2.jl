module HDF5Vectors2

using Serialization
import HDF5
import StaticArrays

include("codecs.jl")
include("schemas.jl")
include("inference.jl")
include("format.jl")
include("storage.jl")

export AbstractCodec, AbstractRecordCodec, AbstractSchema
export ScalarSchema, DenseSchema, RecordSchema, BlobSchema, ConstantSchema
export IdentityCodec, CharCodec, SymbolCodec, EnumCodec, SerializationCodec
export StructCodec, TupleCodec, NamedTupleCodec, StaticArrayCodec, ConstantCodec
export SchemaPolicy, infer_schema, serialization_schema
export logical_type, encoded_type, encode_value, decode_value
export write_schema, read_schema

end # module HDF5Vectors2
