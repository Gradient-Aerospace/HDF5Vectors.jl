module HDF5Vectors2

using Serialization
import HDF5
import StaticArrays

include("codecs.jl")
include("schemas.jl")
include("inference.jl")
include("format.jl")
include("storage.jl")
include("vector.jl")

export AbstractCodec, AbstractRecordCodec, AbstractSchema
export ScalarSchema, DenseSchema, RecordSchema, BlobSchema, ConstantSchema
export IdentityCodec, CharCodec, SymbolCodec, EnumCodec, SerializationCodec
export StructCodec, TupleCodec, NamedTupleCodec, StaticArrayCodec, ConstantCodec
export SchemaPolicy, infer_schema, serialization_schema
export logical_type, encoded_type, encode_value, decode_value
export write_schema, read_schema
export HDF5Vector, create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector

end # module HDF5Vectors2
