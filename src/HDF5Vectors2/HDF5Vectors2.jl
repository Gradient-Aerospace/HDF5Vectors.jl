module HDF5Vectors2

using Serialization
import HDF5
import StaticArrays
import ..HDF5Vectors: AbstractHDF5Vector

include("codecs.jl")
include("schemas.jl")
include("inference.jl")
include("format.jl")
include("storage.jl")
include("representations/scalar.jl")
include("representations/blob.jl")
include("representations/constant.jl")
include("representations/dense.jl")
include("representations/record.jl")
include("vector.jl")

export HDF5Vector, create_hdf5_vector, load_hdf5_vector, copy_to_hdf5_vector

public AbstractCodec, AbstractRecordCodec, AbstractSchema
public ScalarSchema, DenseSchema, RecordSchema, BlobSchema, ConstantSchema
public IdentityCodec, CharCodec, SymbolCodec, EnumCodec, JSONCodec, SerializationCodec
public StructCodec, TupleCodec, NamedTupleCodec, StaticArrayCodec, ConstantCodec
public SchemaPolicy, infer_schema, json_schema, serialization_schema
public logical_type, encoded_type, encode_value, decode_value
public codec_identifier, schema_identifier, write_schema, read_schema

end # module HDF5Vectors2
