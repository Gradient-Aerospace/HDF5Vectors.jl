"""
HDF5Vectors provides an append-only `AbstractVector` whose values are stored in an HDF5
file. A vector's schema describes its logical type, encoding, and physical HDF5 layout.
"""
module HDF5Vectors

using Serialization
import HDF5
import StaticArrays

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

public AbstractCodec, AbstractRecordCodec, AbstractSchema, AbstractStore
public ScalarSchema, DenseSchema, RecordSchema, BlobSchema, ConstantSchema
public IdentityCodec, CharCodec, SymbolCodec, EnumCodec, JSONCodec, SerializationCodec
public StructCodec, TupleCodec, NamedTupleCodec, StaticArrayCodec, ConstantCodec
public SchemaPolicy, infer_schema, json_schema, serialization_schema
public logical_type, encoded_type, encode_value, decode_value
public decompose, compose, encoded_value_type, encode_batch, decode_batch
public codec_identifier, schema_identifier, write_schema, read_schema
public write_schema_node, validate_schema_node, write_common_schema, validate_common_schema
public write_encoded_type, validate_encoded_type, write_codec, validate_codec
public create_store, open_store, physical_length, initialize_encoded!, append_encoded!
public read_encoded, read_encoded_batch, validate_encoded_batch

end # module HDF5Vectors
