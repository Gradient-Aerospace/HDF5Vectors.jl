# API Reference

```@meta
CurrentModule = HDF5Vectors
```

## Ordinary Vector Interface

These four exported names form the common user interface.

```@docs
HDF5Vector
create_hdf5_vector
copy_to_hdf5_vector
load_hdf5_vector
```

`HDF5Vector` also supports `length`, `size`, `eltype`, `push!`, scalar and nonscalar `getindex`, `collect`, iteration, broadcasting, `map`, and the other applicable operations supplied by `AbstractVector`. It is append-only and does not implement `setindex!` or deletion.

## Schema Inference

These public names are normally accessed as `HDF5Vectors.name` rather than imported.

```@docs
SchemaPolicy
infer_schema
json_schema
serialization_schema
```

## Schemas

```@docs
AbstractSchema
ScalarSchema
DenseSchema
RecordSchema
BlobSchema
ConstantSchema
logical_type
encoded_type
encoded_value_type
encode_batch
decode_batch
```

## Codecs

```@docs
AbstractCodec
AbstractRecordCodec
encode_value
decode_value
decompose
compose
IdentityCodec
CharCodec
SymbolCodec
EnumCodec
JSONCodec
SerializationCodec
StructCodec
TupleCodec
NamedTupleCodec
StaticArrayCodec
ConstantCodec
```

## Stored Schema Metadata

```@docs
schema_identifier
codec_identifier
write_schema
read_schema
write_schema_node
validate_schema_node
write_common_schema
validate_common_schema
write_encoded_type
validate_encoded_type
write_codec
validate_codec
```

## Physical Store Protocol

These methods are needed only by packages that define a new physical schema. [Custom Schemas](custom_schemas.md) explains how they fit together.

```@docs
AbstractStore
create_store
open_store
physical_length
initialize_encoded!
append_encoded!
read_encoded
read_encoded_batch
validate_encoded_batch
```
