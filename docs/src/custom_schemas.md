# Custom Schemas

```@meta
CurrentModule = HDF5Vectors
```

A codec should be preferred whenever a new logical type can reuse scalar, dense, record, blob, or constant storage. A new schema is appropriate only when the physical HDF5 representation itself must change. For example, a package might add a ragged numeric-array representation with one numeric dataset and one offset dataset.

This interface is intended primarily for package authors. Application code continues to use [`create_hdf5_vector`](@ref), [`copy_to_hdf5_vector`](@ref), and [`load_hdf5_vector`](@ref).

## The Three Layers

Each built-in representation follows the same division:

1. A schema is an immutable description of a logical type, its encoded value, and the physical layout. It contains no open HDF5 objects.
2. Pure conversion methods encode and decode values according to that schema.
3. A store holds the open HDF5 objects and reads or writes values that are already encoded.

Schema inference finishes before a destination group is created. Bulk conversion also finishes before physical storage is initialized. This makes codecs independently testable and keeps user conversion errors outside the HDF5 mutation layer.

The files under `src/representations` are complete examples rather than hidden alternate implementations. `scalar.jl` is the smallest ordinary representation, `constant.jl` shows storage without a per-element payload, `dense.jl` shows a specialized batch shape, `blob.jl` manages two coordinated datasets, and `record.jl` demonstrates recursive child schemas and stores.

## Defining the Schema

A schema is a subtype of [`AbstractSchema{T}`](@ref), where `T` is the logical vector element type:

```julia
struct RaggedSchema{T, E} <: HDF5Vectors.AbstractSchema{T}
end
```

The schema normally stores any codecs, dimensions, or child schemas needed to describe one value. [`logical_type`](@ref) is supplied by `AbstractSchema`. A new schema provides these pure conversion methods:

```julia
HDF5Vectors.encoded_value_type(schema::RaggedSchema)
HDF5Vectors.encode_value(schema::RaggedSchema{T}, value::T)
HDF5Vectors.decode_value(schema::RaggedSchema, encoded)
```

`encoded_value_type` describes one encoded logical value. `encode_value` and `decode_value` must round-trip without consulting an HDF5 object. The generic batch conversion methods use these scalar methods and a `Vector` of encoded values; a schema can specialize `HDF5Vectors.encode_batch` and `HDF5Vectors.decode_batch` when its natural batch representation is more efficient.

## Recording the Schema

Each schema writes a readable description beneath `metadata/schema`. The following methods write that description and validate its compatibility with a selected schema:

```julia
function HDF5Vectors.write_schema_node(group::HDF5.Group, schema::RaggedSchema)
    HDF5Vectors.write_common_schema(group, "ragged", schema)
    group["element_type"] = string(eltype(HDF5Vectors.encoded_value_type(schema)))
    return nothing
end

function HDF5Vectors.validate_schema_node(group::HDF5.Group, schema::RaggedSchema)
    HDF5Vectors.validate_common_schema(group, "ragged", schema)
    # The implementation would also validate its element type and any other metadata.
    return schema
end
```

The schema-specific metadata should use ordinary HDF5 values wherever possible so external readers can understand the representation. Validation should cover the metadata needed to select the same physical representation, such as encoded datatypes, dimensions, and child layouts. It need not require logical details to equal the stored Julia schema because an explicitly supplied schema is authoritative and may intentionally describe a migrated Julia type. The store's `open_store` method separately validates the physical groups and datasets. [`schema_identifier`](@ref) supplies the implementation identifier recorded by `write_common_schema`; an extension can specialize it when that identifier must survive a Julia type rename.

The complete schema is also serialized by [`write_schema`](@ref). This is what lets ordinary untyped loading recover an extension schema without a registry inside HDF5Vectors. The extension package and its schema type must be loaded before that Julia metadata can be deserialized.

## Defining the Store

A physical store is a subtype of `HDF5Vectors.AbstractStore` containing the open datasets or child stores needed by the schema:

```julia
struct RaggedStore <: HDF5Vectors.AbstractStore
    values::HDF5.Dataset
    stops::HDF5.Dataset
end
```

Creation and loading are handled by three methods:

```julia
HDF5Vectors.create_store(group::HDF5.Group, schema::RaggedSchema; chunk_length)
HDF5Vectors.open_store(group::HDF5.Group, schema::RaggedSchema)
HDF5Vectors.physical_length(store::RaggedStore)
```

`create_store` receives the already-created `data` group and creates the representation beneath it. `open_store` opens and validates an existing physical layout. `physical_length` returns the number of logical values represented by the store, or `nothing` when the representation has no physical length of its own. Public loading compares this result with `metadata/count`.

The store then implements the append-only operations used by `HDF5Vector`:

```julia
HDF5Vectors.initialize_encoded!(store::RaggedStore, encoded_batch)
HDF5Vectors.append_encoded!(store::RaggedStore, next_index::Int, encoded_value)
HDF5Vectors.read_encoded(store::RaggedStore, index::Int)
HDF5Vectors.read_encoded(store::RaggedStore, indices::UnitRange{Int})
```

`initialize_encoded!` fills a newly created empty store during `copy_to_hdf5_vector`. `append_encoded!` receives the next logical one-based index during `push!`. The scalar and range read methods return encoded values for the schema layer to decode. A store whose natural range representation differs from `Vector{encoded_value_type(schema)}` can specialize `HDF5Vectors.read_encoded_batch` as dense and record storage do.

Stores that can be nested inside `RecordSchema` must also implement `HDF5Vectors.validate_encoded_batch(store, batch, expected_count)`. This checks the complete prepared batch before any record child is initialized.

## Selecting the Schema

The logical type selects the completed schema through [`infer_schema`](@ref):

```julia
function HDF5Vectors.infer_schema(
    ::Type{MyRaggedValue};
    dims = nothing,
    policy = HDF5Vectors.SchemaPolicy(),
)
    if !isnothing(dims)
        throw(ArgumentError("MyRaggedValue does not accept declared dimensions."))
    end
    return RaggedSchema{MyRaggedValue, Float64}()
end
```

The method should either honor or explicitly reject relevant inference options. Recursive inference uses this same public dispatch point, so the schema can be selected at the vector root or within a record.

## Testing a Schema

A useful schema test proceeds from the pure layer to the physical layer:

1. `encode_value` and `decode_value` round-trip representative values and failures.
2. `write_schema` and `read_schema` round-trip both typed and untyped metadata.
3. Store creation, initialization, append, scalar read, range read, and reopening preserve the encoded values.
4. The public create, copy, push, collect, and load operations preserve logical values.
5. Empty copies and empty loads work.
6. Corrupt or incomplete physical layouts are rejected when opened.
7. The resulting HDF5 paths and datatypes match the format promised to external readers.

If the schema will be used recursively, tests should also place it inside a record and exercise a bulk copy. This verifies its batch validation and its interaction with the common logical count.
