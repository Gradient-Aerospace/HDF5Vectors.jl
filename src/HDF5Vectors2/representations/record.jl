#########################
# Record Representation #
#########################

# Record storage decomposes one logical value into named child schemas. The same schema,
# codec, recursive format, and recursive physical store protocol is available to application
# record codecs; the built-in codecs differ only in how they decompose and compose values.

#################
# Record Codecs #
#################

"""
A pure conversion between a Julia record value and its ordered logical fields.

Each field is encoded recursively by its own schema after `decompose` runs. `compose`
performs the inverse operation after those fields have been decoded.
"""
abstract type AbstractRecordCodec{T} end

logical_type(::AbstractRecordCodec{T}) where {T} = T

struct StructCodec{T, N} <: AbstractRecordCodec{T}
    names::NTuple{N, Symbol}
end

function decompose(codec::StructCodec{T, N}, value::T) where {T, N}
    return ntuple(index -> getfield(value, codec.names[index]), N)
end

function compose(::StructCodec{T}, values::Tuple) where {T}
    return T(values...)
end

struct TupleCodec{T} <: AbstractRecordCodec{T} end

decompose(::TupleCodec{T}, value::T) where {T} = value

function compose(::TupleCodec{T}, values::Tuple) where {T}
    if !(values isa T)
        throw(ArgumentError("Decoded tuple fields do not have the declared type $T."))
    end
    return values
end

struct NamedTupleCodec{T} <: AbstractRecordCodec{T} end

decompose(::NamedTupleCodec{T}, value::T) where {T} = Tuple(value)
compose(::NamedTupleCodec{T}, values::Tuple) where {T} = T(values)

struct StaticArrayCodec{T} <: AbstractRecordCodec{T} end

decompose(::StaticArrayCodec{T}, value::T) where {T} = (value.data,)
compose(::StaticArrayCodec{T}, values::Tuple) where {T} = T(only(values))

#################
# Record Schema #
#################

struct RecordSchema{
    T,
    N,
    C <: AbstractRecordCodec{T},
    Children <: Tuple,
} <: AbstractSchema{T}
    names::NTuple{N, String}
    codec::C
    children::Children
end

function RecordSchema(
    ::Type{T},
    names::NTuple{N, String},
    codec::AbstractRecordCodec{T},
    children::Tuple,
) where {T, N}

    if length(unique(names)) != N
        throw(ArgumentError("A record schema for $T must use unique field names."))
    end

    for name in names
        if isempty(name) || name == "." || occursin('/', name) || occursin('\0', name)
            throw(ArgumentError(
                "The record field name $(repr(name)) for $T cannot be used as one " *
                "HDF5 path component.",
            ))
        end
    end

    if length(children) != N
        throw(ArgumentError(
            "A record schema for $T needs one child for each of its $N fields.",
        ))
    end
    return RecordSchema{T, N, typeof(codec), typeof(children)}(
        names,
        codec,
        children,
    )

end
# A single encoded record is a row-like tuple, but a record batch follows the field-oriented
# HDF5 layout. `count` remains explicit because a record whose fields are all constants has
# no physical child column from which its logical length could be recovered.
struct RecordBatch{Columns <: Tuple}
    columns::Columns
    count::Int
end
function encoded_value_type(schema::RecordSchema)
    child_types = map(encoded_value_type, schema.children)
    return Core.apply_type(Tuple, child_types...)
end
function decompose_record(schema::RecordSchema{T, N}, value::T) where {T, N}

    fields = decompose(schema.codec, value)
    if length(fields) != N
        throw(ArgumentError(
            "The record codec for $T produced $(length(fields)) fields instead of $N.",
        ))
    end

    return fields

end

function encode_value(schema::RecordSchema{T, N}, value::T) where {T, N}
    fields = decompose_record(schema, value)
    return ntuple(
        index -> encode_value(schema.children[index], fields[index]),
        N,
    )
end

function decode_value(schema::RecordSchema{T, N}, encoded::Tuple) where {T, N}
    if length(encoded) != N
        throw(ArgumentError(
            "Encoded record data for $T has $(length(encoded)) fields instead of $N.",
        ))
    end
    fields = ntuple(
        index -> decode_value(schema.children[index], encoded[index]),
        N,
    )
    return compose(schema.codec, fields)
end

function store_record_field!(
    column::Vector{F},
    field,
    value_index,
    field_index,
    type,
) where {F}
    if !(field isa F)
        throw(ArgumentError(
            "The record codec for $type produced a field of type $(typeof(field)) at " *
            "position $field_index instead of $F.",
        ))
    end
    column[value_index] = field
    return nothing
end

function encode_batch(
    schema::RecordSchema{T, N},
    values::AbstractVector{T},
) where {T, N}

    # Logical field columns let each child schema construct its natural encoded batch. A
    # value is decomposed exactly once, preserving the scalar interface's behavior for
    # codecs with computed fields while avoiding an encoded row vector and a later
    # transposition.
    columns = map(
        child -> Vector{logical_type(child)}(undef, length(values)),
        schema.children,
    )
    for (value_index, value) in enumerate(values)
        fields = decompose_record(schema, value)
        for field_index in eachindex(columns)
            store_record_field!(
                columns[field_index],
                fields[field_index],
                value_index,
                field_index,
                T,
            )
        end
    end

    encoded_columns = map(encode_batch, schema.children, columns)
    return RecordBatch(encoded_columns, length(values))

end

function decode_batch(
    schema::RecordSchema{T, N},
    encoded::RecordBatch,
) where {T, N}

    if length(encoded.columns) != N
        throw(ArgumentError(
            "Encoded record data for $T has $(length(encoded.columns)) fields instead " *
            "of $N.",
        ))
    end

    # Each child batch is decoded once as a column. Final values are then constructed
    # directly from those columns, without first allocating encoded row tuples.
    columns = map(decode_batch, schema.children, encoded.columns)
    for (index, column) in enumerate(columns)
        if length(column) != encoded.count
            throw(DimensionMismatch(
                "Decoded record field $index has $(length(column)) values instead of " *
                "$(encoded.count).",
            ))
        end
    end

    values = Vector{T}(undef, encoded.count)
    for value_index in eachindex(values)
        fields = map(column -> column[value_index], columns)
        values[value_index] = compose(schema.codec, fields)
    end
    return values

end

####################
# Schema Inference #
####################

function record_schema(type::Type, context::SchemaContext)

    field_names = fieldnames(type)
    names = Tuple(string(field_name) for field_name in field_names)
    types = fieldtypes(type)
    children = Tuple(
        infer_child_schema(field_type, SchemaContext(context.policy, nothing))
        for field_type in types
    )

    codec = record_codec(type)
    return RecordSchema(type, names, codec, children)

end

record_codec(::Type{T}) where {T <: NamedTuple} = NamedTupleCodec{T}()
record_codec(::Type{T}) where {T <: Tuple} = TupleCodec{T}()

function record_codec(::Type{T}) where {T <: StaticArrays.StaticArray}
    return StaticArrayCodec{T}()
end

function record_codec(::Type{T}) where {T}
    names = fieldnames(T)
    return StructCodec{T, length(names)}(names)
end

###################
# Stored Metadata #
###################

function write_schema_node(group::HDF5.Group, schema::RecordSchema)

    write_common_schema(group, "record", schema)
    write_codec(group, schema.codec)
    group["field_names"] = collect(schema.names)

    children_group = HDF5.create_group(group, "children")
    for (index, child) in enumerate(schema.children)
        child_group = HDF5.create_group(children_group, string(index))
        write_schema_node(child_group, child)
    end
    return nothing

end

function validate_schema_node(group::HDF5.Group, schema::RecordSchema)

    validate_common_schema(group, "record", schema)
    validate_codec(group, schema.codec)
    stored_names = Tuple(String(name) for name in read(group["field_names"]))
    if stored_names != schema.names
        throw(ArgumentError(
            "Stored record fields $stored_names do not match selected fields " *
            "$(schema.names).",
        ))
    end

    children_group = group["children"]
    stored_children = Set(String(name) for name in keys(children_group))
    expected_children = Set(string(index) for index in eachindex(schema.children))
    if stored_children != expected_children
        throw(ArgumentError(
            "Stored record children $stored_children do not match $expected_children.",
        ))
    end
    for index in eachindex(schema.children)
        validate_schema_node(
            children_group[string(index)],
            schema.children[index],
        )
    end
    return schema

end


##################
# Physical Store #
##################

struct RecordStore{Children <: Tuple} <: AbstractStore
    children::Children
end
function create_store(
    group::HDF5.Group,
    schema::RecordSchema{T, N};
    chunk_length,
) where {T, N}

    chunk_length = validate_chunk_length(chunk_length)
    children = ntuple(N) do index
        child_group = HDF5.create_group(group, schema.names[index])
        return create_store(
            child_group,
            schema.children[index];
            chunk_length,
        )
    end
    return RecordStore(children)

end

function open_store(
    group::HDF5.Group,
    schema::RecordSchema{T, N},
) where {T, N}

    validate_store_children(group, schema.names)
    children = ntuple(N) do index
        return open_store(group[schema.names[index]], schema.children[index])
    end

    # All nonconstant columns must describe the same number of records. Running this check
    # while opening catches incomplete or manually altered layouts before they are read.
    store = RecordStore(children)
    physical_length(store)
    return store

end

function physical_length(store::RecordStore)

    record_length = nothing
    for (index, child) in enumerate(store.children)
        child_length = physical_length(child)
        if isnothing(child_length)
            continue
        elseif isnothing(record_length)
            record_length = child_length
        elseif child_length != record_length
            throw(DimensionMismatch(
                "Record child $index has length $child_length, while the other " *
                "nonconstant children have length $record_length.",
            ))
        end
    end
    return record_length

end
###########################
# Record Store Operations #
###########################

# The encoded value type is a property of physical storage. It lets record batches build
# concretely typed child columns without consulting a logical schema or running a codec.
function stored_value_type(store::RecordStore)
    child_types = map(stored_value_type, store.children)
    return Core.apply_type(Tuple, child_types...)
end

function validate_record_value(store::RecordStore, value::Tuple)

    child_count = length(store.children)
    if length(value) != child_count
        throw(ArgumentError(
            "Encoded record data has $(length(value)) fields instead of $child_count.",
        ))
    end

    for index in eachindex(store.children)
        validate_encoded(store.children[index], value[index])
    end
    return value

end
function validate_encoded(store::RecordStore, value::Tuple)
    validate_record_value(store, value)
    return nothing
end

# Record batches arrive with one recursively encoded column per child store. This complete
# preflight happens before the first child changes, so a malformed later column cannot
# leave earlier columns initialized while later ones remain empty.
function validate_encoded_batch(
    store::RecordStore,
    batch::RecordBatch,
    expected_count::Int = batch.count,
)

    if batch.count != expected_count
        throw(DimensionMismatch(
            "An encoded record column has $(batch.count) values instead of " *
            "$expected_count.",
        ))
    elseif length(batch.columns) != length(store.children)
        throw(ArgumentError(
            "Encoded record data has $(length(batch.columns)) fields instead of " *
            "$(length(store.children)).",
        ))
    end

    for index in eachindex(store.children)
        validate_encoded_batch(
            store.children[index],
            batch.columns[index],
            batch.count,
        )
    end
    return nothing

end

function initialize_encoded!(
    store::RecordStore,
    values::AbstractVector{<:Tuple},
)

    # Preflighting the complete batch keeps a bad value in a later record from leaving
    # earlier columns initialized while later columns remain empty.
    for value in values
        validate_record_value(store, value)
    end

    for child_index in eachindex(store.children)
        child = store.children[child_index]
        child_values = Vector{stored_value_type(child)}(undef, length(values))
        for value_index in eachindex(values)
            child_values[value_index] = values[value_index][child_index]
        end
        initialize_encoded!(child, child_values)
    end
    return store

end

function initialize_encoded!(
    store::RecordStore,
    batch::RecordBatch,
)

    # Column validation reaches every nested field before physical mutation. Each child
    # can therefore receive its natural batch representation directly, with no row-to-
    # column rearrangement inside the storage layer.
    validate_encoded_batch(store, batch)

    for index in eachindex(store.children)
        initialize_encoded!(
            store.children[index],
            batch.columns[index],
        )
    end

    return store

end

function read_encoded(store::RecordStore, index::Int)

    record_length = physical_length(store)
    if isnothing(record_length)
        if index < 1
            throw(BoundsError(1:typemax(Int), index))
        end
    elseif index < 1 || index > record_length
        throw(BoundsError(store, index))
    end
    return map(child -> read_encoded(child, index), store.children)

end

function read_encoded(store::RecordStore, indices::UnitRange{Int})

    record_length = physical_length(store)
    if isnothing(record_length)
        if !isempty(indices) && first(indices) < 1
            throw(BoundsError(1:typemax(Int), indices))
        end
    elseif !isempty(indices) && (first(indices) < 1 || last(indices) > record_length)
        throw(BoundsError(store, indices))
    end

    child_columns = map(child -> read_encoded(child, indices), store.children)
    values = Vector{stored_value_type(store)}(undef, length(indices))
    for value_index in eachindex(values)
        values[value_index] = map(column -> column[value_index], child_columns)
    end
    return values

end

function read_encoded_batch(store::RecordStore, indices::UnitRange{Int})

    record_length = physical_length(store)
    if isnothing(record_length)
        if !isempty(indices) && first(indices) < 1
            throw(BoundsError(1:typemax(Int), indices))
        end
    elseif !isempty(indices) && (first(indices) < 1 || last(indices) > record_length)
        throw(BoundsError(store, indices))
    end

    # Child stores retain their natural batch shapes. The schema layer consumes these
    # columns recursively and constructs only the final logical record vector.
    columns = map(child -> read_encoded_batch(child, indices), store.children)
    return RecordBatch(columns, length(indices))

end

function append_encoded!(store::RecordStore, index::Int, value::Tuple)

    # Encoding has recursively prepared every field before this method begins. Passing the
    # known index down the store tree avoids re-reading every child length at each record
    # level and again inside each leaf write.
    for child_index in eachindex(store.children)
        append_encoded!(
            store.children[child_index],
            index,
            value[child_index],
        )
    end
    return store

end
