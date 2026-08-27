###########################
# Constant Representation #
###########################

# A constant schema has no per-element physical data. Its codec owns the one logical value,
# while the public vector's persisted count records how many times that value occurs.

##################
# Constant Codec #
##################

struct ConstantCodec{T} <: AbstractCodec{T, Nothing}
    value::T
end

encode_value(::ConstantCodec{T}, ::T) where {T} = nothing
decode_value(codec::ConstantCodec, ::Nothing) = codec.value

###################
# Constant Schema #
###################

struct ConstantSchema{T, C <: AbstractCodec{T, Nothing}} <: AbstractSchema{T}
    codec::C
end
encoded_value_type(::ConstantSchema) = Nothing
encode_value(schema::ConstantSchema{T}, value::T) where {T} = encode_value(
    schema.codec,
    value,
)

decode_value(schema::ConstantSchema, ::Nothing) = decode_value(schema.codec, nothing)

####################
# Schema Inference #
####################

function infer_builtin_schema(::Type{Tuple{}}, context::SchemaContext)
    reject_dims(Tuple{}, context.dims)
    value = ()
    return ConstantSchema(ConstantCodec{Tuple{}}(value))
end

function infer_constant_schema(type::Type)

    if !Base.issingletontype(type)
        return unsupported_schema(type, "it has no fields but does not have one value.")
    elseif type === NamedTuple{(), Tuple{}}
        value = (;)
    elseif applicable(type)
        value = type()
    else
        return unsupported_schema(
            type,
            "its constant value cannot be reconstructed through a supported interface.",
        )
    end

    if !(value isa type)
        return unsupported_schema(
            type,
            "its zero-argument constructor returned $(typeof(value)).",
        )
    end
    return ConstantSchema(ConstantCodec{type}(value))

end


###################
# Stored Metadata #
###################

function write_schema_node(group::HDF5.Group, schema::ConstantSchema)
    write_common_schema(group, "constant", schema)
    write_codec(group, schema.codec)
    group["serialized_value"] = serialize_metadata_value(schema.codec.value)
    return nothing
end

function validate_schema_node(group::HDF5.Group, schema::ConstantSchema)

    validate_common_schema(group, "constant", schema)
    validate_codec(group, schema.codec)
    bytes = Vector{UInt8}(read(group["serialized_value"]))
    stored_value = deserialize_metadata_value(bytes)
    if !isequal(stored_value, schema.codec.value)
        throw(ArgumentError(
            "The stored constant does not match the selected constant value.",
        ))
    end
    return schema

end

##################
# Physical Store #
##################

struct ConstantStore <: AbstractStore
    group::HDF5.Group
end
function create_store(
    group::HDF5.Group,
    ::ConstantSchema;
    chunk_length,
)
    validate_chunk_length(chunk_length)
    return ConstantStore(group)
end

function open_store(group::HDF5.Group, ::ConstantSchema)
    validate_store_children(group, ())
    return ConstantStore(group)
end

physical_length(::ConstantStore) = nothing
stored_value_type(::ConstantStore) = Nothing
validate_encoded(::ConstantStore, ::Nothing) = nothing
function validate_encoded_batch(
    ::ConstantStore,
    values::AbstractVector{Nothing},
    expected_count::Int,
)
    return validate_encoded_column_count(values, expected_count)
end

#############################
# Constant Store Operations #
#############################

# Constant values have no physical payload. The vector-level logical length determines how
# many values exist, so initialization and append operations do not change the store.
function initialize_encoded!(
    store::ConstantStore,
    ::AbstractVector{Nothing},
)
    return store
end

function read_encoded(::ConstantStore, index::Int)
    if index < 1
        throw(BoundsError(1:typemax(Int), index))
    end
    return nothing
end

function read_encoded(::ConstantStore, indices::UnitRange{Int})
    if !isempty(indices) && first(indices) < 1
        throw(BoundsError(1:typemax(Int), indices))
    end
    return fill(nothing, length(indices))
end

append_encoded!(store::ConstantStore, ::Int, ::Nothing) = store
