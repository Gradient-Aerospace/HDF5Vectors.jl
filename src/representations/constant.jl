###########################
# Constant Representation #
###########################

# A constant schema has no per-element physical data. Its codec owns the one logical value,
# while the public vector's persisted count records how many times that value occurs.

##################
# Constant Codec #
##################

"""A codec that represents every element with one stored constant value."""
struct ConstantCodec{T} <: AbstractCodec{T, Nothing}
    value::T
end

encode_value(::ConstantCodec{T}, ::T) where {T} = nothing
decode_value(codec::ConstantCodec, ::Nothing) = codec.value

###################
# Constant Schema #
###################

"""
    ConstantSchema(codec::AbstractCodec{T, Nothing})

Describes a logical type with one stored constant value and no per-element physical data.
"""
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
    return nothing
end

function validate_schema_node(group::HDF5.Group, schema::ConstantSchema)
    validate_common_schema(group, "constant", schema)
    validate_codec(group, schema.codec)
    return schema
end

##################
# Physical Store #
##################

struct ConstantStore <: AbstractStore end

function create_store(
    ::HDF5.Group,
    ::ConstantSchema;
    chunk_length,
)
    return ConstantStore()
end

function open_store(group::HDF5.Group, ::ConstantSchema)
    validate_store_children(group, ())
    return ConstantStore()
end

physical_length(::ConstantStore) = nothing
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

read_encoded(::ConstantStore, ::Int) = nothing

function read_encoded(::ConstantStore, indices::UnitRange{Int})
    return fill(nothing, length(indices))
end

append_encoded!(store::ConstantStore, ::Int, ::Nothing) = store
