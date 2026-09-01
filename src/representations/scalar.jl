#########################
# Scalar Representation #
#########################

# This file is a complete built-in representation example. It defines logical codecs, the
# schema that owns them, schema inference, human-readable format metadata, and the physical
# HDF5 store used to execute that schema.

#################
# Scalar Codecs #
#################

"""A codec that stores a supported Julia scalar without conversion."""
struct IdentityCodec{T} <: AbstractCodec{T, T} end

encode_value(::IdentityCodec{T}, value::T) where {T} = value
decode_value(::IdentityCodec{T}, value::T) where {T} = value

"""A codec that stores a `Char` as its `Int32` Unicode code point."""
struct CharCodec <: AbstractCodec{Char, Int32} end

encode_value(::CharCodec, value::Char) = Int32(value)
decode_value(::CharCodec, value::Int32) = Char(value)

"""A codec that stores a `Symbol` as a `String`."""
struct SymbolCodec <: AbstractCodec{Symbol, String} end

encode_value(::SymbolCodec, value::Symbol) = String(value)
decode_value(::SymbolCodec, value::String) = Symbol(value)

# JSON storage has the same physical shape as ordinary scalar string storage. The codec
# belongs to the core schema vocabulary so schemas containing it can be constructed and
# deserialized without JSON3. Its conversion methods are supplied only when JSON3 loads.
"""A codec that stores a logical value of type `T` as a JSON string using JSON3."""
struct JSONCodec{T} <: AbstractCodec{T, String} end

"""A codec that stores an enum using its integer base type `H`."""
struct EnumCodec{T, H} <: AbstractCodec{T, H} end

encode_value(::EnumCodec{T, H}, value::T) where {T, H} = H(value)
decode_value(::EnumCodec{T, H}, value::H) where {T, H} = T(value)


#################
# Scalar Schema #
#################

"""
    ScalarSchema(codec::AbstractCodec)

Describes one logical value encoded as one scalar HDF5-compatible value.
"""
struct ScalarSchema{T, H, C <: AbstractCodec{T, H}} <: AbstractSchema{T}
    codec::C
end

"""
    json_schema(type::Type)

Returns a scalar schema that stores each value as a JSON string. JSON3 must be loaded before
values can be encoded or decoded.
"""
json_schema(::Type{T}) where {T} = ScalarSchema(JSONCodec{T}())

encoded_type(::ScalarSchema{T, H}) where {T, H} = H
encoded_value_type(::ScalarSchema{T, H}) where {T, H} = H
encode_value(schema::ScalarSchema{T}, value::T) where {T} = encode_value(
    schema.codec,
    value,
)

decode_value(schema::ScalarSchema, value) = decode_value(schema.codec, value)
# HDF5 can consume and produce the same vector representation used by an identity scalar
# codec. Returning the collection directly avoids a copy on both sides of the boundary.
function encode_batch(
    ::ScalarSchema{T, T, IdentityCodec{T}},
    values::AbstractVector{T},
) where {T}
    return values
end

function decode_batch(
    ::ScalarSchema{T, T, IdentityCodec{T}},
    encoded::Vector{T},
) where {T}
    return encoded
end

####################
# Schema Inference #
####################

const hdf5_scalar_types = Union{
    Bool,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

native_scalar_schema(type::Type) = ScalarSchema(IdentityCodec{type}())

function infer_builtin_schema(type::Type{<:hdf5_scalar_types}, context::SchemaContext)
    reject_dims(type, context.dims)
    return ScalarSchema(IdentityCodec{type}())
end

function infer_builtin_schema(::Type{String}, context::SchemaContext)
    reject_dims(String, context.dims)
    return ScalarSchema(IdentityCodec{String}())
end

function infer_builtin_schema(::Type{Char}, context::SchemaContext)
    reject_dims(Char, context.dims)
    return ScalarSchema(CharCodec())
end

function infer_builtin_schema(::Type{Symbol}, context::SchemaContext)
    reject_dims(Symbol, context.dims)
    return ScalarSchema(SymbolCodec())
end

function infer_builtin_schema(
    type::Type{T},
    context::SchemaContext,
) where {H <: hdf5_scalar_types, T <: Enum{H}}
    reject_dims(type, context.dims)
    return ScalarSchema(EnumCodec{T, H}())
end

function infer_builtin_schema(
    type::Type{T},
    context::SchemaContext,
) where {H, T <: Enum{H}}
    reject_dims(type, context.dims)
    return unsupported_schema(type, "its enum base type $H is not HDF5-native.")
end


###################
# Stored Metadata #
###################

function write_schema_node(group::HDF5.Group, schema::ScalarSchema)
    write_common_schema(group, "scalar", schema)
    write_encoded_type(group, schema)
    write_codec(group, schema.codec)
    return nothing
end

function validate_schema_node(group::HDF5.Group, schema::ScalarSchema)
    validate_common_schema(group, "scalar", schema)
    validate_encoded_type(group, schema)
    validate_codec(group, schema.codec)
    return schema
end


##################
# Physical Store #
##################

struct ScalarStore{H} <: AbstractStore
    dataset::HDF5.Dataset
end
function create_store(
    group::HDF5.Group,
    schema::ScalarSchema{T, H};
    chunk_length,
) where {T, H}

    dataspace = HDF5.dataspace((0,), (-1,))
    dataset = HDF5.create_dataset(
        group,
        "values",
        H,
        dataspace;
        chunk = (chunk_length,),
    )
    return ScalarStore{H}(dataset)

end

function open_store(group::HDF5.Group, ::ScalarSchema{T, H}) where {T, H}

    validate_store_children(group, ("values",))
    dataset = group["values"]
    if ndims(dataset) != 1
        throw(DimensionMismatch(
            "Scalar storage must be one-dimensional, but its size is $(size(dataset)).",
        ))
    elseif !dataset_matches_encoded_type(dataset, H)
        throw(ArgumentError(
            "Scalar storage does not use the HDF5 datatype required for $H.",
        ))
    end
    return ScalarStore{H}(dataset)

end

physical_length(store::ScalarStore) = length(store.dataset)

###########################
# Scalar Store Operations #
###########################

function initialize_encoded!(
    store::ScalarStore{H},
    values::AbstractVector{H},
) where {H}

    if isempty(values)
        return store
    end

    HDF5.set_extent_dims(store.dataset, (length(values),))
    store.dataset[:] = values
    return store

end

function read_encoded(store::ScalarStore{H}, index::Int) where {H}
    return read(store.dataset, H, index)
end

function read_encoded(store::ScalarStore{H}, indices::UnitRange{Int}) where {H}
    if isempty(indices)
        return H[]
    end
    return read(store.dataset, H, indices)
end

function validate_encoded_batch(
    ::ScalarStore{H},
    values::AbstractVector{H},
    expected_count::Int,
) where {H}
    return validate_encoded_column_count(values, expected_count)
end
function append_encoded!(store::ScalarStore{H}, index::Int, value::H) where {H}
    HDF5.set_extent_dims(store.dataset, (index,))
    store.dataset[index] = value
    return store
end
