###################
# Storage Schemas #
###################

"""
An explicit, recursive description of the logical value type, its encoding, and its
physical storage shape.

Schemas contain no open HDF5 objects. They can therefore be inferred, inspected, and tested
before an HDF5 destination is created.
"""
abstract type AbstractSchema{T} end

logical_type(::AbstractSchema{T}) where {T} = T

struct ScalarSchema{T, H, C <: AbstractCodec{T, H}} <: AbstractSchema{T}
    codec::C
end

encoded_type(::ScalarSchema{T, H}) where {T, H} = H

struct DenseSchema{T, E, H, N, C <: AbstractCodec{E, H}} <: AbstractSchema{T}
    dims::NTuple{N, Int}
    element_codec::C
end

function DenseSchema(
    ::Type{T},
    dims::NTuple{N, Int},
    element_codec::AbstractCodec{E, H},
) where {T, E, H, N}
    return DenseSchema{T, E, H, N, typeof(element_codec)}(dims, element_codec)
end

encoded_type(::DenseSchema{T, E, H}) where {T, E, H} = H

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

struct BlobSchema{T, C <: AbstractCodec{T, Vector{UInt8}}} <: AbstractSchema{T}
    codec::C
end

struct ConstantSchema{T, C <: AbstractCodec{T, Nothing}} <: AbstractSchema{T}
    codec::C
end

##########################
# Pure Value Conversions #
##########################

encode_value(schema::ScalarSchema{T}, value::T) where {T} = encode_value(
    schema.codec,
    value,
)

decode_value(schema::ScalarSchema, value) = decode_value(schema.codec, value)

function encode_value(schema::DenseSchema{T, E, H, N}, value::T) where {T, E, H, N}

    actual_dims = value isa Tuple ? (length(value),) : size(value)
    if actual_dims != schema.dims
        throw(DimensionMismatch(
            "Expected a $T value with dimensions $(schema.dims), but got $actual_dims.",
        ))
    end

    encoded = Array{H, N}(undef, schema.dims)
    for (index, element) in enumerate(value)
        encoded[index] = encode_value(schema.element_codec, element)
    end
    return encoded

end
function decode_value(
    schema::DenseSchema{T, E, H, N},
    encoded::Array{H, N},
) where {T, E, H, N}

    if size(encoded) != schema.dims
        throw(DimensionMismatch(
            "Expected encoded dimensions $(schema.dims), but got $(size(encoded)).",
        ))
    end

    decoded = Array{E, N}(undef, schema.dims)
    for index in eachindex(encoded)
        decoded[index] = decode_value(schema.element_codec, encoded[index])
    end

    if T <: Tuple
        value = Tuple(decoded)
        if !(value isa T)
            throw(ArgumentError("Decoded dense tuple does not have the declared type $T."))
        end
        return value
    elseif T <: StaticArrays.StaticArray
        return T(decoded)
    else
        return decoded::T
    end

end
function encode_value(schema::RecordSchema{T, N}, value::T) where {T, N}
    fields = decompose(schema.codec, value)
    if length(fields) != N
        throw(ArgumentError(
            "The record codec for $T produced $(length(fields)) fields instead of $N.",
        ))
    end
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

encode_value(schema::BlobSchema{T}, value::T) where {T} = encode_value(schema.codec, value)
decode_value(schema::BlobSchema, bytes::Vector{UInt8}) = decode_value(schema.codec, bytes)

encode_value(schema::ConstantSchema{T}, value::T) where {T} = encode_value(
    schema.codec,
    value,
)

decode_value(schema::ConstantSchema, ::Nothing) = decode_value(schema.codec, nothing)
