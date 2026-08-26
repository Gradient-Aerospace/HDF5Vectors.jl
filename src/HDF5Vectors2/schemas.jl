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

################################
# Encoded Representation Types #
################################

# The generic batch fallback needs a concrete destination type even when the source
# collection is empty. This type is determined entirely by the schema and mirrors one
# value accepted by the corresponding physical store.
encoded_value_type(::ScalarSchema{T, H}) where {T, H} = H
encoded_value_type(::DenseSchema{T, E, H, N}) where {T, E, H, N} = Array{H, N}
encoded_value_type(::BlobSchema) = Vector{UInt8}
encoded_value_type(::ConstantSchema) = Nothing

function encoded_value_type(schema::RecordSchema)
    child_types = map(encoded_value_type, schema.children)
    return Core.apply_type(Tuple, child_types...)
end

##########################
# Pure Value Conversions #
##########################

function validate_dense_value(schema::DenseSchema{T}, value::T) where {T}
    actual_dims = value isa Tuple ? (length(value),) : size(value)
    if actual_dims != schema.dims
        throw(DimensionMismatch(
            "Expected a $T value with dimensions $(schema.dims), but got $actual_dims.",
        ))
    end
    return value
end

function validate_dense_encoding(schema::DenseSchema, encoded::AbstractArray)
    if size(encoded) != schema.dims
        throw(DimensionMismatch(
            "Expected encoded dimensions $(schema.dims), but got $(size(encoded)).",
        ))
    end
    return encoded
end

encode_value(schema::ScalarSchema{T}, value::T) where {T} = encode_value(
    schema.codec,
    value,
)

decode_value(schema::ScalarSchema, value) = decode_value(schema.codec, value)

function encode_value(schema::DenseSchema{T, E, H, N}, value::T) where {T, E, H, N}

    validate_dense_value(schema, value)
    encoded = Array{H, N}(undef, schema.dims)
    for (index, element) in enumerate(value)
        encoded[index] = encode_value(schema.element_codec, element)
    end
    return encoded

end
function decode_value(
    schema::DenseSchema{T, E, H, N},
    encoded::AbstractArray{H, N},
) where {T, E, H, N}

    validate_dense_encoding(schema, encoded)
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

function decode_value(
    schema::DenseSchema{T, E, E, N, IdentityCodec{E}},
    encoded::AbstractArray{E, N},
) where {T, E, N}

    # Identity encoding allows the HDF5 result to become the logical value directly. A
    # dynamic Array read is already the required type, while tuples and static arrays can
    # copy directly into their inline representations without an intermediate Array.
    validate_dense_encoding(schema, encoded)
    if T <: Tuple
        value = Tuple(encoded)
        if !(value isa T)
            throw(ArgumentError("Decoded dense tuple does not have the declared type $T."))
        end
        return value
    elseif T <: StaticArrays.StaticArray
        return T(encoded)
    elseif encoded isa T
        return encoded
    else
        return Array(encoded)::T
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

##########################
# Pure Batch Conversions #
##########################

# The fallback keeps the encoded values in row order. Record and blob storage initially
# use this path, while representations with a natural contiguous layout specialize it
# below. Every value is encoded before the caller creates or changes HDF5 storage.
function encode_batch(
    schema::AbstractSchema{T},
    values::AbstractVector{T},
) where {T}
    encoded = Vector{encoded_value_type(schema)}(undef, length(values))
    for index in eachindex(values)
        encoded[index] = encode_value(schema, values[index])
    end
    return encoded
end

function decode_batch(
    schema::AbstractSchema{T},
    encoded::AbstractVector,
) where {T}
    values = Vector{T}(undef, length(encoded))
    for index in eachindex(values)
        values[index] = decode_value(schema, encoded[index])
    end
    return values
end

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

function encode_batch(
    schema::DenseSchema{T, E, H, N},
    values::AbstractVector{T},
) where {T, E, H, N}

    # Dense HDF5 storage stacks logical values along one final dimension. Filling that
    # array directly avoids allocating one encoded frame per logical value and then
    # copying every frame into the same layout afterward.
    stacked = Array{H, N + 1}(undef, (schema.dims..., length(values)))
    for (value_index, value) in enumerate(values)
        validate_dense_value(schema, value)
        frame = selectdim(stacked, N + 1, value_index)
        for (element_index, element) in enumerate(value)
            frame[element_index] = encode_value(schema.element_codec, element)
        end
    end
    return stacked

end

function decode_batch(
    schema::DenseSchema{T, E, H, N},
    stacked::Array{H, M},
) where {T, E, H, N, M}

    # Only the final dimension counts logical values. Validating all leading dimensions
    # here keeps a malformed physical batch from being interpreted as correctly shaped
    # Julia values.
    expected_dims = (schema.dims..., size(stacked, N + 1))
    if size(stacked) != expected_dims
        throw(DimensionMismatch(
            "Expected an encoded batch with leading dimensions $(schema.dims), but got " *
            "$(size(stacked)).",
        ))
    end

    # Each view borrows the HDF5 read buffer only during reconstruction. Dynamic Arrays
    # receive their own copy, while tuples and static arrays copy into inline storage.
    values = Vector{T}(undef, size(stacked, N + 1))
    for index in eachindex(values)
        values[index] = decode_value(schema, selectdim(stacked, N + 1, index))
    end
    return values

end
