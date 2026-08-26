#################
# Scalar Codecs #
#################

"""
A pure conversion between a Julia value of type `T` and an encoded value of type `H`.

Codecs know nothing about HDF5 objects or vector operations. This makes their round-trip
behavior testable independently of physical storage.
"""
abstract type AbstractCodec{T, H} end

logical_type(::AbstractCodec{T, H}) where {T, H} = T
encoded_type(::AbstractCodec{T, H}) where {T, H} = H

struct IdentityCodec{T} <: AbstractCodec{T, T} end

encode_value(::IdentityCodec{T}, value::T) where {T} = value
decode_value(::IdentityCodec{T}, value::T) where {T} = value

struct CharCodec <: AbstractCodec{Char, Int32} end

encode_value(::CharCodec, value::Char) = Int32(value)
decode_value(::CharCodec, value::Int32) = Char(value)

struct SymbolCodec <: AbstractCodec{Symbol, String} end

encode_value(::SymbolCodec, value::Symbol) = String(value)
decode_value(::SymbolCodec, value::String) = Symbol(value)

struct EnumCodec{T, H} <: AbstractCodec{T, H} end

encode_value(::EnumCodec{T, H}, value::T) where {T, H} = H(value)
decode_value(::EnumCodec{T, H}, value::H) where {T, H} = T(value)

struct SerializationCodec{T} <: AbstractCodec{T, Vector{UInt8}} end

function encode_value(::SerializationCodec{T}, value::T) where {T}
    io = IOBuffer()
    Serialization.serialize(io, value)
    return take!(io)
end

function decode_value(::SerializationCodec{T}, bytes::Vector{UInt8}) where {T}
    value = Serialization.deserialize(IOBuffer(bytes))
    if !(value isa T)
        throw(ArgumentError(
            "Serialized data for $T produced a value of type $(typeof(value)).",
        ))
    end
    return value
end

struct ConstantCodec{T} <: AbstractCodec{T, Nothing}
    value::T
end

encode_value(::ConstantCodec{T}, ::T) where {T} = nothing
decode_value(codec::ConstantCodec, ::Nothing) = codec.value

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
