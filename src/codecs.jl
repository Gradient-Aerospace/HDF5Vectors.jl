###################
# Codec Interface #
###################

"""
A pure conversion between a Julia value of type `T` and an encoded value of type `H`.

Codecs know nothing about HDF5 objects or vector operations. This makes their round-trip
behavior testable independently of physical storage. An application codec needs
`encode_value` and `decode_value` methods, and its logical type selects the completed schema
through [`infer_schema`](@ref). The stored schema serializes the codec itself for exact
untyped loading; [`codec_identifier`](@ref) is descriptive metadata rather than a registry.
"""
abstract type AbstractCodec{T, H} end

"""
    logical_type(codec_or_schema)

Returns the Julia value type represented by a codec or schema.
"""
logical_type(::AbstractCodec{T, H}) where {T, H} = T

"""
    encoded_type(codec_or_schema)

Returns the scalar HDF5-compatible type produced by a scalar or dense codec or schema.
"""
encoded_type(::AbstractCodec{T, H}) where {T, H} = H

"""
    encode_value(codec_or_schema, value)

Converts one logical Julia value into the representation described by a codec or schema.
Custom codec methods should be pure and should not access HDF5 objects.
"""
function encode_value end

"""
    decode_value(codec_or_schema, encoded)

Reconstructs one logical Julia value from the representation produced by
[`encode_value`](@ref).
"""
function decode_value end
