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

logical_type(::AbstractCodec{T, H}) where {T, H} = T
encoded_type(::AbstractCodec{T, H}) where {T, H} = H
