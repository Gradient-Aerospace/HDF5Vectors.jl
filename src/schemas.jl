####################
# Schema Interface #
####################

"""
An explicit, recursive description of the logical value type, its encoding, and its
physical storage shape.

Schemas contain no open HDF5 objects. They can therefore be inferred, inspected, and tested
before an HDF5 destination is created.

A new physical schema implements one vertical protocol:

* `encode_value`, `decode_value`, and `encoded_value_type` describe pure conversion.
* `write_schema_node` and `validate_schema_node` describe stored metadata.
* `create_store`, `open_store`, and `physical_length` manage its HDF5 representation.
* `initialize_encoded!` writes a complete encoded batch into a newly created empty store.
* `append_encoded!`, `read_encoded`, and `read_encoded_batch` provide append and read access.

Batch conversion and storage methods can specialize the generic fallbacks when the physical
representation supports more efficient whole-column I/O. The files under `representations`
contain the complete built-in implementations grouped according to this protocol.
"""
abstract type AbstractSchema{T} end

logical_type(::AbstractSchema{T}) where {T} = T

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
