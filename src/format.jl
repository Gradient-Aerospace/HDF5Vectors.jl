#########################
# Stored Schema Format  #
#########################

const format_name = "HDF5Vectors"
const format_version = Int64(1)

"""Writes one schema implementation's readable metadata node."""
function write_schema_node end

"""Validates one stored metadata node against a selected schema implementation."""
function validate_schema_node end

# Schema metadata has two complementary forms. The ordinary HDF5 tree describes every
# physical representation for people and non-Julia readers. A Julia-serialized schema
# reconstructs the exact codec objects without requiring HDF5Vectors itself to know every
# application codec type. Typed loading can instead repeat public schema inference and
# validate the result against the ordinary tree.

function serialize_metadata_value(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return take!(io)
end

function deserialize_metadata_value(bytes::Vector{UInt8})
    return Serialization.deserialize(IOBuffer(bytes))
end

function implementation_identifier(value)
    type = typeof(value)
    return string(parentmodule(type), ".", nameof(type))
end

"""
    schema_identifier(schema::AbstractSchema)

Returns the stable, human-readable identifier stored for a schema implementation. The
default uses the implementation type. An application can specialize this function when
the identifier must remain stable across a type rename.
"""
schema_identifier(schema::AbstractSchema) = implementation_identifier(schema)

"""
    codec_identifier(codec)

Returns the stable, human-readable identifier stored for a codec implementation. The
default uses the implementation type. An application can specialize this function when
the identifier must remain stable across a type rename.
"""
codec_identifier(codec) = implementation_identifier(codec)

function write_inference_options(metadata_group::HDF5.Group, options)

    metadata_group["schema_was_inferred"] = !isnothing(options)
    if isnothing(options)
        return nothing
    end

    metadata_group["dimensions_were_declared"] = !isnothing(options.dims)
    metadata_group["dimensions"] = if isnothing(options.dims)
        Int64[]
    else
        Int64[options.dims...,]
    end
    metadata_group["portable"] = options.policy.portable
    metadata_group["serialize_arrays"] = options.policy.serialize_arrays
    metadata_group["serialize_nonconcrete"] = options.policy.serialize_nonconcrete
    return nothing

end

function read_stored_bool(group::HDF5.Group, name::AbstractString)
    value = read(group[name])
    if !(value isa Bool)
        throw(ArgumentError("Stored schema option $name must be Bool; got $value."))
    end
    return value
end

function read_inference_options(metadata_group::HDF5.Group)

    if !read_stored_bool(metadata_group, "schema_was_inferred")
        return nothing
    end

    dimensions_were_declared = read_stored_bool(
        metadata_group,
        "dimensions_were_declared",
    )
    dims = if dimensions_were_declared
        Tuple(Int(dimension) for dimension in read(metadata_group["dimensions"]))
    else
        nothing
    end
    policy = SchemaPolicy(;
        portable = read_stored_bool(metadata_group, "portable"),
        serialize_arrays = read_stored_bool(metadata_group, "serialize_arrays"),
        serialize_nonconcrete = read_stored_bool(
            metadata_group,
            "serialize_nonconcrete",
        ),
    )
    return (; dims, policy)

end

"""
    write_schema(group::HDF5.Group, schema::AbstractSchema)

Writes the versioned logical type and complete storage schema into a new `metadata` child
of `group`. Custom schema and codec implementations are serialized as ordinary Julia
metadata and also describe themselves through the extensible schema-node interface.
"""
function write_schema(
    group::HDF5.Group,
    schema::AbstractSchema;
    inference_options = nothing,
)

    type = logical_type(schema)
    metadata_group = HDF5.create_group(group, "metadata")
    metadata_group["format_name"] = format_name
    metadata_group["format_version"] = format_version
    metadata_group["logical_type"] = string(type)
    metadata_group["serialized_schema"] = serialize_metadata_value(schema)
    write_inference_options(metadata_group, inference_options)

    schema_group = HDF5.create_group(metadata_group, "schema")
    write_schema_node(schema_group, schema)
    return schema

end


"""
    read_schema(group::HDF5.Group)
    read_schema(group::HDF5.Group, type::Type)
    read_schema(group::HDF5.Group, schema::AbstractSchema)

Reads and validates the exact schema stored by [`write_schema`](@ref). Untyped loading
deserializes the stored schema, allowing application-defined codecs to reconstruct without
a package-owned registry. Typed loading repeats public schema inference when the vector was
created from a type. Supplying an explicit schema avoids metadata deserialization entirely.
"""
function read_schema(group::HDF5.Group)

    metadata_group = group["metadata"]
    validate_format(metadata_group)
    bytes = Vector{UInt8}(read(metadata_group["serialized_schema"]))
    schema = deserialize_metadata_value(bytes)
    if !(schema isa AbstractSchema)
        throw(ArgumentError(
            "Stored schema metadata produced a value of type $(typeof(schema)).",
        ))
    end
    validate_type_name(metadata_group, logical_type(schema))
    return read_schema(group, schema)

end

function read_schema(group::HDF5.Group, type::Type)

    metadata_group = group["metadata"]
    validate_format(metadata_group)
    validate_type_name(metadata_group, type)
    options = read_inference_options(metadata_group)
    if isnothing(options)
        bytes = Vector{UInt8}(read(metadata_group["serialized_schema"]))
        schema = deserialize_metadata_value(bytes)
        if !(schema isa AbstractSchema{type})
            throw(ArgumentError(
                "Stored schema metadata does not describe the requested type $type.",
            ))
        end
    else
        schema = infer_schema(type; options...)
    end
    return read_schema(group, schema)

end

function read_schema(group::HDF5.Group, schema::AbstractSchema)

    metadata_group = group["metadata"]
    validate_format(metadata_group)
    validate_type_name(metadata_group, logical_type(schema))
    validate_schema_node(metadata_group["schema"], schema)
    return schema

end


function validate_format(metadata_group::HDF5.Group)

    stored_name = read_string(metadata_group, "format_name")
    if stored_name != format_name
        throw(ArgumentError(
            "Expected the format $format_name, but found $stored_name.",
        ))
    end

    stored_version = Int64(read(metadata_group["format_version"]))
    if stored_version != format_version
        throw(ArgumentError(
            "HDF5Vectors does not support schema format version $stored_version; " *
            "this implementation reads version $format_version.",
        ))
    end
    return nothing

end

function validate_type_name(group::HDF5.Group, type::Type)
    stored_name = read_string(group, "logical_type")
    expected_name = string(type)
    if stored_name != expected_name
        throw(ArgumentError(
            "The stored schema describes $stored_name, but $expected_name was requested.",
        ))
    end
    return nothing
end

read_string(group::HDF5.Group, name::AbstractString) = String(read(group[name]))

########################
# Schema Node Protocol #
########################

# Every schema implementation writes and validates its own ordinary HDF5 description.
# These methods are the complete format-side interface for a new schema. Built-in schema
# implementations below intentionally use the same dispatch points available to packages.

"""Writes the metadata fields shared by every schema node."""
function write_common_schema(group::HDF5.Group, kind, schema)
    group["kind"] = kind
    group["schema"] = schema_identifier(schema)
    group["logical_type"] = string(logical_type(schema))
    return nothing
end

"""Writes the scalar encoded type used by a scalar or dense schema."""
function write_encoded_type(group::HDF5.Group, schema)
    group["encoded_type"] = string(encoded_type(schema))
    return nothing
end

"""Validates the metadata fields shared by every schema node."""
function validate_common_schema(group::HDF5.Group, kind, schema)

    validate_type_name(group, logical_type(schema))
    stored_kind = read_string(group, "kind")
    if stored_kind != kind
        throw(ArgumentError(
            "The stored schema kind is $stored_kind, but $kind was selected.",
        ))
    end

    stored_schema = read_string(group, "schema")
    expected_schema = schema_identifier(schema)
    if stored_schema != expected_schema
        throw(ArgumentError(
            "The stored schema implementation is $stored_schema, but " *
            "$expected_schema was selected.",
        ))
    end
    return nothing

end

"""Writes a schema node's human-readable codec identifier."""
function write_codec(group::HDF5.Group, codec)
    group["codec"] = codec_identifier(codec)
    return nothing
end

"""Validates a schema node's stored codec identifier."""
function validate_codec(group::HDF5.Group, codec)
    stored_codec = read_string(group, "codec")
    expected_codec = codec_identifier(codec)
    if stored_codec != expected_codec
        throw(ArgumentError(
            "The stored codec is $stored_codec, but $expected_codec was selected.",
        ))
    end
    return nothing
end

"""Validates a scalar or dense schema node's stored encoded type."""
function validate_encoded_type(group::HDF5.Group, schema)
    stored_name = read_string(group, "encoded_type")
    expected_name = string(encoded_type(schema))
    if stored_name != expected_name
        throw(ArgumentError(
            "The stored encoded type is $stored_name, but the codec uses $expected_name.",
        ))
    end
    return nothing
end
