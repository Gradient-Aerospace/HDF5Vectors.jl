#########################
# Stored Schema Format  #
#########################

const format_name = "HDF5Vectors2"
const format_version = Int64(1)

# Every HDF5Vectors2 group has one metadata child. Its format name and version identify the
# reader needed for the group, while its recursive schema describes the representation that
# was actually selected. Human-readable type names make the layout inspectable outside
# Julia. Serialized top-level type metadata is only a convenience for Julia callers that do
# not supply the logical type explicitly.
#
# Within schema metadata, record child schemas use numeric names so their order remains
# explicit. The corresponding field names are stored as ordinary string data on the schema
# node and are used as the meaningful paths of physical record fields under `data`.

function serialize_metadata_value(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return take!(io)
end

function deserialize_metadata_value(bytes::Vector{UInt8})
    return Serialization.deserialize(IOBuffer(bytes))
end

"""
    write_schema(group::HDF5.Group, schema::AbstractSchema)

Writes the versioned logical type and complete storage schema into a new `metadata` child of
`group`. The schema records the selected representation directly, so loading does not need
to repeat schema inference or recover the original creation policy.
"""
function write_schema(group::HDF5.Group, schema::AbstractSchema)

    type = logical_type(schema)
    metadata_group = HDF5.create_group(group, "metadata")
    metadata_group["format_name"] = format_name
    metadata_group["format_version"] = format_version
    metadata_group["logical_type"] = string(type)
    metadata_group["serialized_logical_type"] = serialize_metadata_value(type)

    schema_group = HDF5.create_group(metadata_group, "schema")
    write_schema_node(schema_group, schema)
    return schema

end

"""
    read_schema(group::HDF5.Group)
    read_schema(group::HDF5.Group, type::Type)

Reads the exact schema stored by [`write_schema`](@ref). The first form recovers the logical
type from Julia-serialized metadata. The explicit-type form avoids deserializing that type
and verifies the supplied type against the human-readable metadata.
"""
function read_schema(group::HDF5.Group)
    metadata_group = group["metadata"]
    validate_format(metadata_group)
    bytes = Vector{UInt8}(read(metadata_group["serialized_logical_type"]))
    type = deserialize_metadata_value(bytes)
    if !(type isa Type)
        throw(ArgumentError(
            "Stored logical-type metadata produced a value of type $(typeof(type)).",
        ))
    end
    validate_type_name(metadata_group, type)
    return read_schema_node(metadata_group["schema"], type)
end

function read_schema(group::HDF5.Group, type::Type)
    metadata_group = group["metadata"]
    validate_format(metadata_group)
    validate_type_name(metadata_group, type)
    return read_schema_node(metadata_group["schema"], type)
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
            "HDF5Vectors2 does not support schema format version $stored_version; " *
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
# Writing Schema Nodes #
########################

function write_common_schema(group::HDF5.Group, kind, schema)
    group["kind"] = kind
    group["logical_type"] = string(logical_type(schema))
    return nothing
end

function write_encoded_type(group::HDF5.Group, schema)
    group["encoded_type"] = string(encoded_type(schema))
    return nothing
end

codec_name(::IdentityCodec) = "identity"
codec_name(::CharCodec) = "char_int32"
codec_name(::SymbolCodec) = "symbol_string"
codec_name(::EnumCodec) = "enum"
codec_name(::SerializationCodec) = "julia_serialization"
codec_name(::ConstantCodec) = "constant"

record_codec_name(::StructCodec) = "struct"
record_codec_name(::TupleCodec) = "tuple"
record_codec_name(::NamedTupleCodec) = "named_tuple"
record_codec_name(::StaticArrayCodec) = "static_array"

function write_schema_node(group::HDF5.Group, schema::ScalarSchema)
    write_common_schema(group, "scalar", schema)
    write_encoded_type(group, schema)
    group["codec"] = codec_name(schema.codec)
    return nothing
end

function write_schema_node(group::HDF5.Group, schema::DenseSchema)
    write_common_schema(group, "dense", schema)
    write_encoded_type(group, schema)
    group["codec"] = codec_name(schema.element_codec)
    group["dimensions"] = Int64[schema.dims...,]
    return nothing
end

function write_schema_node(group::HDF5.Group, schema::RecordSchema)

    write_common_schema(group, "record", schema)
    group["codec"] = record_codec_name(schema.codec)
    group["field_names"] = collect(schema.names)

    children_group = HDF5.create_group(group, "children")
    for (index, child) in enumerate(schema.children)
        child_group = HDF5.create_group(children_group, string(index))
        write_schema_node(child_group, child)
    end
    return nothing

end

function write_schema_node(group::HDF5.Group, schema::BlobSchema)
    write_common_schema(group, "blob", schema)
    group["codec"] = codec_name(schema.codec)
    return nothing
end

function write_schema_node(group::HDF5.Group, schema::ConstantSchema)
    write_common_schema(group, "constant", schema)
    group["codec"] = codec_name(schema.codec)
    group["serialized_value"] = serialize_metadata_value(schema.codec.value)
    return nothing
end

########################
# Reading Schema Nodes #
########################

function read_schema_node(group::HDF5.Group, type::Type)
    validate_type_name(group, type)
    kind = read_string(group, "kind")
    if kind == "scalar"
        return read_scalar_schema(group, type)
    elseif kind == "dense"
        return read_dense_schema(group, type)
    elseif kind == "record"
        return read_record_schema(group, type)
    elseif kind == "blob"
        return read_blob_schema(group, type)
    elseif kind == "constant"
        return read_constant_schema(group, type)
    end
    throw(ArgumentError("The stored schema kind $kind is not supported."))
end

function read_scalar_schema(group::HDF5.Group, type::Type)
    codec = read_scalar_codec(read_string(group, "codec"), type)
    schema = ScalarSchema(codec)
    validate_encoded_type(group, schema)
    return schema
end

function read_dense_schema(group::HDF5.Group, type::Type)

    dims = Tuple(Int(dimension) for dimension in read(group["dimensions"]))
    validate_stored_dense_dims(type, dims)
    element_type = eltype(type)
    codec = read_scalar_codec(read_string(group, "codec"), element_type)
    schema = DenseSchema(type, dims, codec)
    validate_encoded_type(group, schema)
    return schema

end

function read_record_schema(group::HDF5.Group, type::Type)

    stored_names = Tuple(String(name) for name in read(group["field_names"]))
    expected_names = Tuple(string(name) for name in fieldnames(type))
    if stored_names != expected_names
        throw(ArgumentError(
            "Stored record fields $stored_names do not match $type fields $expected_names.",
        ))
    end

    codec = read_record_codec(read_string(group, "codec"), type)
    field_types = fieldtypes(type)
    children_group = group["children"]
    stored_children = Set(String(name) for name in keys(children_group))
    expected_children = Set(string(index) for index in eachindex(field_types))
    if stored_children != expected_children
        throw(ArgumentError(
            "Stored record children $stored_children do not match $expected_children.",
        ))
    end

    children = ntuple(
        index -> read_schema_node(children_group[string(index)], field_types[index]),
        length(field_types),
    )
    return RecordSchema(type, stored_names, codec, children)

end

function read_blob_schema(group::HDF5.Group, type::Type)
    codec = read_string(group, "codec")
    if codec != "julia_serialization"
        throw(ArgumentError("The stored blob codec $codec is not supported."))
    end
    return BlobSchema(SerializationCodec{type}())
end

function read_constant_schema(group::HDF5.Group, type::Type)

    codec = read_string(group, "codec")
    if codec != "constant"
        throw(ArgumentError("The stored constant codec $codec is not supported."))
    end

    bytes = Vector{UInt8}(read(group["serialized_value"]))
    value = deserialize_metadata_value(bytes)
    if !(value isa type)
        throw(ArgumentError(
            "The stored constant for $type produced a value of type $(typeof(value)).",
        ))
    end
    return ConstantSchema(ConstantCodec{type}(value))

end

function read_scalar_codec(name::String, type::Type)
    if name == "identity"
        return IdentityCodec{type}()
    elseif name == "char_int32" && type === Char
        return CharCodec()
    elseif name == "symbol_string" && type === Symbol
        return SymbolCodec()
    elseif name == "enum" && type <: Enum
        return enum_codec(type)
    end
    throw(ArgumentError("The stored scalar codec $name is not valid for $type."))
end

enum_codec(::Type{T}) where {H, T <: Enum{H}} = EnumCodec{T, H}()

function read_record_codec(name::String, type::Type)
    if name == "named_tuple" && type <: NamedTuple
        return NamedTupleCodec{type}()
    elseif name == "tuple" && type <: Tuple
        return TupleCodec{type}()
    elseif name == "static_array" && type <: StaticArrays.StaticArray
        return StaticArrayCodec{type}()
    elseif name == "struct" && !(type <: Tuple) && !(type <: StaticArrays.StaticArray)
        names = fieldnames(type)
        return StructCodec{type, length(names)}(names)
    end
    throw(ArgumentError("The stored record codec $name is not valid for $type."))
end

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

function validate_stored_dense_dims(type::Type, dims::Tuple)

    if !all(dimension -> dimension > 0, dims)
        throw(ArgumentError("Stored dense dimensions must be positive; got $dims."))
    elseif type <: Tuple
        expected_dims = (fieldcount(type),)
    elseif type <: StaticArrays.StaticArray
        expected_dims = Tuple(StaticArrays.Size(type))
    elseif type <: Array
        expected_rank = ndims(type)
        if length(dims) != expected_rank
            throw(DimensionMismatch(
                "Stored dimensions $dims do not have the $expected_rank dimensions " *
                "of $type.",
            ))
        end
        return nothing
    else
        throw(ArgumentError("The stored dense schema cannot represent $type."))
    end

    if dims != expected_dims
        throw(DimensionMismatch(
            "Stored dimensions $dims do not match the $type dimensions $expected_dims.",
        ))
    end
    return nothing

end
