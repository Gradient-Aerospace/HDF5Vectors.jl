#######################
# Blob Representation #
#######################

# Blob storage concatenates independently encoded byte vectors and stores cumulative end
# positions. Julia Serialization is one codec for this physical representation; another
# byte-producing codec can reuse BlobSchema and BlobStore without changing either one.

##############
# Blob Codec #
##############

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

###############
# Blob Schema #
###############

struct BlobSchema{T, C <: AbstractCodec{T, Vector{UInt8}}} <: AbstractSchema{T}
    codec::C
end
encoded_value_type(::BlobSchema) = Vector{UInt8}
encode_value(schema::BlobSchema{T}, value::T) where {T} = encode_value(schema.codec, value)
decode_value(schema::BlobSchema, bytes::Vector{UInt8}) = decode_value(schema.codec, bytes)

####################
# Schema Inference #
####################

"""
    serialization_schema(type::Type)

Builds an explicit Julia-serialization schema for `type`.
"""
function serialization_schema(::Type{T}) where {T}
    return BlobSchema(SerializationCodec{T}())
end

###################
# Stored Metadata #
###################

function write_schema_node(group::HDF5.Group, schema::BlobSchema)
    write_common_schema(group, "blob", schema)
    write_codec(group, schema.codec)
    return nothing
end

function validate_schema_node(group::HDF5.Group, schema::BlobSchema)
    validate_common_schema(group, "blob", schema)
    validate_codec(group, schema.codec)
    return schema
end


##################
# Physical Store #
##################

struct BlobStore <: AbstractStore
    bytes::HDF5.Dataset
    stops::HDF5.Dataset
end
function create_store(
    group::HDF5.Group,
    ::BlobSchema;
    chunk_length,
)

    dataspace = HDF5.dataspace((0,), (-1,))
    bytes = HDF5.create_dataset(
        group,
        "bytes",
        UInt8,
        dataspace;
        chunk = (chunk_length,),
    )
    stops = HDF5.create_dataset(
        group,
        "stops",
        Int64,
        dataspace;
        chunk = (chunk_length,),
    )
    return BlobStore(bytes, stops)

end

function open_store(group::HDF5.Group, ::BlobSchema)

    validate_store_children(group, ("bytes", "stops"))
    bytes = group["bytes"]
    stops = group["stops"]
    if ndims(bytes) != 1 || ndims(stops) != 1
        throw(DimensionMismatch(
            "Blob byte and stop storage must both be one-dimensional.",
        ))
    elseif !dataset_matches_encoded_type(bytes, UInt8)
        throw(ArgumentError("Blob byte storage must use the HDF5 datatype UInt8."))
    elseif !dataset_matches_encoded_type(stops, Int64)
        throw(ArgumentError("Blob stop storage must use the HDF5 datatype Int64."))
    end

    # The last stop is the total number of concatenated bytes. Checking this one boundary
    # detects interrupted writes and truncated datasets without scanning every element.
    stop_count = length(stops)
    final_stop = iszero(stop_count) ? Int64(0) : read(stops, Int64, stop_count)
    if final_stop != length(bytes)
        throw(DimensionMismatch(
            "The final blob stop is $final_stop, but byte storage has length " *
            "$(length(bytes)).",
        ))
    end
    return BlobStore(bytes, stops)

end

physical_length(store::BlobStore) = length(store.stops)

#########################
# Blob Store Operations #
#########################

# Variable-length encoded values are concatenated in `bytes`. Each entry in `stops` is the
# cumulative byte count after the corresponding value, so repeated stops represent empty
# values without any special cases in the stored format.
function blob_end_offset(store::BlobStore, count::Int)
    if iszero(count)
        return Int64(0)
    end
    return read(store.stops, Int64, count)
end

function prepare_blob_batch(values::AbstractVector{<:Vector{UInt8}})

    total_bytes = 0
    for value in values
        total_bytes = Base.Checked.checked_add(total_bytes, length(value))
    end

    concatenated = Vector{UInt8}(undef, total_bytes)
    stops = Vector{Int64}(undef, length(values))
    next_byte = 1
    cumulative_stop = Int64(0)
    for (index, value) in enumerate(values)
        if !isempty(value)
            copyto!(concatenated, next_byte, value, 1, length(value))
            next_byte += length(value)
        end
        cumulative_stop = Base.Checked.checked_add(cumulative_stop, length(value))
        stops[index] = cumulative_stop
    end
    return concatenated, stops

end

function initialize_encoded!(
    store::BlobStore,
    values::AbstractVector{<:Vector{UInt8}},
)

    if isempty(values)
        return store
    end

    # Concatenation and cumulative-stop arithmetic finish before either HDF5 dataset is
    # extended. Once writing begins, an unrecoverable HDF5 failure can still leave the two
    # datasets inconsistent; `open_store` detects that state through their final boundary.
    concatenated, stops = prepare_blob_batch(values)
    final_stop = last(stops)
    if !isempty(concatenated)
        HDF5.set_extent_dims(store.bytes, (final_stop,))
        store.bytes[1:final_stop] = concatenated
    end

    HDF5.set_extent_dims(store.stops, (length(values),))
    store.stops[:] = stops
    return store

end

function read_encoded(store::BlobStore, index::Int)

    # Opening validates the shared final boundary. Reads can then use the requested stops
    # directly without another validation pass.
    initial_stop = blob_end_offset(store, index - 1)
    final_stop = blob_end_offset(store, index)
    if final_stop == initial_stop
        return UInt8[]
    end
    return read(store.bytes, UInt8, (initial_stop + 1):final_stop)

end

function read_encoded(store::BlobStore, indices::UnitRange{Int})

    if isempty(indices)
        return Vector{UInt8}[]
    end

    # Only the stops needed to split this range are loaded. Avoiding a second validation
    # pass keeps the work proportional to the data the caller actually requested.
    initial_stop = blob_end_offset(store, first(indices) - 1)
    stops = read(store.stops, Int64, indices)
    concatenated = if last(stops) == initial_stop
        UInt8[]
    else
        read(store.bytes, UInt8, (initial_stop + 1):last(stops))
    end

    values = Vector{Vector{UInt8}}(undef, length(indices))
    previous_stop = initial_stop
    next_byte = 1
    for index in eachindex(values)
        byte_count = Int(stops[index] - previous_stop)
        values[index] = concatenated[next_byte:(next_byte + byte_count - 1)]
        next_byte += byte_count
        previous_stop = stops[index]
    end
    return values

end

function validate_encoded_batch(
    ::BlobStore,
    values::AbstractVector{<:Vector{UInt8}},
    expected_count::Int,
)
    return validate_encoded_column_count(values, expected_count)
end
function append_encoded!(store::BlobStore, index::Int, value::Vector{UInt8})

    # The byte dataset's current extent is also the next cumulative stop. Creation and
    # loading have already checked that it agrees with the preceding stored stop, so the
    # append path does not need to read that stop again.
    initial_stop = Int64(length(store.bytes))
    final_stop = Base.Checked.checked_add(initial_stop, Int64(length(value)))
    if !isempty(value)
        HDF5.set_extent_dims(store.bytes, (final_stop,))
        store.bytes[(initial_stop + 1):final_stop] = value
    end

    HDF5.set_extent_dims(store.stops, (index,))
    store.stops[index] = final_stop
    return store

end
