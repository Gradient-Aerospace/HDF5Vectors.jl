########################
# Dense Representation #
########################

# Dense storage adds one vector dimension to a fixed-size logical value. Its element codec
# remains an ordinary scalar codec, while reconstruction dispatches on the logical container
# type so additional fixed-size containers can reuse this schema without modifying it.

################
# Dense Schema #
################

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
encoded_value_type(::DenseSchema{T, E, H, N}) where {T, E, H, N} = Array{H, N}
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
    return reconstruct_dense_value(T, decoded)

end

function decode_value(
    schema::DenseSchema{T, E, E, N, IdentityCodec{E}},
    encoded::AbstractArray{E, N},
) where {T, E, N}

    # Identity encoding allows the HDF5 result to become the logical value directly. A
    # dynamic Array read is already the required type, while tuples and static arrays can
    # copy directly into their inline representations without an intermediate Array.
    validate_dense_encoding(schema, encoded)
    return reconstruct_dense_value(T, encoded)

end

# Dense reconstruction is a property of the logical container type. Keeping that choice
# behind dispatch permits another fixed-size container to reuse DenseSchema without adding
# a branch to its implementation.
function reconstruct_dense_value(::Type{T}, values) where {T <: Tuple}
    value = Tuple(values)
    if !(value isa T)
        throw(ArgumentError("Decoded dense tuple does not have the declared type $T."))
    end
    return value
end

function reconstruct_dense_value(
    ::Type{T},
    values,
) where {T <: StaticArrays.StaticArray}
    return T(values)
end

function reconstruct_dense_value(::Type{T}, values::T) where {T <: Array}
    return values
end

function reconstruct_dense_value(::Type{T}, values::AbstractArray) where {T <: Array}
    return Array(values)::T
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

function validate_dense_batch_encoding(
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
    return size(stacked, N + 1)

end

function decode_dense_batch_by_frame(
    schema::DenseSchema{T},
    stacked::Array,
    count::Int,
) where {T}

    # Each view borrows the HDF5 read buffer only during reconstruction. Dynamic Arrays
    # receive their own copy, while tuples and static arrays copy into inline storage.
    final_dimension = ndims(stacked)
    values = Vector{T}(undef, count)
    for index in eachindex(values)
        values[index] = decode_value(
            schema,
            selectdim(stacked, final_dimension, index),
        )
    end
    return values

end

function decode_batch(
    schema::DenseSchema{T, E, H, N},
    stacked::Array{H, M},
) where {T, E, H, N, M}
    count = validate_dense_batch_encoding(schema, stacked)
    return decode_dense_batch_by_frame(schema, stacked, count)
end

function decode_batch(
    schema::DenseSchema{T, E, E, N, IdentityCodec{E}},
    stacked::Array{E, M},
) where {T <: StaticArrays.SArray, E, N, M}

    count = validate_dense_batch_encoding(schema, stacked)
    # Immutable bits-backed static arrays store their elements inline in the same
    # column-major order used by the dense HDF5 frame. Reinterpreting the complete buffer
    # and then collecting it performs one contiguous copy into `Vector{T}`.
    if isbitstype(T) && sizeof(T) == prod(schema.dims) * sizeof(E)
        return collect(reinterpret(T, vec(stacked)))
    end
    return decode_dense_batch_by_frame(schema, stacked, count)

end

####################
# Schema Inference #
####################

function infer_builtin_schema(
    type::Type{NTuple{N, E}},
    context::SchemaContext,
) where {N, E}

    dims = isnothing(context.dims) ? (N,) : validate_dims(context.dims, 1)
    if dims != (N,)
        throw(DimensionMismatch("The dimensions $dims do not match the tuple length $N."))
    elseif iszero(N)
        return ConstantSchema(ConstantCodec{type}(()))
    end

    child_context = SchemaContext(context.policy, nothing)
    element_schema = infer_child_schema(E, child_context)
    if element_schema isa ScalarSchema
        return DenseSchema(type, dims, element_schema.codec)
    end
    return record_schema(type, context)

end

function infer_builtin_schema(
    type::Type{<:StaticArrays.StaticArray},
    context::SchemaContext,
)

    expected_dims = Tuple(StaticArrays.Size(type))
    dims = isnothing(context.dims) ? expected_dims : validate_dims(
        context.dims,
        length(expected_dims),
    )
    if dims != expected_dims
        throw(DimensionMismatch(
            "The dimensions $dims do not match the static dimensions $expected_dims.",
        ))
    elseif iszero(prod(expected_dims))
        value = type()
        return ConstantSchema(ConstantCodec{type}(value))
    end

    child_context = SchemaContext(context.policy, nothing)
    element_schema = infer_child_schema(eltype(type), child_context)
    if element_schema isa ScalarSchema
        return DenseSchema(type, dims, element_schema.codec)
    end
    return record_schema(type, SchemaContext(context.policy, nothing))

end

function infer_builtin_schema(
    type::Type{<:Array{E, N}},
    context::SchemaContext,
) where {E, N}

    dims = validate_dims(context.dims, N)
    if isnothing(dims)
        if context.policy.serialize_arrays
            return serialization_schema(type)
        end
        return unsupported_schema(type, "its dimensions were not declared.")
    end

    child_context = SchemaContext(context.policy, nothing)
    element_schema = infer_child_schema(E, child_context)
    if element_schema isa ScalarSchema
        return DenseSchema(type, dims, element_schema.codec)
    elseif context.policy.serialize_arrays
        return serialization_schema(type)
    end
    return unsupported_schema(type, "its element type does not have a scalar encoding.")

end


###################
# Stored Metadata #
###################

function write_schema_node(group::HDF5.Group, schema::DenseSchema)
    write_common_schema(group, "dense", schema)
    write_encoded_type(group, schema)
    write_codec(group, schema.element_codec)
    group["dimensions"] = Int64[schema.dims...,]
    return nothing
end

function validate_schema_node(group::HDF5.Group, schema::DenseSchema)

    validate_common_schema(group, "dense", schema)
    validate_encoded_type(group, schema)
    validate_codec(group, schema.element_codec)
    stored_dims = Tuple(Int(dimension) for dimension in read(group["dimensions"]))
    if stored_dims != schema.dims
        throw(DimensionMismatch(
            "Stored dimensions $stored_dims do not match selected dimensions " *
            "$(schema.dims).",
        ))
    end
    return schema

end


##################
# Physical Store #
##################

struct DenseStore{H, N} <: AbstractStore
    dataset::HDF5.Dataset
    dims::NTuple{N, Int}
end
function create_store(
    group::HDF5.Group,
    schema::DenseSchema{T, E, H, N};
    chunk_length,
) where {T, E, H, N}

    chunk_length = validate_chunk_length(chunk_length)
    initial_dims = (schema.dims..., 0)
    maximum_dims = (schema.dims..., -1)
    dataspace = HDF5.dataspace(initial_dims, maximum_dims)
    dataset = HDF5.create_dataset(
        group,
        "values",
        H,
        dataspace;
        chunk = (schema.dims..., chunk_length),
    )
    return DenseStore{H, N}(dataset, schema.dims)

end

function open_store(
    group::HDF5.Group,
    schema::DenseSchema{T, E, H, N},
) where {T, E, H, N}

    validate_store_children(group, ("values",))
    dataset = group["values"]
    if ndims(dataset) != N + 1
        throw(DimensionMismatch(
            "Dense storage for $T must have $(N + 1) dimensions, but its size is " *
            "$(size(dataset)).",
        ))
    elseif size(dataset)[1:N] != schema.dims
        throw(DimensionMismatch(
            "Dense storage for $T must have leading dimensions $(schema.dims), but " *
            "its size is $(size(dataset)).",
        ))
    elseif !dataset_matches_encoded_type(dataset, H)
        throw(ArgumentError(
            "Dense storage does not use the HDF5 datatype required for $H.",
        ))
    end
    return DenseStore{H, N}(dataset, schema.dims)

end

physical_length(store::DenseStore{H, N}) where {H, N} = size(store.dataset, N + 1)

##########################
# Dense Store Operations #
##########################

# A dense dataset stacks fixed-size encoded frames along its final dimension. Keeping the
# frame dimensions on the store makes the checks below independent of logical Julia types
# and codecs.
dense_colons(::DenseStore{H, N}) where {H, N} = ntuple(_ -> Colon(), N)
dense_extent(store::DenseStore, count::Int) = (store.dims..., count)

function validate_dense_frame(store::DenseStore, frame::Array)
    if size(frame) != store.dims
        throw(DimensionMismatch(
            "Expected an encoded frame with dimensions $(store.dims), but got " *
            "$(size(frame)).",
        ))
    end
    return frame
end

function stack_dense_frames(
    store::DenseStore{H, N},
    frames::AbstractVector{<:Array{H, N}},
) where {H, N}

    # Every frame is checked before allocating or writing the stacked HDF5 representation.
    # This matters for dynamic Arrays, whose dimensions are not guaranteed by their type.
    for frame in frames
        validate_dense_frame(store, frame)
    end

    stacked = Array{H, N + 1}(undef, (store.dims..., length(frames)))
    for (index, frame) in enumerate(frames)
        copyto!(selectdim(stacked, N + 1, index), frame)
    end
    return stacked

end

function initialize_encoded!(
    store::DenseStore{H, N},
    frames::AbstractVector{<:Array{H, N}},
) where {H, N}

    # Stacking validates the complete batch before the dataset extent changes.
    stacked = stack_dense_frames(store, frames)
    return initialize_encoded!(store, stacked)

end

function initialize_encoded!(
    store::DenseStore{H, N},
    stacked::Array{H, M},
) where {H, N, M}

    # A directly encoded dense batch already has the physical HDF5 layout. Its leading
    # dimensions are still checked before changing the dataset because the final dimension
    # alone determines how many logical values it contains.
    frame_count = size(stacked, N + 1)
    expected_extent = dense_extent(store, frame_count)
    if size(stacked) != expected_extent
        throw(DimensionMismatch(
            "Expected an encoded batch with dimensions $expected_extent, but got " *
            "$(size(stacked)).",
        ))
    end

    if iszero(frame_count)
        return store
    end

    HDF5.set_extent_dims(store.dataset, dense_extent(store, frame_count))
    selection = (dense_colons(store)..., Colon())
    store.dataset[selection...] = stacked
    return store

end

function read_encoded_batch(
    store::DenseStore{H, N},
    indices::UnitRange{Int},
) where {H, N}

    if isempty(indices)
        return Array{H, N + 1}(undef, dense_extent(store, 0))
    elseif first(indices) < 1 || last(indices) > physical_length(store)
        throw(BoundsError(store, indices))
    end

    # The dataset already stores a dense batch in the representation consumed by the
    # schema layer. Preserving it avoids allocating an independent Array for every frame.
    selection = (dense_colons(store)..., indices)
    return read(store.dataset, H, selection...)

end

function read_encoded(store::DenseStore{H, N}, index::Int) where {H, N}

    if index < 1 || index > physical_length(store)
        throw(BoundsError(store, index))
    end

    # Reading a one-element range preserves the final dimension. Dropping it explicitly
    # then returns an N-dimensional Array even for the unusual zero-dimensional case.
    selection = (dense_colons(store)..., index:index)
    stacked = read(store.dataset, H, selection...)
    return dropdims(stacked; dims = N + 1)

end

function read_encoded(
    store::DenseStore{H, N},
    indices::UnitRange{Int},
) where {H, N}

    if isempty(indices)
        return Array{H, N}[]
    elseif first(indices) < 1 || last(indices) > physical_length(store)
        throw(BoundsError(store, indices))
    end

    # The storage layer returns independent frames, matching the scalar store's vector of
    # encoded values and preventing a decoded mutable Array from aliasing a larger buffer.
    selection = (dense_colons(store)..., indices)
    stacked = read(store.dataset, H, selection...)
    return [Array(selectdim(stacked, N + 1, index)) for index in 1:length(indices)]

end

stored_value_type(::DenseStore{H, N}) where {H, N} = Array{H, N}
function validate_encoded(
    store::DenseStore{H, N},
    frame::Array{H, N},
) where {H, N}
    validate_dense_frame(store, frame)
    return nothing
end

function validate_encoded_batch(
    store::DenseStore{H, N},
    stacked::Array{H, M},
    expected_count::Int,
) where {H, N, M}
    expected_extent = dense_extent(store, expected_count)
    if size(stacked) != expected_extent
        throw(DimensionMismatch(
            "An encoded dense column has dimensions $(size(stacked)) instead of " *
            "$expected_extent.",
        ))
    end
    return nothing
end
function append_encoded!(
    store::DenseStore{H, N},
    index::Int,
    frame::Array{H, N},
) where {H, N}
    HDF5.set_extent_dims(store.dataset, dense_extent(store, index))
    selection = (dense_colons(store)..., index:index)
    store.dataset[selection...] = reshape(frame, (store.dims..., 1))
    return store
end
