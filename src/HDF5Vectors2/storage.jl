####################
# Physical Storage #
####################

# A store contains open HDF5 objects but no logical type conversion. Its methods read and
# write values that have already passed through the corresponding schema. This boundary
# keeps validation and codec failures out of the physical mutation layer.
abstract type AbstractStore end

struct ScalarStore{H} <: AbstractStore
    dataset::HDF5.Dataset
end

struct DenseStore{H, N} <: AbstractStore
    dataset::HDF5.Dataset
    dims::NTuple{N, Int}
end

struct RecordStore{Children <: Tuple} <: AbstractStore
    children::Children
end

struct ConstantStore <: AbstractStore
    group::HDF5.Group
end

function validate_chunk_length(chunk_length)
    if !(chunk_length isa Integer) || chunk_length isa Bool || chunk_length <= 0
        throw(ArgumentError(
            "The HDF5 chunk length must be a positive integer; got $chunk_length.",
        ))
    end
    return Int(chunk_length)
end

function validate_store_children(group::HDF5.Group, expected_names)
    stored_names = Set(String(name) for name in keys(group))
    expected_names = Set(String(name) for name in expected_names)
    if stored_names != expected_names
        throw(ArgumentError(
            "Stored data children $stored_names do not match $expected_names.",
        ))
    end
    return nothing
end

# HDF5.jl reports variable-length strings as Cstring and compound structs as structural
# NamedTuples when `eltype` is queried. Comparing HDF5 datatype objects validates their
# actual physical representations without mistaking either case for a schema mismatch.
function dataset_matches_encoded_type(dataset::HDF5.Dataset, encoded_type::Type)
    stored_datatype = HDF5.datatype(dataset)
    expected_datatype = HDF5.datatype(encoded_type)
    try
        return stored_datatype == expected_datatype
    finally
        close(stored_datatype)
        close(expected_datatype)
    end
end

##################
# Creating Stores #
##################

function create_store(
    group::HDF5.Group,
    schema::ScalarSchema{T, H};
    chunk_length,
) where {T, H}

    chunk_length = validate_chunk_length(chunk_length)
    dataspace = HDF5.dataspace((0,), (-1,))
    dataset = HDF5.create_dataset(
        group,
        "values",
        H,
        dataspace;
        chunk = (chunk_length,),
    )
    return ScalarStore{H}(dataset)

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

function create_store(
    group::HDF5.Group,
    schema::RecordSchema{T, N};
    chunk_length,
) where {T, N}

    chunk_length = validate_chunk_length(chunk_length)
    children = ntuple(N) do index
        child_group = HDF5.create_group(group, schema.names[index])
        return create_store(
            child_group,
            schema.children[index];
            chunk_length,
        )
    end
    return RecordStore(children)

end

function create_store(
    group::HDF5.Group,
    ::ConstantSchema;
    chunk_length,
)
    validate_chunk_length(chunk_length)
    return ConstantStore(group)
end

##################
# Opening Stores #
##################

function open_store(group::HDF5.Group, ::ScalarSchema{T, H}) where {T, H}

    validate_store_children(group, ("values",))
    dataset = group["values"]
    if ndims(dataset) != 1
        throw(DimensionMismatch(
            "Scalar storage must be one-dimensional, but its size is $(size(dataset)).",
        ))
    elseif !dataset_matches_encoded_type(dataset, H)
        throw(ArgumentError(
            "Scalar storage does not use the HDF5 datatype required for $H.",
        ))
    end
    return ScalarStore{H}(dataset)

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

function open_store(
    group::HDF5.Group,
    schema::RecordSchema{T, N},
) where {T, N}

    validate_store_children(group, schema.names)
    children = ntuple(N) do index
        return open_store(group[schema.names[index]], schema.children[index])
    end

    # All nonconstant columns must describe the same number of records. Running this check
    # while opening catches incomplete or manually altered layouts before they are read.
    store = RecordStore(children)
    physical_length(store)
    return store

end

function open_store(group::HDF5.Group, ::ConstantSchema)
    validate_store_children(group, ())
    return ConstantStore(group)
end

###########################
# Shared Store Operations #
###########################

physical_length(store::ScalarStore) = length(store.dataset)
physical_length(store::DenseStore{H, N}) where {H, N} = size(store.dataset, N + 1)
physical_length(::ConstantStore) = nothing

function physical_length(store::RecordStore)

    record_length = nothing
    for (index, child) in enumerate(store.children)
        child_length = physical_length(child)
        if isnothing(child_length)
            continue
        elseif isnothing(record_length)
            record_length = child_length
        elseif child_length != record_length
            throw(DimensionMismatch(
                "Record child $index has length $child_length, while the other " *
                "nonconstant children have length $record_length.",
            ))
        end
    end
    return record_length

end

function validate_write_start(store::Union{ScalarStore, DenseStore}, start::Int)
    current_length = physical_length(store)
    if start < 1 || start > current_length + 1
        throw(BoundsError(1:(current_length + 1), start))
    end
    return current_length
end

###########################
# Scalar Store Operations #
###########################

function write_encoded!(store::ScalarStore{H}, index::Int, value::H) where {H}
    current_length = validate_write_start(store, index)
    if index > current_length
        HDF5.set_extent_dims(store.dataset, (index,))
    end
    store.dataset[index] = value
    return store
end

function write_encoded_batch!(
    store::ScalarStore{H},
    start::Int,
    values::AbstractVector{H},
) where {H}

    current_length = validate_write_start(store, start)
    if isempty(values)
        return store
    end

    final_index = start + length(values) - 1
    if final_index > current_length
        HDF5.set_extent_dims(store.dataset, (final_index,))
    end
    store.dataset[start:final_index] = values
    return store

end

function read_encoded(store::ScalarStore{H}, index::Int) where {H}
    if index < 1 || index > physical_length(store)
        throw(BoundsError(store, index))
    end
    return read(store.dataset, H, index)
end

function read_encoded(store::ScalarStore{H}, indices::UnitRange{Int}) where {H}
    if isempty(indices)
        return H[]
    elseif first(indices) < 1 || last(indices) > physical_length(store)
        throw(BoundsError(store, indices))
    end
    return read(store.dataset, H, indices)
end

function truncate_store!(store::ScalarStore, count::Int)
    current_length = physical_length(store)
    if count < 0 || count > current_length
        throw(BoundsError(0:current_length, count))
    end
    HDF5.set_extent_dims(store.dataset, (count,))
    return store
end

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

function write_encoded!(
    store::DenseStore{H, N},
    index::Int,
    frame::Array{H, N},
) where {H, N}

    # Validation happens before extending the dataset. A rejected frame therefore cannot
    # leave an unwritten physical slot at the end of the store.
    validate_dense_frame(store, frame)
    current_length = validate_write_start(store, index)
    if index > current_length
        HDF5.set_extent_dims(store.dataset, dense_extent(store, index))
    end

    selection = (dense_colons(store)..., index:index)
    store.dataset[selection...] = reshape(frame, (store.dims..., 1))
    return store

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

function write_encoded_batch!(
    store::DenseStore{H, N},
    start::Int,
    frames::AbstractVector{<:Array{H, N}},
) where {H, N}

    current_length = validate_write_start(store, start)
    if isempty(frames)
        return store
    end

    # Stacking validates the complete batch before the dataset extent changes.
    stacked = stack_dense_frames(store, frames)
    final_index = start + length(frames) - 1
    if final_index > current_length
        HDF5.set_extent_dims(store.dataset, dense_extent(store, final_index))
    end

    selection = (dense_colons(store)..., start:final_index)
    store.dataset[selection...] = stacked
    return store

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

function truncate_store!(store::DenseStore, count::Int)
    current_length = physical_length(store)
    if count < 0 || count > current_length
        throw(BoundsError(0:current_length, count))
    end
    HDF5.set_extent_dims(store.dataset, dense_extent(store, count))
    return store
end

###########################
# Record Store Operations #
###########################

# The encoded value type is a property of physical storage. It lets record batches build
# concretely typed child columns without consulting a logical schema or running a codec.
stored_value_type(::ScalarStore{H}) where {H} = H
stored_value_type(::DenseStore{H, N}) where {H, N} = Array{H, N}
stored_value_type(::ConstantStore) = Nothing

function stored_value_type(store::RecordStore)
    child_types = map(stored_value_type, store.children)
    return Core.apply_type(Tuple, child_types...)
end

function validate_record_value(store::RecordStore, value::Tuple)

    child_count = length(store.children)
    if length(value) != child_count
        throw(ArgumentError(
            "Encoded record data has $(length(value)) fields instead of $child_count.",
        ))
    end

    for index in eachindex(store.children)
        validate_encoded(store.children[index], value[index])
    end
    return value

end

validate_encoded(::ScalarStore{H}, ::H) where {H} = nothing
function validate_encoded(
    store::DenseStore{H, N},
    frame::Array{H, N},
) where {H, N}
    validate_dense_frame(store, frame)
    return nothing
end

validate_encoded(::ConstantStore, ::Nothing) = nothing
function validate_encoded(store::RecordStore, value::Tuple)
    validate_record_value(store, value)
    return nothing
end

function validate_write_start(store::RecordStore, start::Int)
    current_length = physical_length(store)
    if isnothing(current_length)
        if start < 1
            throw(BoundsError(1:typemax(Int), start))
        end
    elseif start < 1 || start > current_length + 1
        throw(BoundsError(1:(current_length + 1), start))
    end
    return current_length
end

function write_encoded!(store::RecordStore, index::Int, value::Tuple)

    # Recursive validation finishes before the first child store changes. Codec failures
    # have already occurred above this layer, while ordinary shape or type errors are
    # caught here before any record column is extended.
    validate_record_value(store, value)
    validate_write_start(store, index)
    for child_index in eachindex(store.children)
        write_encoded!(
            store.children[child_index],
            index,
            value[child_index],
        )
    end
    return store

end

function write_encoded_batch!(
    store::RecordStore,
    start::Int,
    values::AbstractVector{<:Tuple},
)

    validate_write_start(store, start)

    # Preflighting the complete batch keeps a bad value in a later record from leaving
    # earlier columns longer than later columns. An HDF5 failure between child writes can
    # still produce unequal physical lengths, which reopening detects explicitly.
    for value in values
        validate_record_value(store, value)
    end

    for child_index in eachindex(store.children)
        child = store.children[child_index]
        child_values = Vector{stored_value_type(child)}(undef, length(values))
        for value_index in eachindex(values)
            child_values[value_index] = values[value_index][child_index]
        end
        write_encoded_batch!(child, start, child_values)
    end
    return store

end

function read_encoded(store::RecordStore, index::Int)

    record_length = physical_length(store)
    if isnothing(record_length)
        if index < 1
            throw(BoundsError(1:typemax(Int), index))
        end
    elseif index < 1 || index > record_length
        throw(BoundsError(store, index))
    end
    return map(child -> read_encoded(child, index), store.children)

end

function read_encoded(store::RecordStore, indices::UnitRange{Int})

    record_length = physical_length(store)
    if isnothing(record_length)
        if !isempty(indices) && first(indices) < 1
            throw(BoundsError(1:typemax(Int), indices))
        end
    elseif !isempty(indices) && (first(indices) < 1 || last(indices) > record_length)
        throw(BoundsError(store, indices))
    end

    child_columns = map(child -> read_encoded(child, indices), store.children)
    values = Vector{stored_value_type(store)}(undef, length(indices))
    for value_index in eachindex(values)
        values[value_index] = map(column -> column[value_index], child_columns)
    end
    return values

end

function truncate_store!(store::RecordStore, count::Int)

    current_length = physical_length(store)
    if isnothing(current_length)
        if count < 0
            throw(BoundsError(0:typemax(Int), count))
        end
    elseif count < 0 || count > current_length
        throw(BoundsError(0:current_length, count))
    end

    for child in store.children
        truncate_store!(child, count)
    end
    return store

end

#############################
# Constant Store Operations #
#############################

# Constant values have no physical payload. The vector-level logical length determines how
# many values exist, so these operations require the caller to perform ordinary bounds
# checks against that length.
function write_encoded!(store::ConstantStore, index::Int, ::Nothing)
    if index < 1
        throw(BoundsError(1:typemax(Int), index))
    end
    return store
end

function write_encoded_batch!(
    store::ConstantStore,
    start::Int,
    values::AbstractVector{Nothing},
)
    if start < 1
        throw(BoundsError(1:typemax(Int), start))
    end
    return store
end

function read_encoded(::ConstantStore, index::Int)
    if index < 1
        throw(BoundsError(1:typemax(Int), index))
    end
    return nothing
end

function read_encoded(::ConstantStore, indices::UnitRange{Int})
    if !isempty(indices) && first(indices) < 1
        throw(BoundsError(1:typemax(Int), indices))
    end
    return fill(nothing, length(indices))
end

function truncate_store!(store::ConstantStore, count::Int)
    if count < 0
        throw(BoundsError(0:typemax(Int), count))
    end
    return store
end
