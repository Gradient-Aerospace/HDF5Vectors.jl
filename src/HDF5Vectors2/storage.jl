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

function open_store(group::HDF5.Group, ::ConstantSchema)
    validate_store_children(group, ())
    return ConstantStore(group)
end

###########################
# Scalar Store Operations #
###########################

physical_length(store::ScalarStore) = length(store.dataset)
physical_length(::ConstantStore) = nothing

function validate_write_start(store::ScalarStore, start::Int)
    current_length = physical_length(store)
    if start < 1 || start > current_length + 1
        throw(BoundsError(1:(current_length + 1), start))
    end
    return current_length
end

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
