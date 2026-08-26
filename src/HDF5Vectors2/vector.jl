#######################
# Public Vector Type  #
#######################

"""
    HDF5Vector{T} <: AbstractVector{T}

An HDF5-backed vector with logical element type `T`. Vectors are normally constructed with
[`create_hdf5_vector`](@ref), [`copy_to_hdf5_vector`](@ref), or
[`load_hdf5_vector`](@ref).
"""
mutable struct HDF5Vector{
    T,
    S <: AbstractSchema{T},
    Store <: AbstractStore,
} <: AbstractVector{T}
    schema::S
    store::Store
    count_dataset::HDF5.Dataset
    count::Int
end

Base.length(vector::HDF5Vector) = vector.count
Base.size(vector::HDF5Vector) = (length(vector),)
Base.IndexStyle(::Type{<:HDF5Vector}) = IndexLinear()
Base.similar(::HDF5Vector, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

########################
# Creating and Loading #
########################

function validate_destination(group::HDF5.Group, name::AbstractString)

    name = String(name)
    if isempty(name) || name == "." || occursin('/', name) || occursin('\0', name)
        throw(ArgumentError(
            "The HDF5 vector name $(repr(name)) must be one HDF5 path component.",
        ))
    elseif haskey(group, name)
        throw(ArgumentError("The HDF5 group already contains a child named $name."))
    end
    return name

end

function read_logical_count(metadata_group::HDF5.Group)

    if !haskey(metadata_group, "count")
        throw(ArgumentError("HDF5 vector metadata does not contain a logical count."))
    end

    dataset = metadata_group["count"]
    if ndims(dataset) != 0
        throw(DimensionMismatch("The stored logical count must be a scalar."))
    elseif !dataset_matches_encoded_type(dataset, Int64)
        throw(ArgumentError("The stored logical count must use the HDF5 datatype Int64."))
    end

    count = read(dataset)
    if count < 0 || count > typemax(Int)
        throw(ArgumentError("The stored logical count $count is not a valid length."))
    end
    return Int(count), dataset

end

function validate_logical_count(store::AbstractStore, count::Int)
    stored_length = physical_length(store)
    if !isnothing(stored_length) && stored_length != count
        throw(DimensionMismatch(
            "The HDF5 vector count is $count, but its physical storage has length " *
            "$stored_length.",
        ))
    end
    return nothing
end

function persist_count!(vector::HDF5Vector, count::Int)
    write(vector.count_dataset, Int64(count))
    vector.count = count
    return vector
end

"""
    create_hdf5_vector(
        group::HDF5.Group,
        name::AbstractString,
        type::Type;
        dims = nothing,
        chunk_length = 1000,
        portable = true,
        serialize_arrays = true,
        serialize_nonconcrete = true,
    )

Creates an empty HDF5-backed vector under `name`. Schema inference selects and records the
complete physical representation when the vector is created.
"""
function create_hdf5_vector(
    group::HDF5.Group,
    name::AbstractString,
    type::Type;
    dims = nothing,
    chunk_length = 1000,
    portable = true,
    serialize_arrays = true,
    serialize_nonconcrete = true,
)

    policy = SchemaPolicy(;
        portable,
        serialize_arrays,
        serialize_nonconcrete,
    )
    schema = infer_schema(type; dims, policy)
    return create_hdf5_vector(group, name, schema; chunk_length)

end

"""
    create_hdf5_vector(
        group::HDF5.Group,
        name::AbstractString,
        schema::AbstractSchema;
        chunk_length = 1000,
    )

Creates an empty HDF5-backed vector using an explicit schema.
"""
function create_hdf5_vector(
    group::HDF5.Group,
    name::AbstractString,
    schema::AbstractSchema;
    chunk_length = 1000,
)

    name = validate_destination(group, name)
    chunk_length = validate_chunk_length(chunk_length)
    vector_group = HDF5.create_group(group, name)
    write_schema(vector_group, schema)
    metadata_group = vector_group["metadata"]
    metadata_group["count"] = Int64(0)
    count_dataset = metadata_group["count"]

    data_group = HDF5.create_group(vector_group, "data")
    store = create_store(data_group, schema; chunk_length)
    return HDF5Vector(schema, store, count_dataset, 0)

end

"""
    load_hdf5_vector(group::HDF5.Group)
    load_hdf5_vector(group::HDF5.Group, type::Type)

Loads an HDF5-backed vector from its stored schema. The explicit-type form avoids
deserializing the logical Julia type from metadata.
"""
function load_hdf5_vector(group::HDF5.Group)
    return load_hdf5_vector(group, read_schema(group))
end

function load_hdf5_vector(group::HDF5.Group, type::Type)
    return load_hdf5_vector(group, read_schema(group, type))
end

function load_hdf5_vector(group::HDF5.Group, schema::AbstractSchema)

    validate_store_children(group, ("metadata", "data"))
    count, count_dataset = read_logical_count(group["metadata"])
    store = open_store(group["data"], schema)
    validate_logical_count(store, count)
    return HDF5Vector(schema, store, count_dataset, count)

end

#####################
# Vector Operations #
#####################

function Base.push!(vector::HDF5Vector{T}, value::T) where {T}

    # Encoding can run user constructors or Julia serialization, so it finishes before
    # physical storage changes. The logical count is persisted only after the value write
    # succeeds, and the in-memory count changes last.
    encoded = encode_value(vector.schema, value)
    next_count = vector.count + 1
    write_encoded!(vector.store, next_count, encoded)
    persist_count!(vector, next_count)
    return vector

end

function Base.getindex(vector::HDF5Vector, index::Int)
    checkbounds(vector, index)
    encoded = read_encoded(vector.store, index)
    return decode_value(vector.schema, encoded)
end

function decode_range(vector::HDF5Vector{T}, indices::UnitRange{Int}) where {T}

    encoded = read_encoded(vector.store, indices)
    values = Vector{T}(undef, length(indices))
    for index in eachindex(values)
        values[index] = decode_value(vector.schema, encoded[index])
    end
    return values

end

function Base.getindex(vector::HDF5Vector, indices::UnitRange{Int})
    checkbounds(vector, indices)
    return decode_range(vector, indices)
end

function Base.getindex(vector::HDF5Vector{T}, indices::AbstractRange{<:Integer}) where {T}
    values = Vector{T}(undef, length(indices))
    for (output_index, source_index) in enumerate(indices)
        values[output_index] = vector[source_index]
    end
    return values
end

function Base.getindex(
    vector::HDF5Vector{T},
    indices::AbstractVector{<:Integer},
) where {T}
    values = Vector{T}(undef, length(indices))
    for (output_index, source_index) in enumerate(indices)
        values[output_index] = vector[source_index]
    end
    return values
end

function Base.getindex(vector::HDF5Vector, mask::AbstractVector{Bool})
    checkbounds(vector, mask)
    return collect(vector)[mask]
end

Base.getindex(vector::HDF5Vector, ::Colon) = collect(vector)

function Base.collect(vector::HDF5Vector{T}) where {T}
    if iszero(length(vector))
        return T[]
    end
    return decode_range(vector, 1:length(vector))
end

"""
    copy_to_hdf5_vector(
        group::HDF5.Group,
        name::AbstractString,
        collection::AbstractVector;
        kwargs...,
    )

Creates an HDF5-backed vector and copies an ordinary vector into it with one recursive bulk
write.
"""
function copy_to_hdf5_vector(
    group::HDF5.Group,
    name::AbstractString,
    collection::AbstractVector{T};
    dims = nothing,
    chunk_length = 1000,
    portable = true,
    serialize_arrays = true,
    serialize_nonconcrete = true,
) where {T}

    # Destination and schema validation happen before potentially expensive encoding. The
    # complete encoded collection is then prepared before the destination group is created.
    validate_destination(group, name)
    chunk_length = validate_chunk_length(chunk_length)
    policy = SchemaPolicy(;
        portable,
        serialize_arrays,
        serialize_nonconcrete,
    )
    schema = infer_schema(T; dims, policy)
    encoded = Vector{encoded_value_type(schema)}(undef, length(collection))
    for (index, value) in enumerate(collection)
        encoded[index] = encode_value(schema, value)
    end

    vector = create_hdf5_vector(group, name, schema; chunk_length)
    write_encoded_batch!(vector.store, 1, encoded)
    persist_count!(vector, length(collection))
    return vector

end
