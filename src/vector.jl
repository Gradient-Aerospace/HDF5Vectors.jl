#######################
# Public Vector Type  #
#######################

"""
    HDF5Vector{T} <: AbstractVector{T}

An append-only HDF5-backed vector with logical element type `T`. It supports ordinary
`AbstractVector` reads and grows with `push!`. Vectors are normally constructed with
[`create_hdf5_vector`](@ref), [`copy_to_hdf5_vector`](@ref), or [`load_hdf5_vector`](@ref),
and remain usable only while their HDF5 file is open.
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

Creates an empty `HDF5Vector{type}` in the new child `name` of `group`. `name` must be one
unused HDF5 path component. Schema inference selects and records the complete physical
representation before the vector is returned.

`dims` declares the fixed shape of a dynamically sized array element. `chunk_length`
controls the extensible dimension of physical HDF5 chunks. `portable = true` gives bits-type
records field-oriented storage. `serialize_arrays` and `serialize_nonconcrete` control
whether schema inference may select Julia serialization for those categories.

Values later passed to `push!` must already have exactly the declared element type.
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
    inference_options = (; dims, policy)
    return create_hdf5_vector_from_schema(
        group,
        name,
        schema;
        chunk_length,
        inference_options,
    )

end

"""
    create_hdf5_vector(
        group::HDF5.Group,
        name::AbstractString,
        schema::AbstractSchema;
        chunk_length = 1000,
    )

Creates an empty HDF5-backed vector using an already selected schema. The schema determines
the logical type, codecs, and physical representation; only `chunk_length` remains a
creation option.
"""
function create_hdf5_vector(
    group::HDF5.Group,
    name::AbstractString,
    schema::AbstractSchema;
    chunk_length = 1000,
)
    return create_hdf5_vector_from_schema(
        group,
        name,
        schema;
        chunk_length,
        inference_options = nothing,
    )
end

function create_hdf5_vector_from_schema(
    group::HDF5.Group,
    name::AbstractString,
    schema::AbstractSchema;
    chunk_length,
    inference_options,
)

    name = validate_destination(group, name)
    chunk_length = validate_chunk_length(chunk_length)
    vector_group = HDF5.create_group(group, name)
    write_schema(vector_group, schema; inference_options)
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
    load_hdf5_vector(group::HDF5.Group, schema::AbstractSchema)

Opens and validates an existing HDF5 vector group.

The one-argument form deserializes the exact stored schema. The explicit-type form repeats
schema inference from the options stored when the vector was created from a type, avoiding
schema deserialization. A vector created from an explicit schema still needs that stored
schema unless the caller supplies a schema directly. A supplied schema is authoritative:
loading validates that it can read the physical layout, but does not require all of its
logical details to equal the stored schema. This can help load a file after its Julia type
or schema implementation has changed.

The returned vector remains usable only while the HDF5 file is open. Files whose schema or
values require Julia deserialization should be trusted.
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
    next_count = Base.Checked.checked_add(vector.count, 1)
    append_encoded!(vector.store, next_count, encoded)
    persist_count!(vector, next_count)
    return vector

end

function Base.getindex(vector::HDF5Vector, index::Int)
    checkbounds(vector, index)
    encoded = read_encoded(vector.store, index)
    return decode_value(vector.schema, encoded)
end

function decode_range(vector::HDF5Vector{T}, indices::UnitRange{Int}) where {T}
    encoded = read_encoded_batch(vector.store, indices)
    return decode_batch(vector.schema, encoded)
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

Creates an HDF5-backed vector and copies `collection` into it with one recursive bulk write.
The declared element type is `eltype(collection)`, and the creation options have the same
meaning as for [`create_hdf5_vector`](@ref).

Destination validation, schema inference, and encoding of the complete collection finish
before the destination group is created. The input must currently be an `AbstractVector`.
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
    encoded = encode_batch(schema, collection)

    vector = create_hdf5_vector_from_schema(
        group,
        name,
        schema;
        chunk_length,
        inference_options = (; dims, policy),
    )
    initialize_encoded!(vector.store, encoded)
    persist_count!(vector, length(collection))
    return vector

end
