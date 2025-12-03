module HDF5VectorsJSON3Ext

# println("Loading HDF5VectorsJSON3Ext")

################################
# Types that Serialize to JSON #
################################

import HDF5
import JSON3
import HDF5Vectors

# Create a type to handle anything that needs to go to/from JSON. We'll just store a single-
# dataset HDF5 vector of strings inside.
mutable struct HDF5VectorWithJSONStorage{T} <: HDF5Vectors.AbstractHDF5Vector{T}
    storage::HDF5Vectors.HDF5VectorOfElementalTypes{String, String}
end
function HDF5Vectors.create_hdf5_vector(style::HDF5Vectors.JSONStorageStyle, group, name, el_type; portable, kwargs...)
    this_group = HDF5.create_group(group, string(name))
    HDF5Vectors.store_metadata(style, this_group, el_type; portable)
    data_group = HDF5.create_group(this_group, "data")
    return HDF5VectorWithJSONStorage{el_type}(
        HDF5Vectors.create_hdf5_vector(data_group, "json", String; kwargs...),
    )
end
function HDF5Vectors.load_hdf5_vector(style::HDF5Vectors.JSONStorageStyle, group_or_dataset, el_type; kwargs...)
    return HDF5VectorWithJSONStorage{el_type}(
        HDF5Vectors.load_hdf5_vector(group_or_dataset["data"]["json"], String; kwargs...),
    )
end
Base.length(arr::HDF5VectorWithJSONStorage) = length(arr.storage)
function Base.push!(arr::HDF5VectorWithJSONStorage, el)
    push!(arr.storage, JSON3.write(el))
    return arr
end
function Base.setindex!(arr::HDF5VectorWithJSONStorage, el, k)
    setindex!(arr.storage, JSON3.write(el), k)
    return el
end
function Base.getindex(arr::HDF5VectorWithJSONStorage{T}, k) where {T}
    return JSON3.read(getindex(arr.storage, k), T)
end
function Base.collect(arr::HDF5VectorWithJSONStorage{T}) where {T}
    data = collect(arr.storage)
    return T[JSON3.read(el, T) for el in data]
end

end
