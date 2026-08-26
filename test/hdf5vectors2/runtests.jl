include(joinpath(@__DIR__, "..", "..", "src", "HDF5Vectors2", "HDF5Vectors2.jl"))

using .HDF5Vectors2
import HDF5
import StaticArrays

include("schema_and_codecs.jl")
include("format.jl")
include("scalar_and_constant_storage.jl")
