# Build documentation:
#
#   julia --project=docs docs/make.jl
#
# View it:
#
#   julia -e "using LiveServer; serve(dir=\"docs/build/\");"
#

# push!(LOAD_PATH,"../src/") # Do I need this?

using Documenter, HDF5Vectors

makedocs(;
    sitename = "HDF5Vectors",
    remotes = nothing,
    pages = [
        "Home" => "index.md",
        "Public Interface" => "api.md",
        "Custom Element Types" => "custom_element_types.md",
        "Custom HDF5 Vector Types" => "custom_vector_types.md",
    ],
)

deploydocs(
    repo = "github.com/Gradient-Aerospace/HDF5Vectors.jl.git",
)
