# Build documentation:
#
#   julia --project=docs docs/make.jl
#
# View it:
#
#   julia -e "using LiveServer; serve(dir=\"docs/build/\");"
#

using Documenter, HDF5Vectors

makedocs(;
    sitename = "HDF5Vectors",
    remotes = nothing,
    pages = [
        "Home" => "index.md",
        "Supported Types and Options" => "supported_types.md",
        "HDF5 Storage Layout" => "storage_layout.md",
        "Custom Element Types" => "custom_element_types.md",
        "Custom Schemas" => "custom_schemas.md",
        "API Reference" => "api.md",
        "When Writing Fails" => "write_failures.md",
    ],
)

deploydocs(
    repo = "github.com/Gradient-Aerospace/HDF5Vectors.jl.git",
)
