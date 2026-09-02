# HDF5Vectors

HDF5Vectors provides append-only vectors whose values live in an HDF5 file rather than in RAM. They support familiar `AbstractVector` reads and grow with `push!`, making them useful for incrementally logging more data than will fit in memory. Existing Julia vectors can also be copied efficiently into portable, documented HDF5 layouts.

The package can be installed from the Julia General package registry:

```
pkg> add HDF5Vectors
```

We can then create an HDF5 vector and append values to it:

```julia
import HDF5
using HDF5Vectors

HDF5.h5open("storage.h5", "w") do file
    values = create_hdf5_vector(file["/"], "values", Float64)
    for value in 1.0 : 100.0
        push!(values, value)
    end
end
```

The [documentation](https://gradient-aerospace.github.io/HDF5Vectors.jl/) begins with ordinary creation, loading, and copying, then covers supported element types, cross-language storage layouts, codecs, and custom schemas.
