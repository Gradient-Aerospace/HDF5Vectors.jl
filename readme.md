# HDF5Vectors

HDF5Vectors provides vectors whose underlying values live in an HDF5 file rather than in RAM. They support familiar `AbstractVector` operations and can grow over time with `push!`, making them useful for incrementally logging more data than will fit in memory.

The package can be installed directly from GitHub:

```
pkg> add https://github.com/Gradient-Aerospace/HDF5Vectors.jl
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

The [documentation](https://gradient-aerospace.github.io/HDF5Vectors.jl/) covers supported element types, loading stored vectors, storage layouts, and customization.
