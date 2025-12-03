# HDF5Vectors

This package provides a mechanism for storing vectors in HDF5 files rather than in RAM. Those vectors adhere to the AbstractVector syntax and can grow over time via `push!`. This can be particularly useful for long-running calculations, where the data that gets produced is simply too much to fit in RAM.

See the documentation (TODO: link) for more on using this package.
