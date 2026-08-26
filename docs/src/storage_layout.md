# HDF5 Storage Layout

```@meta
CurrentModule = HDF5Vectors
```

The HDF5 layout is an important part of HDF5Vectors: many files are written in Julia and read by tools in Python, MATLAB, C++, or other environments. This page describes where values are stored so those readers do not need HDF5Vectors or Julia.

## Common Structure

Every HDF5 vector occupies an HDF5 group. The group contains `metadata` and `data`; `data` is either the value dataset or another group containing recursively stored values.

```
/x/                                      # HDF5 vector group
/x/data                                  # Dataset or group containing values
/x/metadata/                             # HDF5Vectors metadata
/x/metadata/type                         # Human-readable Julia element type
/x/metadata/serialized_type              # Julia-serialized element type
/x/metadata/dimensions_are_constant      # Whether fixed dimensions were stored
/x/metadata/dimensions                   # Julia dimensions, or an empty array
/x/metadata/portable                     # Value of the portable option
```

Readers outside Julia usually need only the paths under `data`. The plain-text `metadata/type` value can be informative, but `metadata/serialized_type` uses Julia's `Serialization` format and is used by `load_hdf5_vector(group)` to reconstruct the Julia element type. Nested HDF5 vectors, such as the fields of a composite value, have their own `metadata` and `data` children.

HDF5Vectors is intended to load trusted files. `load_hdf5_vector(group)` deserializes `metadata/serialized_type`; the explicit-type form `load_hdf5_vector(group, el_type)` avoids that metadata deserialization, but reading a vector with Julia byte serialization still deserializes its element values. Data from an untrusted HDF5 file should not be deserialized.

## Elemental Values

Numbers, booleans, strings, symbols, characters, and enums use a one-dimensional dataset at `data`:

```
/x/               # HDF5 vector group
/x/data           # Dataset with N values
/x/metadata/      # Metadata group
```

Native integers and floats retain their HDF5 datatypes, while `Bool` uses an 8-bit HDF5 bitfield. A `Symbol` is stored as a string, a `Char` as an `Int32` Unicode code point, and an enum as its integer base type. An external reader must know the Julia-level interpretation if it needs to reconstruct those transformed values.

## Array-Like Values

For Julia elements with dimensions `(M, N, ...)`, HDF5Vectors presents the value dataset in Julia with dimensions `(M, N, ..., Z)`, where `Z` is the number of vector elements. The last Julia dimension indexes the HDF5 vector.

HDF5.jl reverses multidimensional extents at the HDF5 C-format boundary to account for Julia's column-major array order. A row-major reader such as h5py therefore observes the raw HDF5 shape `(Z, ..., N, M)`. For example, 100 Julia matrices with size `(2, 3)` are stored in a dataset that HDF5.jl presents as `(2, 3, 100)` and h5py presents as `(100, 3, 2)`. In h5py, `data[k]` selects one vector element, with that element's axes in reversed order relative to Julia.

```
/positions/              # HDF5 vector group
/positions/data          # Multidimensional value dataset
/positions/metadata/     # Includes dimensions in Julia order
```

This layout is used for supported `SVector`, `SMatrix`, `SArray`, and homogeneous `NTuple` values, as well as `Vector`, `Matrix`, and `Array` values created with `dims`.

## Field-Oriented Composite Values

Portable composite storage gives every field its own nested HDF5 vector. These Julia types provide an example:

```julia
struct MySubType
    c::Int64
    d::NTuple{2, Float64}
end

struct MyType
    a::Float64
    b::MySubType
end
```

For a vector named `my_type` inside `/my_group`, the value datasets are stored at the following paths. Each named field path is itself an HDF5 vector group; intermediate metadata groups are omitted from this diagram for clarity.

```
/my_group/my_type/                      # HDF5 vector for MyType
/my_group/my_type/data/a/               # HDF5 vector for field a
/my_group/my_type/data/a/data           # Float64 dataset for field a
/my_group/my_type/data/b/               # HDF5 vector for field b
/my_group/my_type/data/b/data/c/        # HDF5 vector for nested field c
/my_group/my_type/data/b/data/c/data    # Int64 dataset for nested field c
/my_group/my_type/data/b/data/d/        # HDF5 vector for nested field d
/my_group/my_type/data/b/data/d/data    # Array-like dataset for nested field d
```

Each field dataset contains the same number of vector elements. External readers can reconstruct one `MyType` by reading the same vector index from `a`, `b.c`, and `b.d`.

## Native Bits-Type Values

When a bits-type composite is created with `portable = false`, HDF5.jl represents the entire Julia value with one HDF5 datatype:

```
/my_group/my_type/          # HDF5 vector group
/my_group/my_type/data      # Dataset of the HDF5 datatype derived from MyType
```

This avoids separate field datasets and is generally faster. An external reader must inspect and interpret the resulting HDF5 datatype, so field-oriented storage is preferable when simple cross-language access matters more than performance.

## Singleton Values

A singleton element type has only one possible value. Its `data` dataset contains one `Int64`: the logical length of the HDF5 vector.

```
/markers/data       # One Int64 containing the number of marker values
```

No individual values are stored. An external reader needs the element type's meaning from the surrounding application or the metadata.

## Julia-Serialized Values

Byte-array storage concatenates the independently serialized elements and records the cumulative ending byte position of each element:

```
/values/data/bytes/          # Nested HDF5 vector of UInt8
/values/data/bytes/data      # Concatenated serialized bytes
/values/data/stops/          # Nested HDF5 vector of Int64
/values/data/stops/data      # Cumulative byte counts after each element
```

For a zero-based reader, element `k` starts at zero when `k == 0` and otherwise at `stops[k - 1]`; it ends immediately before `stops[k]`. The byte sequences use Julia's `Serialization` format and are not intended for reconstruction outside Julia. Loading them requires the relevant Julia types and modules to remain available and compatible with the serialized representation.

## JSON-Serialized Values

`JSONStorageStyle` stores one compact JSON string for each vector element:

```
/values/data/json/          # Nested HDF5 vector of String
/values/data/json/data      # Dataset containing N JSON strings
```

External readers can parse the strings with their normal JSON library. The JSON representation and supported Julia types are determined by JSON3.
