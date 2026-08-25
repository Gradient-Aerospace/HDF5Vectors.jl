# Custom Element Types

```@meta
CurrentModule = HDF5Vectors
```

Most concrete structs work without customization when their fields have [supported types](supported_types.md). Define a custom storage rule only when the default field-oriented representation is unsuitable or the type needs a different reconstruction process.

## Using the Default Composite Storage

This type needs no HDF5Vectors-specific methods:

```julia
struct Measurement
    time::Float64
    name::String
    values::Vector{Float64}
end

import HDF5
using HDF5Vectors

HDF5.h5open("measurements.h5", "w") do file
    measurements = create_hdf5_vector(file["/"], "measurements", Measurement)
    push!(measurements, Measurement(0.0, "initial", [1.0, 2.0]))
end
```

The struct is stored field-by-field. Because no `dims` are declared for the `values` field, each `Vector{Float64}` is serialized independently. The other fields remain directly readable as HDF5 datasets. See [Field-Oriented Composite Values](storage_layout.md#Field-Oriented-Composite-Values) for their paths.

## Selecting Julia Byte Serialization

Define only [`storage_style`](@ref) to store an entire custom value with Julia's `Serialization` library:

```julia
struct Snapshot
    labels::Vector{String}
    values::Dict{String, Float64}
end

import HDF5Vectors: storage_style, ByteArrayStorageStyle

storage_style(::Type{Snapshot}; kwargs...) = ByteArrayStorageStyle()
```

No [`construct`](@ref) or [`deconstruct`](@ref) method is needed because the byte-array backend serializes and deserializes the complete value. This representation is appropriate only when Julia-specific bytes are acceptable.

## Selecting JSON Storage

[`JSONStorageStyle`](@ref) stores one JSON string per element. JSON3 must be a dependency of the calling project and must be loaded so that the HDF5Vectors JSON extension is available.

```julia
import JSON3
import HDF5Vectors: storage_style, JSONStorageStyle

struct ServerDetails
    hostname::String
    active::Bool
end

storage_style(::Type{ServerDetails}; kwargs...) = JSONStorageStyle()
```

The type must be supported by `JSON3.write` and `JSON3.read(..., ServerDetails)`. The resulting strings are stored at `/vector_name/data/json/data` and can be read by non-Julia HDF5 and JSON libraries.

## Defining an Elemental Representation

A custom type can reuse the elemental backend by specifying an HDF5 datatype and defining conversions between that stored datatype and the Julia value. This example stores grades as `UInt8` bytes:

```julia
struct Grade
    label::String
end

using HDF5Vectors
import HDF5Vectors: storage_style, construct, deconstruct
import HDF5Vectors: ElementalStorageStyle, HDF5VectorOfElementalTypes

storage_style(::Type{Grade}; kwargs...) = ElementalStorageStyle(UInt8)

function deconstruct(
    ::Type{HDF5VectorOfElementalTypes{Grade, UInt8}},
    grade::Grade,
)
    return UInt8(only(grade.label))
end

function construct(
    ::Type{HDF5VectorOfElementalTypes{Grade, UInt8}},
    value::UInt8,
)
    return Grade(string(Char(value)))
end
```

`deconstruct` runs before a value is written, and `construct` runs after its stored representation is read. Their first argument identifies the HDF5 vector backend and both the Julia and HDF5 element types.

The type can now be used through the ordinary interface:

```julia
import HDF5

HDF5.h5open("grades.h5", "w") do file
    grades = create_hdf5_vector(file["/"], "grades", Grade)
    push!(grades, Grade("A"))
    push!(grades, Grade("B"))
    @show read(file["grades/data"])
    @show collect(grades)
end
```

## Customizing Composite Reconstruction

Default composite reconstruction calls the declared element type with its stored field values in field order. If that constructor does not exist, define a more-specific [`construct`](@ref) method. For example, this type deliberately accepts one tuple rather than two scalar constructor arguments:

```julia
struct PointFromTuple
    x::Float64
    y::Float64

    PointFromTuple(values::Tuple{Float64, Float64}) = new(values...)
end

import HDF5Vectors: construct, HDF5VectorOfCompositeTypes

function construct(
    ::Type{HDF5VectorOfCompositeTypes{PointFromTuple}},
    values,
)
    return PointFromTuple((values[1], values[2]))
end
```

The default composite [`deconstruct`](@ref) method still reads `x` and `y` from the value. Define a custom `deconstruct` method as well only when the stored field values must be obtained differently.

## Keeping Style Selection Reproducible

[`storage_style`](@ref) is called when a vector is created and again when it is loaded. Custom methods should make the same choice from the element type and the supplied keyword options every time. Accept `kwargs...` even when the method does not currently use any options, as in the examples above.

The built-in metadata preserves `dims` and `portable`. A custom storage backend can store additional options inside its own HDF5 group, but style selection during loading cannot depend on an option that is unavailable until after the style has been selected.
