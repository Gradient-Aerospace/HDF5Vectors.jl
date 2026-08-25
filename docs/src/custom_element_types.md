# Custom Element Types

```@meta
CurrentModule = HDF5Vectors
```

Many common types can be stored as HDF5 vectors without specifying how that should happen. Sometimes it is useful to have more control. Selecting an existing serialization style requires only [`storage_style`](@ref). Defining a custom representation may also require [`construct`](@ref) and [`deconstruct`](@ref):

* [`storage_style`](@ref)
* [`construct`](@ref)
* [`deconstruct`](@ref)

Here's a complete example of a custom type for recording student grades, where the grade itself is stored as a string, where that string is going to be A, B, C, D, or F. Here is the native type:

```
struct Grade
    label::String
end
```

Here's what's necessary to use the "elemental" style, storing the grade as one `UInt8` byte.

```
using HDF5Vectors
import HDF5Vectors: storage_style, construct, deconstruct, ElementalStorageStyle, HDF5VectorOfElementalTypes

# Tell it we want this stored using the "elemental" style, with UInt8 as the element type.
storage_style(::Type{Grade}; kwargs...) = ElementalStorageStyle(UInt8)

# To store a Grade, pull the first (and only) char from the label string and convert to UInt8.
deconstruct(::Type{HDF5VectorOfElementalTypes{Grade, UInt8}}, el::Grade) = UInt8(only(el.label))

# To rebuild a Grade from what was stored, make a string from the char.
construct(::Type{HDF5VectorOfElementalTypes{Grade, UInt8}}, el::UInt8) = Grade(string(Char(el)))
```

Now let's give that a try:

```
using HDF5
h5open("custom_element_type.h5", "w") do fid

    # Create the vector.
    arr = create_hdf5_vector(fid["/"], "grades", Grade)

    # Add some grades.
    push!(arr, Grade("A"))
    push!(arr, Grade("B"))
    push!(arr, Grade("C"))
    push!(arr, Grade("D"))
    push!(arr, Grade("F"))

    # Show how that's stored in the file itself:
    @show read(fid["grades"]["data"])

    # Show that in fact Grades are rebuilt from that data.
    @show collect(arr)

end
```

The resulting output is what we'd expect:

```
read(fid["grades"]["data"]) = UInt8[0x41, 0x42, 0x43, 0x44, 0x46]
collect(arr) = Grade[Grade("A"), Grade("B"), Grade("C"), Grade("D"), Grade("F")]
```
