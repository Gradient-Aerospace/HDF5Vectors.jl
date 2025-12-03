# Custom Element Types

```@meta
CurrentModule = HDF5Vectors
```

Most types can be stored reasonably well as HDF5 vectors without having to specifying anything about how that should happen. However, sometimes it's desireable to have more control. In that case, it's possible to specify how a custom type should be stored in the HDF5 file, using one of the existing storage styles. The following methods should be implemented for the type:

* [`storage_style`](@ref)
* [`construct`](@ref)
* [`deconstruct`](@ref)

Here's a complete example of a custom type for recording student grades, where the grade itself is stored as a string, but we really that's just going to be A, B, C, D, or F. Here is the native type:

```
struct Grade
    label::String
end
```

Here's what's necessary to store this as the "elemental" style, where the label is stored as a char (UInt8).

```
using HDF5Vectors
import HDF5Vectors: storage_style, construct, deconstruct

# Tell it we want this stored using the "elemental" style, with Chars.
storage_style(::Type{Grade}; kwargs...) = HDF5Vectors.ElementalStorageStyle(UInt8)

# To store a Grade, pull the first (and only) char from the label.
deconstruct(::Type{UInt8}, el) = UInt8(only(el.label))

# To rebuild a Grade from what was stored, make a string from the char.
construct(::Type{Grade}, el) = Grade(string(Char(el)))
```

Now let's give that a try:

```
using HDF5
h5open("custom_element_type.h5", "w") do fid

    # Create the vector.
    arr = create_hdf5_vector(fid, "grades", Grade)

    # Add some grades.
    push!(arr, Grade("A"))
    push!(arr, Grade("B"))
    push!(arr, Grade("C"))
    push!(arr, Grade("D"))
    push!(arr, Grade("F"))

    # Show how that's stored in the file itself:
    @show read(fid["grades"])

    # Show that in fact Grades are rebuilt from that data.
    @show collect(arr)

end
```

The resulting output is what we'd expect:

```
read(fid["grades"]) = UInt8[0x41, 0x42, 0x43, 0x44, 0x46]
collect(arr) = Grade[Grade("A"), Grade("B"), Grade("C"), Grade("D"), Grade("F")]
```
