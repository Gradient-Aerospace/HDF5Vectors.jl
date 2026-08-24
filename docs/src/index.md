# HDF5Vectors

```@meta
CurrentModule = HDF5Vectors
```

This package provides a mechanism for storing vectors in HDF5 files rather than in RAM. Those vectors adhere to the AbstractVector syntax and can grow over time via `push!`. This can be particularly useful for long-running calculations, where the data that gets produced is simply too much to fit in RAM. It's also useful as a way to copy an array from Julia into an HDF5 file in a "portable" way that can be accessed outside of Julia, as well as a way to load those same vectors back into Julia.

## Example

Here is a simple example. We use HDF5 to open a file, and we use HDF5Vectors to create a vector in that HDF5 file.

```
import HDF5
using HDF5Vectors

# Create an HDF5 file to store stuff in.
fid = HDF5.h5open("storage.h5", "w")

# Create an array for Float64s called "x" in that file.
arr = create_hdf5_vector(fid["/"], "x", Float64)

# Push some elments into the array.
for el in 1. : 100.
    push!(arr, el)
end

# We can now do whatever we want with that array, such as
@show arr[end]
@show sum(arr)
@show collect(arr)

# Always close out the file when you're done.
close(fid)
```

We can also create a new HDF5 vector based on existing storage in an HDF5 file.

```
fid = HDF5.h5open("storage.h5")
arr = load_hdf5_vector(fid["/x"], Float64)
```

Further, you can move an existing array into HDF5 like so:

```
fid = HDF5.h5open("storage.h5")
my_hdf5_vector = copy_to_hdf5_vector(fid["/"], "some_name", my_regular_vector)
```

Note that writing into an HDF5 file element-by-element is much slower than keeping arrays in RAM, as is reading data back element-by-element, so HDF5Vectors should be used judiciously.

Values passed to `push!` must already be instances of the HDF5Vector's declared element type; HDF5Vectors does not convert them to that type.

## Supported Types

This works for storing many common types, including:

Elemental types:

* Int8, Int16, Int32, and Int64 (and unsigned forms)
* Float32 and Float64
* Enum
* Char
* Bits-type structs

String-like types:

* String
* Symbol

Singleton types:

* `Nothing`
* Immutable zero-field marker types with zero-argument constructors
* Empty tuples, named tuples, and static arrays

Array-like types:

* SVector, SMatrix, and SArray of elemental type
* Vector, Matrix, and Array of element type, where dimensions are constant from element to element
* NTuple of elemental type

Composite types:

* General tuple of types on this list
* General named tuple of types on this list
* General struct of types on this list

Serialized types:

* Vector, Matrix, and Array of non-elemental type or whose dimensions are not known in advance
* Custom types explicitly assigned `ByteArrayStorageStyle` or `JSONStorageStyle`

Serialization provides a fallback approach for logging to HDF5 when other storage types do not make sense. For instance, a nonconcrete type, whose structure may change from element to element, defaults to serialization. This is slow and is intended only for types selected by the rules above. See "Specifying a Storage Type" below for more.

Primitive types that HDF5.jl does not natively support, including `Float16`, `Int128`, and `UInt128`, are rejected unless the user defines another storage style for them.

## Iteration

When iterating over an HDF5Vector, it's far faster to call [`iterable`](@ref) on the vector and then iterate on what that returns. For example:

```
arr = create_hdf5_vector(...)
...
[el.x^2 + el.y^2 for el in iterable(arr)]
```

The reason this is faster is that [`iterable`](@ref) creates a structure intended to take advantage of the way HDF5.jl will access the data.

## Loading an Existing Array

Loading an HDF5 vector stored in an HDF5 file is straightforward:

```
fid = HDF5.h5open("storage.h5")
arr = load_hdf5_vector(fid["/x"])
```

Where the type to load is known in advance, this works as well:

```
fid = HDF5.h5open("storage.h5")
arr = load_hdf5_vector(fid["/x"], Float64)
```

## How Data Is Stored

### Elemental Types

These will simply be n-element arrays in the HDF5 file.

### Array-Like Types

When the elements to be stored each have dimensions like (M, N, ...), the HDF5 file will have an array of the appropriate type whose dimensions are (M, N, ..., Z), where Z is the number of elements being stored. This is easy to interpret outside of Julia while also allowing fast access and efficient storage.

When the elements to store are Vector, Matrix, or Array (or any AbstractArray whose dimensions are not known from the type), the `dims = (M, N, ...)` argument must be provided to `create_hdf_vector`. Otherwise, Arrays cannot be generally stored with array-like storage and will instead be serialized to byte arrays, which if far slower and unintepretable outside of Julia.

### Composite Types

There are two ways to store a composite type: "portable" and "non-portable".

"Portable" storage means that an HDF5 group is created for each field of the struct, and inside of each is a group/dataset.

To see how structs can be stored portably, imagine we want a vector of 100 `MyType` elements, where:

```
struct MySubType
    c::Int64                # Elemental
    d::NTuple{2, Float64}   # Array-like
end
struct MyType
    a::Float64              # Elemental
    b::MySubType            # Composite
end
```

If the HDF5 vector were created in the `/my_group` group with the name `my_type`, those would be stored portably like so:

```
/my_group/my_type/               # Group
/my_group/my_type/data/a         # Array of 100 Float64
/my_group/my_type/data/b         # Group
/my_group/my_type/data/b/data/c  # Array of 100 Int64
/my_group/my_type/data/b/data/d  # Array of 2-by-100 Float64
```

Note that a failed `push!` for a composite type may result in some fields of the composite type having updated values while others do not.

For bits-type structs, a user can specify that they want "non-portable" storage. This means that the HDF5.jl package can define a custom HDF5 type to store the struct, and the resulting HDF5 file will look like this:

```
/my_group/my_type/ # An array of the HDF5 custom type
```

This is much faster and more efficient, but accessing it outside of Julia will require substantially more code to work with the HDF5 type definition system.

### Serialized Types

When a type is too flexible/odd for the above, it can be stored via serialization. This is far slower, but it works for almost everything. There are two types of serialization currently provided.

The `ByteArrayStorageStyle` uses Julia's Serialization package to serialize a given type to a byte array. The resulting HDF5 dataset will be uninterpretable outside of Julia.

The `JSONStorageStyle` uses the JSON3 package to serialize a given type to a JSON string. The resulting HDF5 dataset will be an array of JSON strings. See the example below. In order to use the `JSONStorageStyle`, your project will have to import JSON3.

## Specifying a Storage Type

Users can specify what "style" of storage should be used for a given type. Storage style is a property of the element type and relevant creation options, not an override for one particular vector. The same [`storage_style`](@ref) method is called again when loading the vector, so it must make a consistent selection from those inputs.

For instance, suppose we had the following type:

```
@enum ServerStatus unknown up down
struct SomeServerDetails
    hostname::String
    status::ServerStatus
end
```

When that's stored in the HDF5 file, let's make it serialize to JSON. To do this, we can apply the following trait:

```
import HDF5Vectors: storage_style, JSONStorageStyle
storage_style(::Type{SomeServerDetails}; kwargs...) = JSONStorageStyle()
```

That's it. Now, when we create an array for this type, each element will be serialized to JSON. The resulting HDF5 file will have an array of strings that can be loaded in any other environment. To complete the example:

```
import HDF5
using HDF5Vectors
HDF5.h5open("server_details.h5", "w") do fid
    details = create_hdf5_vector(fid, "details", SomeServerDetails)
    push!(details, SomeServerDetails("localhost", up))
    push!(details, SomeServerDetails("old_pc", down))
    push!(details, SomeServerDetails("phone", unknown))
    @show collect(details)
end
```

The HDF5 file will have the following:

```
/details/data/json/data # An array of JSON strings, one for each of the pushed elements
```

(By default, the JSON is compact in style, not "pretty", to reduce the extra storage burden of all of those spaces. You can "pretty" on extraction from the HDF5 file if a human is supposed to look at it.)
