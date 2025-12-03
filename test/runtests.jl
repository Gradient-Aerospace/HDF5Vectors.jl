using Test
using HDF5Vectors
using HDF5
using EnumX
using StaticArrays
import JSON3

# This tests what we need from an AbstractArray.
function test_collection(
    fid, name, source::Vector{T};
    chunk_length = 5, # Using a small number to make sure we need multiple chunks
    native = false,
    create_kwargs = (;),
) where {T}

    println("Testing $name")

    # Create the HDF5Vector from the source.
    arr = create_hdf5_vector(fid["/"], name, T; chunk_length, create_kwargs...)
    for el in source
        push!(arr, el)
    end

    # Test the many little array functions.
    @test length(arr) == length(source)
    @test size(arr) == (length(source),)
    @test arr[1] == source[1] # Check indexing.
    @test arr[end] == source[end] # Check end (lastindex).
    @test collect(arr) == source
    @test eltype(arr) == T
    # @show collect(arr)
    @test map(identity, iterable(arr)) == source
    @test mapreduce(identity, (a, b) -> b, iterable(arr)) == source[end]
    if T <: Real
        @test sum(iterable(arr)) == sum(source) # Check mapreduce.
    end

    # We notably don't test pure iteration here; we expect that to be painfully slow, and
    # we also know it will work because indexing works.

    # If the way the array is stored in HDF5 should match the Julia type directly, so load
    # in the raw HDF5 array and compare to that.
    if native
        @test read(fid[name]["data"]) == source
    end

    # Check that copying to an HDF5 vector works too.
    arr2 = copy_to_hdf5_vector(fid["/"], name * "-copy", source; chunk_length, create_kwargs...)
    @test collect(arr2) == source

    # Now try loading a the array.
    # arr3 = load_hdf5_vector(fid[name], T; create_kwargs...)
    arr3 = load_hdf5_vector(fid[name])
    @test collect(arr3) == source

    # Now test that we can continue writing to the HDF5 vector.
    for el in source
        push!(arr3, el)
    end
    @test collect(arr3) == vcat(source, source)

end

# Here are some custom things we can work with.
@enum Birds sparrow hawk sparrowhawk
@enumx Ungulates deer horse bison
struct MyType
    a::Int64
    b::Float64
end
struct MyTypeOfTypes
    x::SVector{3, Float64}
    y::MyType
end
struct MyNonBitsType
    s::String
    v::Vector{Int64}
end
Base.:(==)(a::MyNonBitsType, b::MyNonBitsType) = a.s == b.s && a.v == b.v

struct MySerializingType
    x::String
    y::Vector{Float64}
    z::MyType
end
HDF5Vectors.storage_style(::Type{MySerializingType}; kwargs...) = HDF5Vectors.ByteArrayStorageStyle()
Base.:(==)(a::MySerializingType, b::MySerializingType) = a.x == b.x && a.y == b.y && a.z == b.z

struct MyJSONishType
    x::String
    y::Vector{Float64}
    z::MyType
end
HDF5Vectors.storage_style(::Type{MyJSONishType}; kwargs...) = HDF5Vectors.JSONStorageStyle()
Base.:(==)(a::MyJSONishType, b::MyJSONishType) = a.x == b.x && a.y == b.y && a.z == b.z

out_dir = "out"
mkpath("out")

@testset "elemental types" begin
    h5open("$out_dir/elemental_types.h5", "w") do fid
        test_collection(fid, "ints", collect(1 : 10); native = true)
        test_collection(fid, "floats", collect(1. : 12.); native = true)
        test_collection(fid, "enums", [sparrowhawk, hawk, sparrow])
        test_collection(fid, "enumxs", [Ungulates.horse, Ungulates.deer, Ungulates.bison, Ungulates.deer, Ungulates.horse, Ungulates.deer])
        test_collection(fid, "chars", collect('a' : 'z'))
        test_collection(fid, "strings", collect("element $k" for k in 1:9); native = true)
    end
end

@testset "array types" begin
    h5open("$out_dir/array_types.h5", "w") do fid
        test_collection(fid, "ntuples_of_ints", [(k, 2k) for k in 1:11])
        test_collection(fid, "svectors_of_floats", [SA_F64[k, 2k, 3k] for k in 1:12])
        test_collection(fid, "smatrices_of_floats", [SA_F64[k 2k; 3k 4k] for k in 1:12])
        test_collection(fid, "sarrays_of_ints", [SA_F64[k 2k; 3k 4k;;; 5k 6k; 7k 8k] for k in 1:12])
        test_collection(fid, "vectors_of_floats", [Float64[k, 2k, 3k] for k in 1:12]; create_kwargs = (; dims = (3,), ))
        test_collection(fid, "vectors_of_floats_no_dims", [Float64[k, 2k, 3k] for k in 1:12])
        test_collection(fid, "matrices_of_floats", [Float64[k 2k; 3k 4k] for k in 1:12]; create_kwargs = (; dims = (2,2), ))
        test_collection(fid, "matrices_of_floats_no_dims", [Float64[k 2k; 3k 4k] for k in 1:12])
    end
end

@testset "composite types" for portable in (true, false)
    create_kwargs = (; portable, )
    h5open("$out_dir/composite_types" * (portable ? "_portable" : "") * ".h5", "w") do fid
        test_collection(fid, "complex_numbers", [k * (1. + 2im) for k in 1:11]; create_kwargs) # HDF5 will handle this natively, but for portability, we use a composite type.
        test_collection(fid, "rational_numbers", [k // 100 for k in 1:11]; create_kwargs)
        test_collection(fid, "tuples_of_reals", [(float(k), 2k) for k in 1:11]; create_kwargs) # Different types make this a composite type.
        test_collection(fid, "tuples_of_real", [(float(k),) for k in 1:11]; create_kwargs) # Just to make sure collections of 1 element don't do weird things.
        test_collection(fid, "named_tuples", [(; a = float(k), b = 2k) for k in 1:11]; create_kwargs)
        test_collection(fid, "tuples_of_composites", [(float(k), (; a = float(2k), b = 3k)) for k in 1:11]; create_kwargs)
        test_collection(fid, "tuples_of_non_bits_types", [(string(k), (; a = float(2k), b = 3k)) for k in 1:11]; create_kwargs)
        test_collection(fid, "named_tuples_of_composites", [(; x = float(k), y = (; a = float(2k), b = 3k)) for k in 1:11]; create_kwargs)
        test_collection(fid, "named_tuples_of_non_bits_types", [(; x = string(k), y = (; a = float(2k), b = 3k)) for k in 1:11]; create_kwargs)
        test_collection(fid, "svectors_of_tuples_of_whatever", [SA[(k, string(2k)), (3k, string(4k))] for k in 1:11])
        test_collection(fid, "structs", [MyType(k, 2k) for k in 1:11])
        test_collection(fid, "structs_of_structs", [MyTypeOfTypes(SA_F64[k, 2k, 3k], MyType(k, 2k)) for k in 1:11])
        test_collection(fid, "non_bits_structs", [MyNonBitsType(string(k), [k, 2k, 3k]) for k in 1:11])
    end
end

@testset "serialization types" begin
    h5open("$out_dir/serialization_types.h5", "w") do fid
        test_collection(fid, "serializing_types", [MySerializingType(string(k), [k, 2k, 3k], MyType(4k, 5k)) for k in 1:11])
        test_collection(fid, "json_types", [MyJSONishType(string(k), [k, 2k, 3k], MyType(4k, 5k)) for k in 1:11])
    end
end
