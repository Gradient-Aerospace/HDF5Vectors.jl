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
    @test arr[:] == source
    idxs = 1:min(3, length(source))
    @test arr[idxs] == source[idxs]
    @test arr[collect(idxs)] == source[collect(idxs)]
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

    # If the way the array is stored in HDF5 should match the Julia type directly, load
    # in the raw HDF5 array and compare to that.
    if native
        @test read(fid[name]["data"]) == source
    end

    # Check that copying to an HDF5 vector works too.
    arr2 = copy_to_hdf5_vector(fid["/"], name * "-copy", source; chunk_length, create_kwargs...)
    @test collect(arr2) == source

    # Check that the copied vector was fully persisted, rather than only returning an
    # in-memory vector that matches the source.
    arr2_reloaded = load_hdf5_vector(fid[name * "-copy"])
    @test collect(arr2_reloaded) == source

    # Now try loading the array using metadata and also via the explicit-el_type overload.
    arr3 = load_hdf5_vector(fid[name])
    @test collect(arr3) == source

    # Load again, this time specifying the element type explicitly. We'll need to forward
    # the same kwargs that we used when creating the vector.
    arr4 = load_hdf5_vector(fid[name], T; create_kwargs...)
    @test collect(arr4) == source

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

struct MyNoncreteType
    x::NamedTuple
end

struct MySingletonType
end

struct MySingletonWithoutZeroArgumentConstructor
    MySingletonWithoutZeroArgumentConstructor(::Nothing) = new()
end

mutable struct MyMutableZeroFieldType
end

out_dir = "out"
mkpath("out")

@testset "elemental types (portable = $portable)" for portable in (true, false)
    h5open("$out_dir/elemental_types.h5", "w") do fid
        test_collection(fid, "ints", collect(1 : 10); native = true)
        test_collection(fid, "floats", collect(1. : 12.); native = true)
        test_collection(fid, "enums", [sparrowhawk, hawk, sparrow])
        test_collection(fid, "enumxs", [Ungulates.horse, Ungulates.deer, Ungulates.bison, Ungulates.deer, Ungulates.horse, Ungulates.deer])
        test_collection(fid, "chars", collect('a' : 'z'))
        test_collection(fid, "strings", collect("element $k" for k in 1:9); native = true)
        test_collection(fid, "symbols", [:a for _ in 1:9])
        test_collection(fid, "bools", [isodd(k) for k in 1:9])
    end
end

@testset "array types (portable = $portable)" for portable in (true, false)
    h5open("$out_dir/array_types.h5", "w") do fid
        test_collection(fid, "ntuples_of_ints", [(k, 2k) for k in 1:11])
        test_collection(fid, "svectors_of_floats", [SA_F64[k, 2k, 3k] for k in 1:12])
        test_collection(fid, "smatrices_of_floats", [SA_F64[k 2k; 3k 4k] for k in 1:12])
        test_collection(fid, "sarrays_of_ints", [SA_F64[k 2k; 3k 4k;;; 5k 6k; 7k 8k] for k in 1:12])
        test_collection(fid, "vectors_of_floats", [Float64[k, 2k, 3k] for k in 1:12]; create_kwargs = (; dims = (3,), ))
        test_collection(fid, "vectors_of_floats_no_dims", [Float64[k, 2k, 3k] for k in 1:12])
        test_collection(fid, "matrices_of_floats", [Float64[k 2k; 3k 4k] for k in 1:12]; create_kwargs = (; dims = (2,2), ))
        test_collection(fid, "matrices_of_floats_no_dims", [Float64[k 2k; 3k 4k] for k in 1:12])
        test_collection(fid, "ntuples_of_symbols", [(:a, :b) for k in 1:3])
        test_collection(fid, "svectors_of_symbols", [SVector{2, Symbol}(:a, :b) for k in 1:2])
    end
end

@testset "singleton types (portable = $portable)" for portable in (true, false)

    # Singleton vectors store only their length, but they should otherwise support the same
    # create, push, copy, reload, and append operations as vectors with stored values.
    create_kwargs = (; portable, )
    h5open("$out_dir/singleton_types.h5", "w") do fid

        # Exercise built-in singleton containers, including zero-size static arrays whose
        # shapes still differ at the type level.
        test_collection(fid, "singleton_ntuples", [Tuple{}() for _ in 1:11]; create_kwargs)
        test_collection(
            fid,
            "singleton_svectors_of_floats",
            [SVector{0, Float64}() for _ in 1:12];
            create_kwargs,
        )

        test_collection(
            fid,
            "singleton_smatrices_of_floats",
            [SMatrix{0, 2, Float64, 0}() for _ in 1:12];
            create_kwargs,
        )
        test_collection(
            fid,
            "singleton_sarrays_of_floats",
            [SArray{Tuple{0, 2, 1}, Float64, 3, 0}() for _ in 1:12];
            create_kwargs,
        )

        # Exercise ordinary user-defined and Julia singleton types, including the empty
        # named tuple whose type does not provide a zero-argument constructor.
        test_collection(
            fid,
            "singleton_types",
            [MySingletonType() for _ in 1:10];
            create_kwargs,
        )
        test_collection(fid, "singleton_named_tuples", [(;) for _ in 1:10]; create_kwargs)
        test_collection(fid, "nothings", fill(nothing, 10); create_kwargs)
        test_collection(fid, "vals", fill(Val(:reset), 10); create_kwargs)

    end

end

@testset "unsupported zero-field types" begin

    h5open("$out_dir/unsupported_zero_field_types.h5", "w") do fid

        # Unsupported primitives must not be mistaken for singletons merely because they
        # have no fields. Likewise, singleton storage must preserve value semantics rather
        # than silently accepting unreconstructible or mutable zero-field types.
        unsupported_types = (
            Float16,
            Int128,
            UInt128,
            MySingletonWithoutZeroArgumentConstructor,
            MyMutableZeroFieldType,
        )
        for type in unsupported_types
            @test_throws ArgumentError create_hdf5_vector(fid["/"], string(type), type)
        end

    end

end

@testset "composite types (portable = $portable)" for portable in (true, false)

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
        test_collection(fid, "named_tuples_of_symbols", [(; a = :a_symbol, b = :b_symbol) for _ in 1:3])
        test_collection(fid, "svectors_of_tuples_of_whatever", [SA[(k, string(2k)), (3k, string(4k))] for k in 1:11])
        test_collection(fid, "structs", [MyType(k, 2k) for k in 1:11])
        test_collection(fid, "structs_of_structs", [MyTypeOfTypes(SA_F64[k, 2k, 3k], MyType(k, 2k)) for k in 1:11])
        test_collection(fid, "non_bits_structs", [MyNonBitsType(string(k), [k, 2k, 3k]) for k in 1:11])

        # composite representation of SMatrix/SArray when element type is non-elemental
        test_collection(fid, "smatrix_of_mytype", [SMatrix{2, 2, MyType, 4}((MyType(k,k), MyType(k+1,k+1), MyType(k+2,k+2), MyType(k+3,k+3))) for k in 1:5]; create_kwargs)
        test_collection(fid, "sarray_of_mytype", [SArray{Tuple{2, 2}, MyType}((MyType(k,k), MyType(k+1,k+1), MyType(k+2,k+2), MyType(k+3,k+3))) for k in 1:5]; create_kwargs)

    end

end

@testset "in-memory counts follow successful writes" begin

    h5open("$out_dir/failed_push.h5", "w") do fid

        # The declared element type is a strict input contract rather than a conversion
        # request.
        arr = create_hdf5_vector(fid["/"], "values", Int64)
        @test_throws MethodError push!(arr, Int32(1))
        @test isempty(arr)

        # A closed dataset provides a deterministic HDF5 write failure. The failed push
        # must not advance the vector's in-memory count.
        close(arr.dataset)
        @test_throws ErrorException push!(arr, 1)
        @test isempty(arr)

    end

end

@testset "serialization types" begin

    h5open("$out_dir/serialization_types.h5", "w") do fid

        test_collection(fid, "nonconcrete_types", [MyNoncreteType((; k, )) for k in 1:10])

        # These custom concrete types select their serialization formats through
        # storage_style, so the same format is selected when copied vectors are reloaded.
        serializing_types = [
            MySerializingType(string(k), [k, 2k, 3k], MyType(4k, 5k)) for k in 1:11
        ]
        json_types = [
            MyJSONishType(string(k), [k, 2k, 3k], MyType(4k, 5k)) for k in 1:11
        ]
        test_collection(fid, "serializing_types", serializing_types)
        test_collection(fid, "json_types", json_types)

    end

end

# verify error behaviour for unsupported operations
@testset "error conditions" begin
    h5open("$out_dir/error_conditions.h5", "w") do fid
        # ByteArray style does not support setindex!
        arr = create_hdf5_vector(fid["/"], "bytes", MySerializingType; portable=true)
        push!(arr, MySerializingType("x", [1.], MyType(1,2)))
        @test_throws ErrorException setindex!(arr, MySerializingType("y",[2.],MyType(3,4)), 1)
    end
end
