using Test
using HDF5Vectors
using HDF5
using EnumX
using StaticArrays
import JSON3

# Exercise the public vector operations that every nonempty HDF5 storage representation is
# expected to provide. The testsets below supply collections from each documented type
# category, so adding an operation here checks it across all applicable backends.
function test_collection(
    fid, name, source::Vector{T};
    chunk_length = 5, # Using a small number to make sure we need multiple chunks
    native = false,
    create_kwargs = (;),
) where {T}

    # Create the HDF5Vector from the source.
    arr = create_hdf5_vector(fid["/"], name, T; chunk_length, create_kwargs...)
    for el in source
        push!(arr, el)
    end

    # Test the many little array functions.
    @test length(arr) == length(source)
    @test size(arr) == (length(source),)
    @test IndexStyle(typeof(arr)) == IndexLinear()
    @test arr[1] == source[1] # Check indexing.
    @test arr[end] == source[end] # Check end (lastindex).
    @test arr[:] == source
    idxs = 1:min(3, length(source))
    @test arr[idxs] == source[idxs]
    @test arr[collect(idxs)] == source[collect(idxs)]

    # Boolean vectors and BitVectors are logical masks, not collections of integer indices.
    mask = [isodd(k) for k in eachindex(source)]
    @test arr[mask] == source[mask]
    @test arr[BitVector(mask)] == source[BitVector(mask)]
    @test arr[falses(length(source))] == source[falses(length(source))]
    @test_throws BoundsError arr[vcat(mask, false)]

    # Exercise operations directly on the HDF5 vector. These methods are part of the
    # documented AbstractVector interface and must not rely on callers remembering to use
    # iterable first.
    @test [el for el in arr] == source
    @test map(identity, arr) == source
    @test identity.(arr) == source
    @test mapreduce(identity, (a, b) -> b, arr) == source[end]
    if T <: Real
        @test sum(arr) == sum(source)
    end

    # iterable performs one bulk read before iteration. It should produce the same values
    # and support the same ordinary Julia operations as the HDF5 vector itself.
    @test collect(arr) == source
    @test eltype(arr) == T
    @test map(identity, iterable(arr)) == source
    @test mapreduce(identity, (a, b) -> b, iterable(arr)) == source[end]
    if T <: Real
        @test sum(iterable(arr)) == sum(source) # Check mapreduce.
    end

    # In-place replacement is optional because some storage styles are append-only. Every
    # style that advertises replacement support should be able to replace and then restore
    # a value through the normal AbstractVector interface.
    if HDF5Vectors.supports_setindex(arr)
        arr[1] = source[end]
        @test arr[1] == source[end]
        arr[1] = source[1]
        @test collect(arr) == source
    end

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

    # Load again, this time specifying the element type explicitly. The storage options
    # should still come from the metadata rather than needing to be repeated by the caller.
    arr4 = load_hdf5_vector(fid[name], T)
    @test collect(arr4) == source

    # Now test that we can continue writing to the HDF5 vector.
    for el in source
        push!(arr3, el)
    end
    @test collect(arr3) == vcat(source, source)

end

# Here are some custom things we can work with.
@enum Birds sparrow hawk sparrowhawk
@enum UInt8Values::UInt8 uint8_zero = 0 uint8_max = 255
@enum Int64Values::Int64 int64_low = -3_000_000_000 int64_high = 3_000_000_000
@enum Int128Values::Int128 int128_value = 1
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

struct MyFallibleElementalType
    value::Int64
end
function HDF5Vectors.storage_style(::Type{MyFallibleElementalType}; kwargs...)
    return HDF5Vectors.ElementalStorageStyle(Int64)
end
function HDF5Vectors.construct(
    ::Type{HDF5Vectors.HDF5VectorOfElementalTypes{MyFallibleElementalType, Int64}},
    value::Int64,
)
    return MyFallibleElementalType(value)
end
function HDF5Vectors.deconstruct(
    ::Type{HDF5Vectors.HDF5VectorOfElementalTypes{MyFallibleElementalType, Int64}},
    el::MyFallibleElementalType,
)

    if el.value < 0
        throw(ArgumentError("MyFallibleElementalType values must be nonnegative."))
    end
    return el.value

end

# This type deliberately stores a value that differs from its Julia field. It lets the
# tests distinguish write paths that honor the composite deconstruct interface from paths
# that incorrectly read the field with getproperty.
struct MyOffsetCompositeType
    value::Int64
end
function HDF5Vectors.construct(
    ::Type{HDF5Vectors.HDF5VectorOfCompositeTypes{MyOffsetCompositeType}},
    values,
)
    return MyOffsetCompositeType(values[1] - 100)
end
function HDF5Vectors.deconstruct(
    ::Type{HDF5Vectors.HDF5VectorOfCompositeTypes{MyOffsetCompositeType}},
    el::MyOffsetCompositeType,
)
    return (el.value + 100,)
end

struct MyNoncreteType
    x::NamedTuple
end

# A directly nonconcrete element type selects byte-array storage because its complete field
# layout is unknown. Two concrete subtypes make sure one HDF5 vector can round-trip different
# runtime types without narrowing its declared element type.
abstract type MyAbstractSerializableType end

struct MySerializableInteger <: MyAbstractSerializableType
    value::Int64
end

struct MySerializableString <: MyAbstractSerializableType
    value::String
end

struct MySingletonType
end

struct MySingletonWithoutZeroArgumentConstructor
    MySingletonWithoutZeroArgumentConstructor(::Nothing) = new()
end

# These types reproduce a downstream use case in which a heterogeneous tuple has exactly
# one possible value, but the tuple type itself has no zero-argument constructor. Its two
# fields can still be stored and reconstructed through composite storage.
struct MyParametricSingleton1{Whatever}
end

struct MyParametricSingleton2{Whatever}
end

mutable struct MyMutableZeroFieldType
end

out_dir = "out"
mkpath("out")

@testset "elemental types (portable = $portable)" for portable in (true, false)

    h5open("$out_dir/elemental_types.h5", "w") do fid

        # The documentation promises every HDF5-native signed and unsigned integer width.
        # Boundary values make sure no implementation accidentally narrows a declared type.
        integer_types = (
            Int8,
            UInt8,
            Int16,
            UInt16,
            Int32,
            UInt32,
            Int64,
            UInt64,
        )
        for integer_type in integer_types
            source = integer_type[
                typemin(integer_type),
                zero(integer_type),
                typemax(integer_type),
            ]
            name = lowercase(string(integer_type)) * "s"
            test_collection(fid, name, source; native = true)
        end

        # Float32 and Float64 likewise have distinct native HDF5 representations.
        for float_type in (Float32, Float64)
            source = float_type[-1.5, 0, 2.25]
            name = lowercase(string(float_type)) * "s"
            test_collection(fid, name, source; native = true)
        end

        test_collection(fid, "enums", [sparrowhawk, hawk, sparrow])
        uint8_enums = [uint8_max, uint8_zero, uint8_max]
        int64_enums = [int64_low, int64_high, int64_low]
        test_collection(fid, "uint8_enums", uint8_enums)
        test_collection(fid, "int64_enums", int64_enums)

        # Enums should use their declared base type in HDF5 rather than always using Int32.
        stored_uint8_enums = read(fid["uint8_enums"]["data"])
        stored_int64_enums = read(fid["int64_enums"]["data"])
        @test eltype(stored_uint8_enums) === UInt8
        @test eltype(stored_int64_enums) === Int64
        @test stored_uint8_enums == UInt8.(vcat(uint8_enums, uint8_enums))
        @test stored_int64_enums == Int64.(vcat(int64_enums, int64_enums))

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

        # Vector and Matrix exercise one and two dimensions, while this case confirms that
        # dynamically sized Array elements also preserve higher-dimensional shapes.
        arrays_of_floats = [fill(Float64(k), 2, 1, 2) for k in 1:6]
        test_collection(
            fid,
            "arrays_of_floats",
            arrays_of_floats;
            create_kwargs = (; dims = (2, 1, 2), ),
        )

        test_collection(fid, "ntuples_of_symbols", [(:a, :b) for k in 1:3])
        test_collection(fid, "svectors_of_symbols", [SVector{2, Symbol}(:a, :b) for k in 1:2])

        # Array-like storage must apply the scalar representation of transformed elemental
        # types such as Char and Enum to every element.
        test_collection(fid, "ntuples_of_chars", [('a', 'b'), ('c', 'd')])
        test_collection(fid, "svectors_of_enums", [
            SVector(uint8_zero, uint8_max),
            SVector(uint8_max, uint8_zero),
        ])
        test_collection(fid, "smatrices_of_chars", [
            SMatrix{2, 2, Char, 4}(('a', 'b', 'c', 'd')),
            SMatrix{2, 2, Char, 4}(('e', 'f', 'g', 'h')),
        ])
        test_collection(fid, "sarrays_of_enums", [
            SArray{Tuple{2, 1, 2}, UInt8Values, 3, 4}(
                (uint8_zero, uint8_max, uint8_max, uint8_zero),
            ),
            SArray{Tuple{2, 1, 2}, UInt8Values, 3, 4}(
                (uint8_max, uint8_zero, uint8_zero, uint8_max),
            ),
        ])
        test_collection(
            fid,
            "vectors_of_chars",
            [collect("ab"), collect("cd")];
            create_kwargs = (; dims = (2,), ),
        )
        test_collection(
            fid,
            "matrices_of_enums",
            [
                reshape(
                    UInt8Values[uint8_zero, uint8_max, uint8_max, uint8_zero],
                    2,
                    2,
                ),
                reshape(
                    UInt8Values[uint8_max, uint8_zero, uint8_zero, uint8_max],
                    2,
                    2,
                ),
            ];
            create_kwargs = (; dims = (2, 2), ),
        )

        @test eltype(read(fid["ntuples_of_chars"]["data"])) === Int32
        @test eltype(read(fid["svectors_of_enums"]["data"])) === UInt8
        @test eltype(read(fid["smatrices_of_chars"]["data"])) === Int32
        @test eltype(read(fid["sarrays_of_enums"]["data"])) === UInt8
        @test eltype(read(fid["vectors_of_chars"]["data"])) === Int32
        @test eltype(read(fid["matrices_of_enums"]["data"])) === UInt8

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
            Int128Values,
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

        # A field-bearing singleton that lacks a zero-argument constructor cannot use
        # count-only singleton storage. Composite storage can reconstruct it from its
        # singleton fields, and should do so regardless of the portability option.
        singleton_tuple = (MyParametricSingleton1{:a}(), MyParametricSingleton2{:b}())
        singleton_tuple_type = typeof(singleton_tuple)
        @test Base.issingletontype(singleton_tuple_type)
        @test !applicable(singleton_tuple_type)
        selected_style = HDF5Vectors.storage_style(singleton_tuple_type; portable)
        @test selected_style isa HDF5Vectors.CompositeStorageStyle
        test_collection(
            fid,
            "field_bearing_singleton_tuple",
            fill(singleton_tuple, 3);
            create_kwargs,
        )

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

@testset "creation option validation" begin

    h5open("$out_dir/creation_option_validation.h5", "w") do fid

        # Invalid chunk lengths should fail before creating an HDF5 group.
        @test_throws ArgumentError create_hdf5_vector(
            fid["/"], "zero_chunk", Int64; chunk_length = 0,
        )
        @test_throws ArgumentError create_hdf5_vector(
            fid["/"], "float_chunk", Int64; chunk_length = 2.5,
        )
        @test !haskey(fid, "zero_chunk")
        @test !haskey(fid, "float_chunk")

        # Dynamic arrays require positive integer dimensions with the correct rank.
        @test_throws DimensionMismatch create_hdf5_vector(
            fid["/"],
            "wrong_rank",
            Vector{Float64};
            dims = (2, 3),
        )
        @test_throws ArgumentError create_hdf5_vector(
            fid["/"],
            "float_dims",
            Vector{Float64};
            dims = (2.0,),
        )
        @test_throws ArgumentError create_hdf5_vector(
            fid["/"],
            "zero_dims",
            Vector{Float64};
            dims = (0,),
        )

        # Dimensions supplied for statically sized arrays must match their type.
        @test_throws DimensionMismatch create_hdf5_vector(
            fid["/"],
            "wrong_static_dims",
            SVector{2, Float64};
            dims = (3,),
        )

    end

end

@testset "destination validation" begin

    h5open("$out_dir/destination_validation.h5", "w") do fid

        # The destination must be an HDF5 group, including when writing at the file root.
        @test_throws MethodError create_hdf5_vector(fid, "file_parent", Int64)
        @test_throws MethodError copy_to_hdf5_vector(fid, "file_parent", Int64[1])
        @test !haskey(fid, "file_parent")

        # Destination names must be strings regardless of the selected storage style.
        cases = (
            (:elemental_name, Int64, Int64[1]),
            (:composite_name, MyType, [MyType(1, 2.0)]),
            (
                :serialized_name,
                MySerializingType,
                [MySerializingType("value", [1.0], MyType(2, 3.0))],
            ),
        )
        for (name, el_type, source) in cases
            @test_throws MethodError create_hdf5_vector(fid["/"], name, el_type)
            @test_throws MethodError copy_to_hdf5_vector(fid["/"], name, source)
            @test !haskey(fid, string(name))
        end

        # The contract accepts AbstractString implementations rather than only String.
        full_name = "substring_name_suffix"
        name = SubString(full_name, 1, 14)
        copied = copy_to_hdf5_vector(fid["/"], name, Int64[1, 2])
        @test collect(copied) == [1, 2]

    end

end

@testset "array-like element validation" begin

    h5open("$out_dir/arrayish_element_validation.h5", "w") do fid

        arr = create_hdf5_vector(fid["/"], "vectors", Vector{Float64}; dims = (2,))
        original = [1.0, 2.0]
        push!(arr, original)

        # Shape errors must be reported before a push extends the dataset or setindex!
        # replaces the existing value.
        @test_throws DimensionMismatch push!(arr, [3.0, 4.0, 5.0])
        @test length(arr) == 1
        @test size(arr.dataset) == (2, 1)
        @test_throws DimensionMismatch setindex!(arr, [3.0, 4.0, 5.0], 1)
        @test collect(arr) == [original]

        replacement = [6.0, 7.0]
        arr[1] = replacement
        @test collect(arr) == [replacement]
        @test_throws HDF5.API.H5Error setindex!(arr, replacement, 2)
        @test collect(arr) == [replacement]

    end

end

@testset "bulk copy preflight" begin

    h5open("$out_dir/bulk_copy_preflight.h5", "w") do fid

        # Elemental conversion must finish before the optimized copy creates its group.
        # This custom elemental representation deliberately rejects one source value.
        elemental_source = [
            MyFallibleElementalType(1),
            MyFallibleElementalType(-1),
        ]
        @test_throws ArgumentError copy_to_hdf5_vector(
            fid["/"],
            "invalid_elemental",
            elemental_source,
        )
        @test !haskey(fid, "invalid_elemental")

        # Fixed-shape array storage must likewise validate every element before creating
        # the destination. The second vector does not have the declared dimensions.
        array_source = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        @test_throws DimensionMismatch copy_to_hdf5_vector(
            fid["/"],
            "invalid_array",
            array_source;
            dims = (2,),
        )
        @test !haskey(fid, "invalid_array")

        # A shape mismatch that Julia could broadcast must still be rejected rather than
        # silently expanded to the declared dimensions.
        broadcastable_array_source = [reshape([1.0, 2.0], 2, 1)]
        @test_throws DimensionMismatch copy_to_hdf5_vector(
            fid["/"],
            "broadcastable_invalid_array",
            broadcastable_array_source;
            dims = (2, 2),
        )
        @test !haskey(fid, "broadcastable_invalid_array")

    end

end

@testset "multi-dataset load validation" begin

    h5open("$out_dir/multi_dataset_load_validation.h5", "w") do fid

        # Every field of a composite vector must contain the same number of values.
        composite_source = [MyType(1, 2.0), MyType(3, 4.0)]
        copy_to_hdf5_vector(fid["/"], "invalid_composite", composite_source)
        second_field_dataset = fid["invalid_composite"]["data"]["b"]["data"]
        HDF5.set_extent_dims(second_field_dataset, (1,))
        @test_throws DimensionMismatch load_hdf5_vector(fid["invalid_composite"])

        # The final serialized stop must coincide with the end of the byte dataset.
        serialized_source = [
            MySerializingType("first", [1.0], MyType(2, 3.0)),
        ]
        copy_to_hdf5_vector(fid["/"], "invalid_serialized", serialized_source)
        serialized_bytes = fid["invalid_serialized"]["data"]["bytes"]["data"]
        HDF5.set_extent_dims(serialized_bytes, (length(serialized_bytes) + 1,))
        serialized_bytes[end] = 0x00
        @test_throws DimensionMismatch load_hdf5_vector(fid["invalid_serialized"])

        # Empty serialized storage must not contain bytes without a corresponding stop.
        copy_to_hdf5_vector(
            fid["/"],
            "invalid_empty_serialized",
            MySerializingType[],
        )
        empty_bytes = fid["invalid_empty_serialized"]["data"]["bytes"]["data"]
        HDF5.set_extent_dims(empty_bytes, (1,))
        empty_bytes[1] = 0x00
        @test_throws DimensionMismatch load_hdf5_vector(fid["invalid_empty_serialized"])

    end

end

@testset "custom composite deconstruction" begin

    h5open("$out_dir/custom_composite_deconstruction.h5", "w") do fid

        # push!, setindex!, and copy_to_hdf5_vector are three ways to write a composite
        # value. They must all pass through the same deconstruct hook so custom stored
        # representations do not depend on which public operation the caller chooses.
        original = MyOffsetCompositeType(1)
        replacement = MyOffsetCompositeType(2)

        values = create_hdf5_vector(fid["/"], "values", MyOffsetCompositeType)
        push!(values, original)
        @test values[1] == original
        @test read(fid["values/data/value/data"]) == [101]

        # Replacement must apply the transformation too. Reading the logical value alone
        # is not sufficient evidence because construct performs the inverse transformation;
        # checking the raw child dataset confirms which representation was actually stored.
        values[1] = replacement
        @test values[1] == replacement
        @test read(fid["values/data/value/data"]) == [102]

        # Bulk copy prepares a typed collection for each field. It should obtain those field
        # values from deconstruct rather than bypassing customization with getproperty.
        source = [original, replacement]
        copied = copy_to_hdf5_vector(fid["/"], "copied_values", source)
        @test collect(copied) == source
        @test read(fid["copied_values/data/value/data"]) == [101, 102]
        @test collect(load_hdf5_vector(fid["copied_values"])) == source

    end

end

@testset "setindex! storage support" begin

    h5open("$out_dir/setindex_support.h5", "w") do fid

        # A composite with entirely in-place child storage supports replacement.
        elemental_composite = create_hdf5_vector(fid["/"], "elemental_composite", MyType)
        push!(elemental_composite, MyType(1, 2.0))
        elemental_composite[1] = MyType(3, 4.0)
        @test collect(elemental_composite) == [MyType(3, 4.0)]

        # This composite's vector field uses append-only byte storage. Reject replacement
        # before changing its earlier string field.
        append_only_composite = create_hdf5_vector(
            fid["/"],
            "append_only_composite",
            MyNonBitsType,
        )
        original = MyNonBitsType("original", [1, 2])
        replacement = MyNonBitsType("replacement", [3, 4])
        push!(append_only_composite, original)
        @test_throws ArgumentError setindex!(append_only_composite, replacement, 1)
        @test collect(append_only_composite) == [original]

    end

end

@testset "empty bulk copies" begin

    h5open("$out_dir/empty_bulk_copies.h5", "w") do fid

        # An empty source still carries an element type, which is enough to select and create
        # each storage representation. Loading must recover an empty vector without relying
        # on a first value from which to infer any part of the layout.
        cases = (
            ("elemental", Int64[], (;)),
            ("array", Vector{Float64}[], (; dims = (2,), )),
            ("composite", MyType[], (;)),
            ("singleton", Nothing[], (;)),
            ("serialized", MySerializingType[], (;)),
            ("json", MyJSONishType[], (;)),
        )
        for (name, source, create_kwargs) in cases
            copied = copy_to_hdf5_vector(fid["/"], name, source; create_kwargs...)
            reloaded = load_hdf5_vector(fid[name])
            @test isempty(copied)
            @test isempty(reloaded)
            @test eltype(copied) == eltype(source)
            @test eltype(reloaded) == eltype(source)
        end

        # Empty byte-array storage is represented by two empty child datasets. Appending
        # after reloading confirms that zero-byte initial storage remains extensible.
        serialized_data_group = fid["serialized/data"]
        @test isempty(read(serialized_data_group["bytes/data"]))
        @test isempty(read(serialized_data_group["stops/data"]))
        reloaded_serialized = load_hdf5_vector(fid["serialized"])
        first_value = MySerializingType("first", [1.0], MyType(2, 3.0))
        push!(reloaded_serialized, first_value)
        @test collect(reloaded_serialized) == [first_value]

    end

end

@testset "serialization types" begin

    h5open("$out_dir/serialization_types.h5", "w") do fid

        # This first type is concrete, but one of its fields is nonconcrete. It exercises
        # recursive fallback to byte-array storage for that field.
        test_collection(fid, "nonconcrete_types", [MyNoncreteType((; k, )) for k in 1:10])

        # Here the vector's declared element type is itself abstract. The byte-array backend
        # must preserve the concrete runtime type of each independently serialized value.
        abstract_serializing_types = MyAbstractSerializableType[
            MySerializableInteger(1),
            MySerializableString("two"),
            MySerializableInteger(3),
        ]
        test_collection(fid, "abstract_serializing_types", abstract_serializing_types)

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

        # The optimized serialization copy stores all serialized values in one byte vector
        # and records the cumulative end position of each value.
        serialized_values = HDF5Vectors.serialize_to_byte_array.(serializing_types)
        expected_bytes = reduce(vcat, serialized_values)
        expected_stops = cumsum(Int64[length(bytes) for bytes in serialized_values])
        serialized_data_group = fid["serializing_types-copy"]["data"]
        @test read(serialized_data_group["bytes"]["data"]) == expected_bytes
        @test read(serialized_data_group["stops"]["data"]) == expected_stops

    end

end

@testset "error conditions" begin

    h5open("$out_dir/error_conditions.h5", "w") do fid

        # ByteArray style does not support setindex!
        arr = create_hdf5_vector(fid["/"], "bytes", MySerializingType; portable = true)
        push!(arr, MySerializingType("x", [1.0], MyType(1, 2)))
        replacement = MySerializingType("y", [2.0], MyType(3, 4))
        @test_throws ErrorException setindex!(arr, replacement, 1)

    end

end
