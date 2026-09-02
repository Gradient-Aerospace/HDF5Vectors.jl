module SupportedTypesAndOperationsTests

include("_test_setup.jl")

@enum SupportedUInt8Enum::UInt8 supported_zero = 0 supported_max = 255
@enum SupportedInt64Enum::Int64 supported_low = -3_000_000_000 supported_high = 3_000_000_000
@enum SupportedInt128Enum::Int128 supported_int128 = 1
EnumX.@enumx SupportedUngulates supported_deer supported_horse supported_bison

struct SupportedPoint
    x::Float64
    y::Int64
end

struct SupportedNestedRecord
    direction::StaticArrays.SVector{3, Float64}
    point::SupportedPoint
end

struct SupportedNonBitsRecord
    label::String
    values::Vector{Int64}
end

function Base.:(==)(first::SupportedNonBitsRecord, second::SupportedNonBitsRecord)
    return first.label == second.label && first.values == second.values
end

struct SupportedNonconcreteField
    value::NamedTuple
end

abstract type SupportedAbstractValue end

struct SupportedIntegerValue <: SupportedAbstractValue
    value::Int64
end

struct SupportedStringValue <: SupportedAbstractValue
    value::String
end

struct SupportedSingleton
end

struct SupportedUnconstructibleSingleton
    SupportedUnconstructibleSingleton(::Nothing) = new()
end

struct SupportedSingleton1{Value}
end

struct SupportedSingleton2{Value}
end

mutable struct SupportedMutableZeroField
end

function test_supported_collection(
    group,
    name,
    source::Vector{T};
    kwargs...,
) where {T}

    vector = HDF5Vectors.copy_to_hdf5_vector(
        group,
        name,
        source;
        chunk_length = 2,
        kwargs...,
    )

    # Every supported representation should expose the same logical AbstractVector
    # behavior through the package's ordinary bulk-copy entry point.
    @test eltype(vector) === T
    @test collect(vector) == source
    @test vector[:] == source
    @test vector[1] == first(source)
    indices = 1:min(2, length(source))
    @test vector[indices] == source[indices]
    @test vector[collect(indices)] == source[indices]
    mask = [isodd(index) for index in eachindex(source)]
    @test vector[mask] == source[mask]
    @test vector[BitVector(mask)] == source[mask]
    @test [value for value in vector] == source
    @test map(identity, vector) == source
    @test identity.(vector) == source

    # Loading without an explicit Julia type must recover the declared element type and
    # permit a subsequent append. Typed loading should recover the same values without
    # deserializing inferred schema metadata.
    loaded = HDF5Vectors.load_hdf5_vector(group[name])
    typed = HDF5Vectors.load_hdf5_vector(group[name], T)
    @test eltype(loaded) === T
    @test collect(typed) == source
    push!(loaded, first(source))
    expected = copy(source)
    push!(expected, first(source))
    @test collect(loaded) == expected

end

function test_scalar_types(group; portable)

    # Boundary values exercise every documented native integer width without permitting an
    # implementation to narrow the physical representation silently.
    for type in (
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Int64,
        UInt64,
    )
        source = type[typemin(type), zero(type), typemax(type)]
        test_supported_collection(
            group,
            lowercase(string(type)),
            source;
            portable,
        )
    end

    cases = (
        ("float32", Float32[-1.5, 0, 2.25]),
        ("float64", Float64[-1.5, 0, 2.25]),
        ("bool", Bool[true, false, true]),
        ("char", ['a', 'λ', 'z']),
        ("string", ["first", "second", "third"]),
        ("symbol", [:first, :second, :third]),
        ("uint8_enum", [supported_zero, supported_max, supported_zero]),
        ("int64_enum", [supported_low, supported_high, supported_low]),
        (
            "enumx",
            [SupportedUngulates.supported_deer, SupportedUngulates.supported_horse],
        ),
    )
    for (name, source) in cases
        test_supported_collection(
            group,
            name,
            source;
            portable,
        )
    end

end

function test_array_types(group; portable)

    # Fixed-size tuples and static arrays infer their dimensions from their types. These
    # cases cover ranks one through three and transformed scalar element codecs.
    fixed_cases = (
        ("int_tuples", [(Int64(1), Int64(2)), (Int64(3), Int64(4))]),
        (
            "float_svectors",
            [StaticArrays.SVector(1.0, 2.0, 3.0), StaticArrays.SVector(4.0, 5.0, 6.0)],
        ),
        (
            "float_smatrices",
            [
                StaticArrays.SMatrix{2, 2, Float64, 4}((1.0, 2.0, 3.0, 4.0)),
                StaticArrays.SMatrix{2, 2, Float64, 4}((5.0, 6.0, 7.0, 8.0)),
            ],
        ),
        (
            "float_sarrays",
            [
                StaticArrays.SArray{Tuple{2, 1, 2}, Float64, 3, 4}(
                    (1.0, 2.0, 3.0, 4.0),
                ),
                StaticArrays.SArray{Tuple{2, 1, 2}, Float64, 3, 4}(
                    (5.0, 6.0, 7.0, 8.0),
                ),
            ],
        ),
        ("char_tuples", [('a', 'b'), ('c', 'd')]),
        (
            "enum_svectors",
            [
                StaticArrays.SVector(supported_zero, supported_max),
                StaticArrays.SVector(supported_max, supported_zero),
            ],
        ),
        (
            "symbol_svectors",
            [StaticArrays.SVector(:a, :b), StaticArrays.SVector(:c, :d)],
        ),
    )
    for (name, source) in fixed_cases
        test_supported_collection(
            group,
            name,
            source;
            portable,
        )
    end

    # Dynamic Arrays use dense storage only when dimensions are declared; omitting them
    # compares the serialized fallback instead.
    dynamic_cases = (
        ("dense_vectors", [[1.0, 2.0], [3.0, 4.0]], (; dims = (2,))),
        (
            "dense_matrices",
            [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]],
            (; dims = (2, 2)),
        ),
        (
            "dense_arrays",
            [fill(1.0, 2, 1, 2), fill(2.0, 2, 1, 2)],
            (; dims = (2, 1, 2)),
        ),
        ("dense_chars", [collect("ab"), collect("cd")], (; dims = (2,))),
        ("serialized_vectors", [[1.0], [2.0, 3.0]], (;)),
        (
            "serialized_matrices",
            [reshape([1.0, 2.0], 1, 2), reshape([3.0, 4.0], 2, 1)],
            (;),
        ),
    )
    for (name, source, options) in dynamic_cases
        test_supported_collection(
            group,
            name,
            source;
            portable,
            options...,
        )
    end

end

function test_constant_types(group; portable)

    singleton_tuple = (SupportedSingleton1{:a}(), SupportedSingleton2{:b}())
    cases = (
        ("empty_tuples", [(), ()]),
        ("empty_named_tuples", [(;), (;)]),
        ("nothings", Nothing[nothing, nothing]),
        ("vals", [Val(:ready), Val(:ready)]),
        ("singletons", [SupportedSingleton(), SupportedSingleton()]),
        ("singleton_records", fill(singleton_tuple, 2)),
        (
            "empty_svectors",
            [
                StaticArrays.SVector{0, Float64}(),
                StaticArrays.SVector{0, Float64}(),
            ],
        ),
        (
            "empty_smatrices",
            [
                StaticArrays.SMatrix{0, 2, Float64, 0}(),
                StaticArrays.SMatrix{0, 2, Float64, 0}(),
            ],
        ),
        (
            "empty_sarrays",
            [
                StaticArrays.SArray{Tuple{0, 2, 1}, Float64, 3, 0}(),
                StaticArrays.SArray{Tuple{0, 2, 1}, Float64, 3, 0}(),
            ],
        ),
    )
    for (name, source) in cases
        test_supported_collection(
            group,
            name,
            source;
            portable,
        )
    end

end

function test_record_types(group; portable)

    singleton_tuple = (SupportedSingleton1{:a}(), SupportedSingleton2{:b}())
    cases = (
        ("complex", ComplexF64[1 + 2im, 3 + 4im]),
        ("rationals", Rational{Int64}[1 // 2, 3 // 4]),
        ("heterogeneous_tuples", [(1.0, Int64(2)), (3.0, Int64(4))]),
        ("one_tuples", [(1.0,), (2.0,)]),
        ("named_tuples", [(a = 1.0, b = Int64(2)), (a = 3.0, b = Int64(4))]),
        (
            "nested_tuples",
            [(1.0, (a = 2.0, b = Int64(3))), (4.0, (a = 5.0, b = Int64(6)))],
        ),
        (
            "tuples_with_strings",
            [("first", (a = 1.0, b = Int64(2))), ("second", (a = 3.0, b = Int64(4)))],
        ),
        ("points", [SupportedPoint(1.0, 2), SupportedPoint(3.0, 4)]),
        (
            "nested_records",
            [
                SupportedNestedRecord(
                    StaticArrays.SVector(1.0, 2.0, 3.0),
                    SupportedPoint(4.0, 5),
                ),
                SupportedNestedRecord(
                    StaticArrays.SVector(6.0, 7.0, 8.0),
                    SupportedPoint(9.0, 10),
                ),
            ],
        ),
        (
            "nonbits_records",
            [
                SupportedNonBitsRecord("first", [1, 2]),
                SupportedNonBitsRecord("second", [3]),
            ],
        ),
        ("singleton_tuple_records", fill(singleton_tuple, 2)),
        (
            "static_record_arrays",
            [
                StaticArrays.SVector(SupportedPoint(1.0, 2), SupportedPoint(3.0, 4)),
                StaticArrays.SVector(SupportedPoint(5.0, 6), SupportedPoint(7.0, 8)),
            ],
        ),
        (
            "static_record_matrices",
            [
                StaticArrays.SMatrix{2, 2}(
                    SupportedPoint(1.0, 2),
                    SupportedPoint(3.0, 4),
                    SupportedPoint(5.0, 6),
                    SupportedPoint(7.0, 8),
                ),
                StaticArrays.SMatrix{2, 2}(
                    SupportedPoint(9.0, 10),
                    SupportedPoint(11.0, 12),
                    SupportedPoint(13.0, 14),
                    SupportedPoint(15.0, 16),
                ),
            ],
        ),
        (
            "static_record_arrays_rank_three",
            [
                StaticArrays.SArray{Tuple{2, 1, 2}}(
                    SupportedPoint(1.0, 2),
                    SupportedPoint(3.0, 4),
                    SupportedPoint(5.0, 6),
                    SupportedPoint(7.0, 8),
                ),
                StaticArrays.SArray{Tuple{2, 1, 2}}(
                    SupportedPoint(9.0, 10),
                    SupportedPoint(11.0, 12),
                    SupportedPoint(13.0, 14),
                    SupportedPoint(15.0, 16),
                ),
            ],
        ),
        (
            "nonconcrete_fields",
            [SupportedNonconcreteField((a = 1,)), SupportedNonconcreteField((b = "two",))],
        ),
        (
            "abstract_values",
            SupportedAbstractValue[SupportedIntegerValue(1), SupportedStringValue("two")],
        ),
    )
    for (name, source) in cases
        test_supported_collection(
            group,
            name,
            source;
            portable,
        )
    end

end

@testset "HDF5Vectors supported types and operations" begin

    mktempdir() do directory

        for portable in (true, false)

            path = joinpath(directory, "supported_types_$portable.h5")
            HDF5.h5open(path, "w") do file

                test_scalar_types(file["/"]; portable)
                test_array_types(file["/"]; portable)
                test_constant_types(file["/"]; portable)
                test_record_types(file["/"]; portable)

            end

        end

    end

end

@testset "HDF5Vectors unsupported types" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "unsupported_types.h5"), "w") do file

            unsupported_types = (
                Float16,
                Int128,
                UInt128,
                SupportedInt128Enum,
                SupportedUnconstructibleSingleton,
                SupportedMutableZeroField,
            )
            for (index, type) in enumerate(unsupported_types)
                name = "unsupported_$index"
                @test_throws ArgumentError HDF5Vectors.create_hdf5_vector(
                    file["/"],
                    name,
                    type,
                )
                @test !haskey(file, name)
            end

        end

    end

end

end # module SupportedTypesAndOperationsTests
