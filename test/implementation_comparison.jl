@enum ParityUInt8Enum::UInt8 parity_zero = 0 parity_max = 255
@enum ParityInt64Enum::Int64 parity_low = -3_000_000_000 parity_high = 3_000_000_000
@enum ParityInt128Enum::Int128 parity_int128 = 1
EnumX.@enumx ParityUngulates parity_deer parity_horse parity_bison

struct ParityPoint
    x::Float64
    y::Int64
end

struct ParityNestedRecord
    direction::StaticArrays.SVector{3, Float64}
    point::ParityPoint
end

struct ParityNonBitsRecord
    label::String
    values::Vector{Int64}
end

function Base.:(==)(first::ParityNonBitsRecord, second::ParityNonBitsRecord)
    return first.label == second.label && first.values == second.values
end

struct ParityNonconcreteField
    value::NamedTuple
end

abstract type ParityAbstractValue end

struct ParityIntegerValue <: ParityAbstractValue
    value::Int64
end

struct ParityStringValue <: ParityAbstractValue
    value::String
end

struct ParitySingleton
end

struct ParityUnconstructibleSingleton
    ParityUnconstructibleSingleton(::Nothing) = new()
end

struct ParitySingleton1{Value}
end

struct ParitySingleton2{Value}
end

mutable struct ParityMutableZeroField
end

function compare_hdf5_vector_implementations(
    old_group,
    new_group,
    name,
    source::Vector{T};
    kwargs...,
) where {T}

    old_vector = HDF5Vectors.copy_baseline_to_hdf5_vector(
        old_group,
        name,
        source;
        chunk_length = 2,
        kwargs...,
    )
    new_vector = HDF5Vectors2.copy_to_hdf5_vector(
        new_group,
        name,
        source;
        chunk_length = 2,
        kwargs...,
    )

    # Both implementations should expose the same logical AbstractVector behavior even
    # though their schemas and physical HDF5 layouts differ.
    @test eltype(old_vector) === T
    @test eltype(new_vector) === T
    @test collect(old_vector) == source
    @test collect(new_vector) == source
    @test old_vector[:] == new_vector[:]
    @test old_vector[1] == new_vector[1] == first(source)
    indices = 1:min(2, length(source))
    @test old_vector[indices] == new_vector[indices] == source[indices]
    @test old_vector[collect(indices)] == new_vector[collect(indices)] == source[indices]
    mask = [isodd(index) for index in eachindex(source)]
    @test old_vector[mask] == new_vector[mask] == source[mask]
    @test old_vector[BitVector(mask)] == new_vector[BitVector(mask)] == source[mask]
    @test [value for value in old_vector] == [value for value in new_vector] == source
    @test map(identity, old_vector) == map(identity, new_vector) == source
    @test identity.(old_vector) == identity.(new_vector) == source

    # Loading without an explicit Julia type must recover the declared element type and
    # permit the same subsequent append in both implementations.
    old_loaded = HDF5Vectors.load_baseline_hdf5_vector(old_group[name])
    new_loaded = HDF5Vectors2.load_hdf5_vector(new_group[name])
    old_typed = HDF5Vectors.load_baseline_hdf5_vector(old_group[name], T)
    new_typed = HDF5Vectors2.load_hdf5_vector(new_group[name], T)
    @test eltype(old_loaded) === T
    @test eltype(new_loaded) === T
    @test collect(old_typed) == source
    @test collect(new_typed) == source
    push!(old_loaded, first(source))
    push!(new_loaded, first(source))
    expected = copy(source)
    push!(expected, first(source))
    @test collect(old_loaded) == expected
    @test collect(new_loaded) == expected

end

function compare_elemental_types(old_group, new_group; portable)

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
        compare_hdf5_vector_implementations(
            old_group,
            new_group,
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
        ("uint8_enum", [parity_zero, parity_max, parity_zero]),
        ("int64_enum", [parity_low, parity_high, parity_low]),
        (
            "enumx",
            [ParityUngulates.parity_deer, ParityUngulates.parity_horse],
        ),
    )
    for (name, source) in cases
        compare_hdf5_vector_implementations(
            old_group,
            new_group,
            name,
            source;
            portable,
        )
    end

end

function compare_array_types(old_group, new_group; portable)

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
                StaticArrays.SVector(parity_zero, parity_max),
                StaticArrays.SVector(parity_max, parity_zero),
            ],
        ),
        (
            "symbol_svectors",
            [StaticArrays.SVector(:a, :b), StaticArrays.SVector(:c, :d)],
        ),
    )
    for (name, source) in fixed_cases
        compare_hdf5_vector_implementations(
            old_group,
            new_group,
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
        compare_hdf5_vector_implementations(
            old_group,
            new_group,
            name,
            source;
            portable,
            options...,
        )
    end

end

function compare_constant_types(old_group, new_group; portable)

    singleton_tuple = (ParitySingleton1{:a}(), ParitySingleton2{:b}())
    cases = (
        ("empty_tuples", [(), ()]),
        ("empty_named_tuples", [(;), (;)]),
        ("nothings", Nothing[nothing, nothing]),
        ("vals", [Val(:ready), Val(:ready)]),
        ("singletons", [ParitySingleton(), ParitySingleton()]),
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
        compare_hdf5_vector_implementations(
            old_group,
            new_group,
            name,
            source;
            portable,
        )
    end

end

function compare_record_types(old_group, new_group; portable)

    singleton_tuple = (ParitySingleton1{:a}(), ParitySingleton2{:b}())
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
        ("points", [ParityPoint(1.0, 2), ParityPoint(3.0, 4)]),
        (
            "nested_records",
            [
                ParityNestedRecord(
                    StaticArrays.SVector(1.0, 2.0, 3.0),
                    ParityPoint(4.0, 5),
                ),
                ParityNestedRecord(
                    StaticArrays.SVector(6.0, 7.0, 8.0),
                    ParityPoint(9.0, 10),
                ),
            ],
        ),
        (
            "nonbits_records",
            [
                ParityNonBitsRecord("first", [1, 2]),
                ParityNonBitsRecord("second", [3]),
            ],
        ),
        ("singleton_tuple_records", fill(singleton_tuple, 2)),
        (
            "static_record_arrays",
            [
                StaticArrays.SVector(ParityPoint(1.0, 2), ParityPoint(3.0, 4)),
                StaticArrays.SVector(ParityPoint(5.0, 6), ParityPoint(7.0, 8)),
            ],
        ),
        (
            "nonconcrete_fields",
            [ParityNonconcreteField((a = 1,)), ParityNonconcreteField((b = "two",))],
        ),
        (
            "abstract_values",
            ParityAbstractValue[ParityIntegerValue(1), ParityStringValue("two")],
        ),
    )
    for (name, source) in cases
        compare_hdf5_vector_implementations(
            old_group,
            new_group,
            name,
            source;
            portable,
        )
    end

end

@testset "HDF5Vectors implementation comparison" begin

    mktempdir() do directory

        for portable in (true, false)

            path = joinpath(directory, "comparison_$portable.h5")
            HDF5.h5open(path, "w") do file

                old_group = HDF5.create_group(file, "old")
                new_group = HDF5.create_group(file, "new")
                compare_elemental_types(old_group, new_group; portable)
                compare_array_types(old_group, new_group; portable)
                compare_constant_types(old_group, new_group; portable)
                compare_record_types(old_group, new_group; portable)

            end

        end

    end

end

@testset "HDF5Vectors shared unsupported types" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "unsupported_comparison.h5"), "w") do file

            old_group = HDF5.create_group(file, "old")
            new_group = HDF5.create_group(file, "new")
            unsupported_types = (
                Float16,
                Int128,
                UInt128,
                ParityInt128Enum,
                ParityUnconstructibleSingleton,
                ParityMutableZeroField,
            )
            for (index, type) in enumerate(unsupported_types)
                name = "unsupported_$index"
                @test_throws ArgumentError HDF5Vectors.create_baseline_hdf5_vector(
                    old_group,
                    name,
                    type,
                )
                @test_throws ArgumentError HDF5Vectors2.create_hdf5_vector(
                    new_group,
                    name,
                    type,
                )
                @test !haskey(old_group, name)
                @test !haskey(new_group, name)
            end

        end

    end

end
