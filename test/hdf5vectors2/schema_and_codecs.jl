@enum PrototypeUInt8Enum::UInt8 prototype_zero = 0 prototype_max = 255

struct PrototypePoint
    x::Float64
    y::Int64
end

struct PrototypeSample
    point::PrototypePoint
    label::Symbol
    values::Vector{Float64}
end

function Base.:(==)(first::PrototypeSample, second::PrototypeSample)
    return first.point == second.point &&
        first.label == second.label &&
        first.values == second.values
end

struct PrototypeSingleton1{Value}
end

struct PrototypeSingleton2{Value}
end

struct PrototypeUnconstructibleSingleton
    PrototypeUnconstructibleSingleton(::Nothing) = new()
end

mutable struct PrototypeMutableMarker
end

abstract type PrototypeAbstractValue end

struct PrototypeConcreteValue <: PrototypeAbstractValue
    value::Int64
end

function test_schema_round_trip(schema, value)
    encoded = encode_value(schema, value)
    @test decode_value(schema, encoded) == value
end

@testset "HDF5Vectors2 schema inference" begin

    @testset "scalar schemas" begin

        # Logical Julia types and physical HDF5 types are separate properties of the
        # schema. Transforming codecs make that distinction explicit.
        float_schema = infer_schema(Float64)
        char_schema = infer_schema(Char)
        symbol_schema = infer_schema(Symbol)
        enum_schema = infer_schema(PrototypeUInt8Enum)

        @test float_schema isa ScalarSchema
        @test encoded_type(float_schema) === Float64
        @test encoded_type(char_schema) === Int32
        @test encoded_type(symbol_schema) === String
        @test encoded_type(enum_schema) === UInt8

        test_schema_round_trip(float_schema, 1.25)
        test_schema_round_trip(char_schema, 'λ')
        test_schema_round_trip(symbol_schema, :ready)
        test_schema_round_trip(enum_schema, prototype_max)

    end

    @testset "dense schemas" begin

        # The same scalar codec is reused inside every dense container. Arrays of
        # transformed values therefore cannot accidentally use their logical element type
        # as the HDF5 datatype.
        tuple_schema = infer_schema(NTuple{2, Char})
        static_schema = infer_schema(StaticArrays.SVector{2, PrototypeUInt8Enum})
        array_schema = infer_schema(Vector{Char}; dims = (2,))

        @test tuple_schema isa DenseSchema
        @test tuple_schema.dims == (2,)
        @test encoded_type(tuple_schema) === Int32
        @test encoded_type(static_schema) === UInt8
        @test encoded_type(array_schema) === Int32

        test_schema_round_trip(tuple_schema, ('a', 'b'))
        test_schema_round_trip(
            static_schema,
            StaticArrays.SVector(prototype_zero, prototype_max),
        )
        test_schema_round_trip(array_schema, ['a', 'b'])

        @test_throws DimensionMismatch encode_value(array_schema, ['a'])
        @test_throws DimensionMismatch infer_schema(Vector{Char}; dims = (2, 1))
        @test_throws ArgumentError infer_schema(Vector{Char}; dims = (0,))

    end

    @testset "record schemas" begin

        # Portable records recursively describe their fields. A dynamic array field uses a
        # blob schema by default because its dimensions are not part of its declared type.
        point_schema = infer_schema(PrototypePoint)
        sample_schema = infer_schema(PrototypeSample)

        @test point_schema isa RecordSchema
        @test point_schema.names == ("x", "y")
        @test all(child -> child isa ScalarSchema, point_schema.children)
        @test sample_schema isa RecordSchema
        @test sample_schema.children[1] isa RecordSchema
        @test sample_schema.children[2] isa ScalarSchema
        @test sample_schema.children[3] isa BlobSchema

        point = PrototypePoint(1.5, 2)
        sample = PrototypeSample(point, :sample, [3.0, 4.0])
        test_schema_round_trip(point_schema, point)
        test_schema_round_trip(sample_schema, sample)

        # A non-portable, nonzero-size bits record may use one native HDF5 datatype. This is
        # an inference policy; later storage operations simply execute the selected schema.
        native_point_schema = infer_schema(
            PrototypePoint;
            policy = SchemaPolicy(; portable = false),
        )
        @test native_point_schema isa ScalarSchema
        test_schema_round_trip(native_point_schema, point)

    end

    @testset "constant and field-bearing singleton schemas" begin

        # Constant storage is selected only when the value can be reconstructed directly.
        # Merely satisfying issingletontype is not enough.
        nothing_schema = infer_schema(Nothing)
        marker_schema = infer_schema(PrototypeSingleton1{:marker})

        @test nothing_schema isa ConstantSchema
        @test marker_schema isa ConstantSchema
        test_schema_round_trip(nothing_schema, nothing)
        test_schema_round_trip(marker_schema, PrototypeSingleton1{:marker}())

        # This exact shape reproduces the regression that motivated the prototype. The tuple
        # has one possible value but no zero-argument constructor, so its fields make it a
        # record rather than a constant.
        singleton_tuple = (PrototypeSingleton1{:a}(), PrototypeSingleton2{:b}())
        singleton_tuple_type = typeof(singleton_tuple)
        singleton_tuple_schema = infer_schema(singleton_tuple_type)

        @test Base.issingletontype(singleton_tuple_type)
        @test !applicable(singleton_tuple_type)
        @test singleton_tuple_schema isa RecordSchema
        test_schema_round_trip(singleton_tuple_schema, singleton_tuple)

        @test_throws ArgumentError infer_schema(PrototypeUnconstructibleSingleton)
        @test_throws ArgumentError infer_schema(PrototypeMutableMarker)

    end

    @testset "serialized and unsupported schemas" begin

        # Serialization is represented by an ordinary explicit schema. Inference policies
        # decide when to select it, but the blob codec itself has no fallback behavior.
        array_schema = infer_schema(Vector{Float64})
        abstract_schema = infer_schema(PrototypeAbstractValue)
        explicit_schema = serialization_schema(Dict{Symbol, Int64})

        @test array_schema isa BlobSchema
        @test abstract_schema isa BlobSchema
        @test explicit_schema isa BlobSchema

        test_schema_round_trip(array_schema, [1.0, 2.0])
        test_schema_round_trip(
            abstract_schema,
            PrototypeConcreteValue(3),
        )
        test_schema_round_trip(explicit_schema, Dict(:a => 1, :b => 2))

        strict_arrays = SchemaPolicy(; serialize_arrays = false)
        strict_nonconcrete = SchemaPolicy(; serialize_nonconcrete = false)
        @test_throws ArgumentError infer_schema(Vector{Float64}; policy = strict_arrays)
        @test_throws ArgumentError infer_schema(
            PrototypeAbstractValue;
            policy = strict_nonconcrete,
        )
        @test_throws ArgumentError infer_schema(Float16)

    end

end
