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

# This application codec is intentionally defined using only the public extension
# interface. It is reused at the root of a vector and recursively inside a record, and its
# schema must survive ordinary untyped loading without any package-owned codec registry.
struct PrototypeGrade
    label::String
end

struct PrototypeGradeCodec <: AbstractCodec{PrototypeGrade, UInt8} end

function HDF5Vectors2.encode_value(::PrototypeGradeCodec, grade::PrototypeGrade)
    return UInt8(only(grade.label))
end

function HDF5Vectors2.decode_value(::PrototypeGradeCodec, value::UInt8)
    return PrototypeGrade(string(Char(value)))
end

function HDF5Vectors2.infer_schema(::Type{PrototypeGrade}; kwargs...)
    return ScalarSchema(PrototypeGradeCodec())
end

# JSON is another logical conversion over scalar storage rather than a separate physical
# schema. Declaring JSON3's struct mapping makes this small type readable in both
# directions; selecting `json_schema` opts only this application type into that codec.
struct PrototypeJSONValue
    name::String
    values::Vector{Int64}
end

function Base.:(==)(first::PrototypeJSONValue, second::PrototypeJSONValue)
    return first.name == second.name && first.values == second.values
end

JSON3.StructTypes.StructType(::Type{PrototypeJSONValue}) = JSON3.StructTypes.Struct()

function HDF5Vectors2.infer_schema(::Type{PrototypeJSONValue}; kwargs...)
    return json_schema(PrototypeJSONValue)
end

struct PrototypeGradedValue
    grade::PrototypeGrade
    value::Float64
end

struct PrototypeJSONRecord
    details::PrototypeJSONValue
    value::Float64
end

function Base.:(==)(first::PrototypeJSONRecord, second::PrototypeJSONRecord)
    return first.details == second.details && first.value == second.value
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

        # An identity scalar batch already has exactly the representation HDF5 needs. The
        # batch conversion deliberately returns the same object rather than copying it into
        # a second vector before a bulk write or after a bulk read.
        float_values = Float64[1.0, 2.0, 3.0]
        @test HDF5Vectors2.encode_batch(float_schema, float_values) === float_values
        @test HDF5Vectors2.decode_batch(float_schema, float_values) === float_values

        # An application codec reaches the same scalar representation through public
        # dispatch. HDF5Vectors does not need a codec-name method or a matching reader.
        grade_schema = infer_schema(PrototypeGrade)
        @test grade_schema isa ScalarSchema
        @test encoded_type(grade_schema) === UInt8
        test_schema_round_trip(grade_schema, PrototypeGrade("A"))

        # JSON values use the same scalar-string store as ordinary strings. Loading JSON3
        # contributes only the codec operations, while the core owns the schema itself.
        json_value = PrototypeJSONValue("example", [1, 2, 3])
        json_value_schema = infer_schema(PrototypeJSONValue)
        @test json_value_schema isa ScalarSchema
        @test json_value_schema.codec isa JSONCodec{PrototypeJSONValue}
        @test encoded_type(json_value_schema) === String
        test_schema_round_trip(json_value_schema, json_value)

    end

    @testset "dense schemas" begin

        # The same scalar codec is reused inside every dense container. Arrays of
        # transformed values therefore cannot accidentally use their logical element type
        # as the HDF5 datatype.
        tuple_schema = infer_schema(NTuple{2, Char})
        static_schema = infer_schema(StaticArrays.SVector{2, PrototypeUInt8Enum})
        string_static_schema = infer_schema(StaticArrays.SVector{2, String})
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

        # Dense batch encoding constructs the stacked physical layout directly. This tests
        # both an identity element codec and a transforming codec, including reconstruction
        # from views into the stacked array rather than independent frame Arrays.
        static_values = [
            StaticArrays.SVector(prototype_zero, prototype_max),
            StaticArrays.SVector(prototype_max, prototype_zero),
        ]
        static_batch = HDF5Vectors2.encode_batch(static_schema, static_values)
        char_values = [('a', 'b'), ('c', 'd')]
        char_batch = HDF5Vectors2.encode_batch(tuple_schema, char_values)

        @test static_batch == UInt8[0 255; 255 0]
        @test char_batch == Int32['a' 'c'; 'b' 'd']
        @test HDF5Vectors2.decode_batch(static_schema, static_batch) == static_values
        @test HDF5Vectors2.decode_batch(tuple_schema, char_batch) == char_values

        # Non-bits static arrays cannot use the contiguous reinterpretation fast path, but
        # they retain the same dense batch interface through frame-by-frame reconstruction.
        string_static_values = [
            StaticArrays.SVector("first", "second"),
            StaticArrays.SVector("third", "fourth"),
        ]
        string_static_batch = HDF5Vectors2.encode_batch(
            string_static_schema,
            string_static_values,
        )
        @test HDF5Vectors2.decode_batch(
            string_static_schema,
            string_static_batch,
        ) == string_static_values

        @test_throws DimensionMismatch encode_value(array_schema, ['a'])
        @test_throws DimensionMismatch decode_value(array_schema, Int32[Int('a')])
        @test_throws DimensionMismatch HDF5Vectors2.decode_batch(
            static_schema,
            zeros(UInt8, 3, 2),
        )
        @test_throws DimensionMismatch infer_schema(Vector{Char}; dims = (2, 1))
        @test_throws ArgumentError infer_schema(Vector{Char}; dims = (0,))

    end

    @testset "record schemas" begin

        # Portable records recursively describe their fields. A dynamic array field uses a
        # blob schema by default because its dimensions are not part of its declared type.
        point_schema = infer_schema(PrototypePoint)
        sample_schema = infer_schema(PrototypeSample)
        graded_schema = infer_schema(PrototypeGradedValue)

        @test point_schema isa RecordSchema
        @test point_schema.names == ("x", "y")
        @test all(child -> child isa ScalarSchema, point_schema.children)
        @test sample_schema isa RecordSchema
        @test sample_schema.children[1] isa RecordSchema
        @test sample_schema.children[2] isa ScalarSchema
        @test sample_schema.children[3] isa BlobSchema
        @test graded_schema.children[1] isa ScalarSchema
        @test graded_schema.children[1].codec isa PrototypeGradeCodec

        # Physical record columns use these names as HDF5 path components. Rejecting names
        # that are ambiguous or unsafe keeps the stored layout readable without requiring
        # an escaping convention.
        for invalid_names in (
            ("x", "x"),
            ("", "y"),
            (".", "y"),
            ("x/y", "y"),
            ("x\0z", "y"),
        )
            @test_throws ArgumentError RecordSchema(
                PrototypePoint,
                invalid_names,
                point_schema.codec,
                point_schema.children,
            )
        end

        point = PrototypePoint(1.5, 2)
        sample = PrototypeSample(point, :sample, [3.0, 4.0])
        test_schema_round_trip(point_schema, point)
        test_schema_round_trip(sample_schema, sample)

        # A record batch follows the stored field layout rather than materializing encoded
        # row tuples. Nested records become nested batches, while scalar fields and blob
        # fields retain their own natural column representations.
        samples = [
            sample,
            PrototypeSample(PrototypePoint(5.0, 6), :other, [7.0]),
        ]
        sample_batch = HDF5Vectors2.encode_batch(sample_schema, samples)

        @test sample_batch isa HDF5Vectors2.RecordBatch
        @test sample_batch.count == length(samples)
        @test sample_batch.columns[1] isa HDF5Vectors2.RecordBatch
        @test sample_batch.columns[2] == ["sample", "other"]
        @test length(sample_batch.columns[3]) == length(samples)
        @test HDF5Vectors2.decode_batch(sample_schema, sample_batch) == samples

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

        # Policy switches are semantic choices rather than integer flags. Rejecting values
        # that Bool could convert prevents a mistyped option from silently changing the
        # stored representation.
        @test_throws ArgumentError SchemaPolicy(; portable = 1)
        @test_throws ArgumentError SchemaPolicy(; serialize_arrays = 0)
        @test_throws ArgumentError SchemaPolicy(; serialize_nonconcrete = :yes)

    end

end
