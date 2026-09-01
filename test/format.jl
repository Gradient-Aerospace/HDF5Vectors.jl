@testset "HDF5Vectors stored schemas" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "schemas.h5"), "w") do file

            # Every built-in schema kind and codec should survive storage without repeating
            # inference. Values are encoded with the loaded schema to verify that its codec
            # was restored along with its physical representation.
            singleton_tuple = (PrototypeSingleton1{:a}(), PrototypeSingleton2{:b}())
            cases = (
                ("float", infer_schema(Float64), 1.5),
                ("char", infer_schema(Char), 'λ'),
                ("symbol", infer_schema(Symbol), :ready),
                ("enum", infer_schema(PrototypeUInt8Enum), prototype_max),
                ("application_codec", infer_schema(PrototypeGrade), PrototypeGrade("A")),
                (
                    "json_codec",
                    infer_schema(PrototypeJSONValue),
                    PrototypeJSONValue("format", [1, 2]),
                ),
                ("tuple", infer_schema(NTuple{2, Char}), ('a', 'b')),
                (
                    "static_array",
                    infer_schema(StaticArrays.SVector{2, PrototypeUInt8Enum}),
                    StaticArrays.SVector(prototype_zero, prototype_max),
                ),
                ("dynamic_array", infer_schema(Vector{Char}; dims = (2,)), ['a', 'b']),
                ("record", infer_schema(PrototypePoint), PrototypePoint(1.5, 2)),
                (
                    "native_record",
                    infer_schema(
                        PrototypePoint;
                        policy = SchemaPolicy(; portable = false),
                    ),
                    PrototypePoint(2.5, 3),
                ),
                ("singleton_tuple", infer_schema(typeof(singleton_tuple)), singleton_tuple),
                ("blob", infer_schema(Vector{Float64}), [1.0, 2.0]),
                (
                    "abstract_blob",
                    infer_schema(PrototypeAbstractValue),
                    PrototypeConcreteValue(5),
                ),
                ("constant", infer_schema(Nothing), nothing),
            )

            for (name, schema, value) in cases

                vector_group = HDF5.create_group(file, name)
                write_schema(vector_group, schema)

                loaded_with_type = read_schema(vector_group, logical_type(schema))
                loaded_from_metadata = read_schema(vector_group)
                @test typeof(loaded_with_type) === typeof(schema)
                @test typeof(loaded_from_metadata) === typeof(schema)
                test_schema_round_trip(loaded_with_type, value)
                test_schema_round_trip(loaded_from_metadata, value)

            end

            # The schema tree is stored in ordinary HDF5 values so readers can inspect the
            # selected representation without understanding Julia's serialized type data.
            metadata = file["record/metadata"]
            @test read(metadata["format_name"]) == "HDF5Vectors"
            @test read(metadata["format_version"]) == 1
            @test read(metadata["schema/kind"]) == "record"
            @test read(metadata["schema/field_names"]) == ["x", "y"]
            @test read(metadata["schema/children/1/kind"]) == "scalar"
            @test read(metadata["schema/children/1/encoded_type"]) == "Float64"

            # Codec metadata is descriptive rather than a package-owned reconstruction
            # registry. The serialized schema and the public typed-inference path both
            # recover this application codec without HDF5Vectors knowing its identifier.
            application_metadata = file["application_codec/metadata"]
            @test read(application_metadata["schema/codec"]) ==
                HDF5Vectors.codec_identifier(PrototypeGradeCodec())
            @test read_schema(file["application_codec"]).codec isa PrototypeGradeCodec
            @test read_schema(
                file["application_codec"],
                PrototypeGrade,
            ).codec isa PrototypeGradeCodec
            @test read_schema(file["json_codec"]).codec isa
                JSONCodec{PrototypeJSONValue}
            @test read_schema(file["json_codec"], PrototypeJSONValue).codec isa
                JSONCodec{PrototypeJSONValue}

            # Loading follows the stored representation, not the current default policy.
            # The two groups have the same logical type but retain different schemas.
            @test read_schema(file["record"], PrototypePoint) isa RecordSchema
            @test read_schema(file["native_record"], PrototypePoint) isa ScalarSchema

        end

    end

end

@testset "HDF5Vectors stored schema validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "invalid_schemas.h5"), "w") do file

            # Supplying a different logical type must not reinterpret the stored physical
            # schema merely because that type could independently use the same schema kind.
            typed_group = HDF5.create_group(file, "typed")
            write_schema(typed_group, infer_schema(Float64))
            @test_throws ArgumentError read_schema(typed_group, Int64)

            # A changed format version is rejected before any schema node is interpreted.
            versioned_group = HDF5.create_group(file, "versioned")
            write_schema(versioned_group, infer_schema(Float64))
            version_dataset = versioned_group["metadata/format_version"]
            write(version_dataset, Int64(2))
            @test_throws ArgumentError read_schema(versioned_group, Float64)

            # Dense dimensions stored in metadata remain constrained by static Julia types.
            dense_group = HDF5.create_group(file, "dense")
            write_schema(dense_group, infer_schema(NTuple{2, Char}))
            dimensions = dense_group["metadata/schema/dimensions"]
            dimensions[:] = Int64[3]
            @test_throws DimensionMismatch read_schema(dense_group, NTuple{2, Char})

        end

    end

end
