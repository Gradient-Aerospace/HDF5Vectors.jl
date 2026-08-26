@testset "HDF5Vectors2 scalar storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "scalar_storage.h5"), "w") do file

            # Scalar stores receive values only after their codecs have run. The raw HDF5
            # datasets should therefore use each schema's encoded type.
            cases = (
                ("floats", infer_schema(Float64), Float64[1.5, 2.5, 3.5]),
                ("chars", infer_schema(Char), collect("abc")),
                ("symbols", infer_schema(Symbol), [:first, :second, :third]),
                (
                    "enums",
                    infer_schema(PrototypeUInt8Enum),
                    [prototype_zero, prototype_max, prototype_zero],
                ),
                (
                    "native_records",
                    infer_schema(
                        PrototypePoint;
                        policy = SchemaPolicy(; portable = false),
                    ),
                    [PrototypePoint(1.0, 2), PrototypePoint(3.0, 4)],
                ),
            )

            for (name, schema, values) in cases

                vector_group = HDF5.create_group(file, name)
                write_schema(vector_group, schema)
                data_group = HDF5.create_group(vector_group, "data")
                store = HDF5Vectors2.create_store(
                    data_group,
                    schema;
                    chunk_length = 2,
                )

                first_value = encode_value(schema, first(values))
                remaining_values = [encode_value(schema, value) for value in values[2:end]]
                HDF5Vectors2.write_encoded!(store, 1, first_value)
                HDF5Vectors2.write_encoded_batch!(store, 2, remaining_values)

                @test HDF5Vectors2.physical_length(store) == length(values)
                first_encoded = HDF5Vectors2.read_encoded(store, 1)
                @test decode_value(schema, first_encoded) == first(values)
                encoded = HDF5Vectors2.read_encoded(store, 1:length(values))
                @test [decode_value(schema, value) for value in encoded] == values
                @test HDF5Vectors2.dataset_matches_encoded_type(
                    data_group["values"],
                    encoded_type(schema),
                )

                # Opening uses the schema that was recovered from metadata and validates the
                # physical dataset before returning a store.
                loaded_schema = read_schema(vector_group)
                loaded_store = HDF5Vectors2.open_store(data_group, loaded_schema)
                @test HDF5Vectors2.physical_length(loaded_store) == length(values)

            end

            @test eltype(file["chars/data/values"]) === Int32
            @test eltype(file["symbols/data/values"]) === Cstring
            @test eltype(file["enums/data/values"]) === UInt8

        end

    end

end

@testset "HDF5Vectors2 scalar tails and validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "scalar_validation.h5"), "w") do file

            # A later vector-level write can overwrite an uncommitted physical tail at the
            # next logical index. Stores also support explicit truncation for blob recovery
            # and future repair operations.
            schema = infer_schema(Int64)
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors2.create_store(data_group, schema; chunk_length = 2)
            HDF5Vectors2.write_encoded_batch!(store, 1, Int64[1, 2, 99])
            HDF5Vectors2.write_encoded!(store, 3, Int64(3))
            @test HDF5Vectors2.read_encoded(store, 1:3) == [1, 2, 3]

            HDF5Vectors2.truncate_store!(store, 2)
            @test HDF5Vectors2.physical_length(store) == 2
            @test_throws BoundsError HDF5Vectors2.read_encoded(store, 3)
            @test_throws BoundsError HDF5Vectors2.write_encoded!(store, 4, Int64(4))
            @test_throws BoundsError HDF5Vectors2.truncate_store!(store, 3)
            @test_throws ArgumentError HDF5Vectors2.create_store(
                HDF5.create_group(file, "invalid_chunk"),
                schema;
                chunk_length = 0,
            )

            # Opening rejects a physical datatype that disagrees with the stored schema.
            invalid_group = HDF5.create_group(file, "invalid_type")
            invalid_group["values"] = Float64[]
            @test_throws ArgumentError HDF5Vectors2.open_store(invalid_group, schema)

        end

    end

end

@testset "HDF5Vectors2 constant storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "constant_storage.h5"), "w") do file

            # Constant stores intentionally contain no value dataset. The logical vector
            # length, which will live at the vector level, is sufficient to reconstruct any
            # requested value.
            marker = PrototypeSingleton1{:marker}()
            cases = (
                ("nothing", infer_schema(Nothing), nothing),
                ("marker", infer_schema(typeof(marker)), marker),
            )

            for (name, schema, value) in cases

                vector_group = HDF5.create_group(file, name)
                write_schema(vector_group, schema)
                data_group = HDF5.create_group(vector_group, "data")
                store = HDF5Vectors2.create_store(
                    data_group,
                    schema;
                    chunk_length = 2,
                )

                encoded = encode_value(schema, value)
                HDF5Vectors2.write_encoded!(store, 1, encoded)
                HDF5Vectors2.write_encoded_batch!(store, 2, fill(nothing, 2))
                @test isempty(keys(data_group))
                @test isnothing(HDF5Vectors2.physical_length(store))
                @test decode_value(schema, HDF5Vectors2.read_encoded(store, 1)) == value

                loaded_schema = read_schema(vector_group)
                loaded_store = HDF5Vectors2.open_store(data_group, loaded_schema)
                loaded = HDF5Vectors2.read_encoded(loaded_store, 1:3)
                decoded = [decode_value(loaded_schema, item) for item in loaded]
                @test decoded == fill(value, 3)

            end

        end

    end

end
