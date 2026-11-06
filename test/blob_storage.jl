module BlobStorageTests

include("_test_setup.jl")

@testset "HDF5Vectors blob storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "blob_storage.h5"), "w") do file

            # Blob schemas differ in logical type but share one physical representation.
            # Encoding remains a pure codec operation performed before the store is called.
            cases = (
                (
                    "vectors",
                    infer_schema(Vector{Float64}),
                    [[1.0, 2.0], Float64[], [3.0]],
                ),
                (
                    "abstract_values",
                    infer_schema(PrototypeAbstractValue),
                    PrototypeAbstractValue[
                        PrototypeConcreteValue(1),
                        PrototypeConcreteValue(2),
                    ],
                ),
                (
                    "dictionaries",
                    serialization_schema(Dict{Symbol, Int64}),
                    [Dict(:first => 1), Dict(:second => 2, :third => 3)],
                ),
            )

            for (name, schema, values) in cases

                vector_group = HDF5.create_group(file, name)
                write_schema(vector_group, schema)
                data_group = HDF5.create_group(vector_group, "data")
                store = HDF5Vectors.create_store(
                    data_group,
                    schema;
                    chunk_length = 8,
                )

                encoded = [encode_value(schema, value) for value in values]
                HDF5Vectors.initialize_encoded!(store, encoded[1:(end - 1)])
                HDF5Vectors.append_encoded!(store, length(encoded), last(encoded))

                @test HDF5Vectors.physical_length(store) == length(values)
                @test Set(String(child) for child in keys(data_group)) ==
                    Set(["bytes", "stops"])
                @test read(data_group["bytes"]) == vcat(encoded...)
                encoded_lengths = Int64[length(item) for item in encoded]
                @test read(data_group["stops"]) == cumsum(encoded_lengths)

                first_encoded = HDF5Vectors.read_encoded(store, 1)
                @test decode_value(schema, first_encoded) == first(values)
                stored = HDF5Vectors.read_encoded_batch(store, 1:length(values))
                @test HDF5Vectors.decode_batch(schema, stored) == values

                loaded_schema = read_schema(vector_group)
                loaded_store = HDF5Vectors.open_store(data_group, loaded_schema)
                @test HDF5Vectors.physical_length(loaded_store) == length(values)

            end

        end

    end

end

@testset "HDF5Vectors blobs inside records" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "blob_records.h5"), "w") do file

            # Recursive record storage treats the blob like any other child store. The
            # serialized bytes remain isolated under the field's meaningful name.
            values = [
                PrototypeSample(PrototypePoint(1.0, 2), :first, [3.0, 4.0]),
                PrototypeSample(PrototypePoint(5.0, 6), :second, [7.0]),
            ]
            schema = infer_schema(PrototypeSample)
            vector_group = HDF5.create_group(file, "samples")
            write_schema(vector_group, schema)
            data_group = HDF5.create_group(vector_group, "data")
            store = HDF5Vectors.create_store(data_group, schema; chunk_length = 8)

            encoded = HDF5Vectors.encode_batch(schema, values)
            HDF5Vectors.initialize_encoded!(store, encoded)
            @test size(data_group["point/x/values"]) == (2,)
            @test size(data_group["label/values"]) == (2,)
            @test Set(String(child) for child in keys(data_group["values"])) ==
                Set(["bytes", "stops"])

            loaded_schema = read_schema(vector_group)
            loaded_store = HDF5Vectors.open_store(data_group, loaded_schema)
            loaded = HDF5Vectors.read_encoded_batch(
                loaded_store,
                1:length(values),
            )
            @test HDF5Vectors.decode_batch(loaded_schema, loaded) == values

        end

    end

end

@testset "HDF5Vectors blob layout validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "blob_validation.h5"), "w") do file

            schema = serialization_schema(Vector{Int64})
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors.create_store(data_group, schema; chunk_length = 4)
            encoded = HDF5Vectors.encode_batch(
                schema,
                [Int64[1, 2], Int64[3]],
            )
            HDF5Vectors.initialize_encoded!(store, encoded)
            @test HDF5Vectors.decode_batch(
                schema,
                HDF5Vectors.read_encoded_batch(store, 1:2),
            ) == [Int64[1, 2], Int64[3]]

            # Repeated stop positions are the natural representation of empty encoded
            # values. They remain distinct logical entries despite consuming no bytes.
            empty_group = HDF5.create_group(file, "empty_values")
            empty_store = HDF5Vectors.create_store(
                empty_group,
                schema;
                chunk_length = 4,
            )
            raw_values = [UInt8[], UInt8[1], UInt8[]]
            HDF5Vectors.initialize_encoded!(empty_store, raw_values)
            @test read(empty_group["bytes"]) == UInt8[1]
            @test read(empty_group["stops"]) == Int64[0, 1, 1]
            @test HDF5Vectors.read_encoded_batch(empty_store, 1:3) == raw_values

            # Opening validates both dataset representations and their shared final byte
            # boundary. It does not need to scan every stored stop position.
            wrong_type_group = HDF5.create_group(file, "wrong_type")
            wrong_type_group["bytes"] = Int64[]
            wrong_type_group["stops"] = Int64[]
            @test_throws ArgumentError HDF5Vectors.open_store(wrong_type_group, schema)

            wrong_stop_type_group = HDF5.create_group(file, "wrong_stop_type")
            wrong_stop_type_group["bytes"] = UInt8[]
            wrong_stop_type_group["stops"] = UInt32[]
            @test_throws ArgumentError HDF5Vectors.open_store(
                wrong_stop_type_group,
                schema,
            )

            wrong_rank_group = HDF5.create_group(file, "wrong_rank")
            wrong_rank_group["bytes"] = zeros(UInt8, 1, 1)
            wrong_rank_group["stops"] = Int64[1]
            @test_throws DimensionMismatch HDF5Vectors.open_store(
                wrong_rank_group,
                schema,
            )

            wrong_boundary_group = HDF5.create_group(file, "wrong_boundary")
            wrong_boundary_group["bytes"] = UInt8[1, 2]
            wrong_boundary_group["stops"] = Int64[1]
            @test_throws DimensionMismatch HDF5Vectors.open_store(
                wrong_boundary_group,
                schema,
            )

        end

    end

end

@testset "HDF5Vectors blob byte chunks" begin

    mktempdir() do directory

        path = joinpath(directory, "blob_chunks.h5")
        values = [(samples = Float32[1, 2, 3],), (samples = Float32[],)]
        HDF5.h5open(path, "w") do file

            # A one-record chunk must not turn a nested serialized array into one-byte
            # chunks. Both ordinary pushes and bulk copies share this storage policy.
            root = HDF5.create_group(file, "vectors")
            pushed = HDF5Vectors.create_hdf5_vector(
                root,
                "pushed",
                eltype(values);
                chunk_length = 1,
            )
            append!(pushed, values)
            HDF5Vectors.copy_to_hdf5_vector(
                root,
                "bulk",
                values;
                chunk_length = 1,
            )
            for name in ("pushed", "bulk")

                bytes = root["$name/data/samples/bytes"]
                stops = root["$name/data/samples/stops"]
                @test HDF5.get_chunk(bytes) == (65536,)
                @test HDF5.get_num_chunks(bytes) == 1
                @test HDF5.get_chunk(stops) == (1,)
                schema = serialization_schema(Vector{Float32})
                encoded = [encode_value(schema, value.samples) for value in values]
                @test read(stops) == cumsum(length.(encoded))
                @test read(bytes) == vcat(encoded...)

            end

        end

        # Closing the file exercises normal metadata and schema reconstruction on reload.
        HDF5.h5open(path, "r") do file
            for name in ("pushed", "bulk")
                loaded = HDF5Vectors.load_hdf5_vector(file["vectors/$name"])
                @test collect(loaded) == values
            end
        end

    end

end

@testset "HDF5Vectors blob chunk boundaries" begin

    mktempdir() do directory

        path = joinpath(directory, "blob_boundaries.h5")
        values = [
            (samples = Float32.(1:10000),),
            (samples = Float32.(10001:20000),),
            (samples = Float32[],),
        ]
        HDF5.h5open(path, "w") do file

            # The combined payload crosses a byte chunk boundary in both write paths,
            # while each stop still occupies its requested one-record chunk.
            root = HDF5.create_group(file, "vectors")
            pushed = HDF5Vectors.create_hdf5_vector(
                root,
                "pushed",
                eltype(values);
                chunk_length = 1,
            )
            append!(pushed, values)
            HDF5Vectors.copy_to_hdf5_vector(
                root,
                "bulk",
                values;
                chunk_length = 1,
            )
            for name in ("pushed", "bulk")
                bytes = root["$name/data/samples/bytes"]
                @test HDF5.get_num_chunks(bytes) == 2
            end

        end

        # Appending after reopening must preserve stops and span another byte boundary.
        extra = (samples = Float32.(1:20000),)
        HDF5.h5open(path, "r+") do file
            for name in ("pushed", "bulk")
                loaded = HDF5Vectors.load_hdf5_vector(file["vectors/$name"])
                @test collect(loaded) == values
                push!(loaded, extra)
            end
        end
        expected = [values; extra]
        HDF5.h5open(path, "r") do file
            for name in ("pushed", "bulk")
                loaded = HDF5Vectors.load_hdf5_vector(file["vectors/$name"])
                @test collect(loaded) == expected
                @test loaded[2:4] == expected[2:4]
                @test loaded[4] == extra
                bytes = file["vectors/$name/data/samples/bytes"]
                @test HDF5.get_num_chunks(bytes) == 3
                @test HDF5.get_chunk(file["vectors/$name/data/samples/stops"]) == (1,)
            end
        end

    end

end

end # module BlobStorageTests
