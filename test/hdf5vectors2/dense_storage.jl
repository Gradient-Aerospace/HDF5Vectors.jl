function zero_dimensional_array(value::T) where {T}
    array = Array{T, 0}(undef)
    array[] = value
    return array
end

@testset "HDF5Vectors2 dense storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "dense_storage.h5"), "w") do file

            # Dense stores apply no codec logic themselves. Each case therefore exercises
            # both the pure schema conversion and the common physical frame layout.
            cases = (
                (
                    "tuples",
                    infer_schema(NTuple{2, Char}),
                    [('a', 'b'), ('c', 'd'), ('e', 'f')],
                ),
                (
                    "static_arrays",
                    infer_schema(StaticArrays.SVector{2, PrototypeUInt8Enum}),
                    [
                        StaticArrays.SVector(prototype_zero, prototype_max),
                        StaticArrays.SVector(prototype_max, prototype_zero),
                    ],
                ),
                (
                    "vectors",
                    infer_schema(Vector{Symbol}; dims = (2,)),
                    [[:first, :second], [:third, :fourth]],
                ),
                (
                    "matrices",
                    infer_schema(Matrix{Float64}; dims = (2, 2)),
                    [
                        [1.0 2.0; 3.0 4.0],
                        [5.0 6.0; 7.0 8.0],
                    ],
                ),
                (
                    "zero_dimensional_arrays",
                    infer_schema(Array{Int64, 0}; dims = ()),
                    [zero_dimensional_array(1), zero_dimensional_array(2)],
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

                first_frame = encode_value(schema, first(values))
                remaining_frames = [
                    encode_value(schema, value) for value in values[2:end]
                ]
                HDF5Vectors2.write_encoded!(store, 1, first_frame)
                HDF5Vectors2.write_encoded_batch!(store, 2, remaining_frames)

                @test HDF5Vectors2.physical_length(store) == length(values)
                @test size(data_group["values"]) == (schema.dims..., length(values))
                @test HDF5Vectors2.dataset_matches_encoded_type(
                    data_group["values"],
                    encoded_type(schema),
                )

                first_encoded = HDF5Vectors2.read_encoded(store, 1)
                @test decode_value(schema, first_encoded) == first(values)
                encoded = HDF5Vectors2.read_encoded(store, 1:length(values))
                @test [decode_value(schema, frame) for frame in encoded] == values

                # Reopening obtains dimensions and codecs from stored metadata, then checks
                # that the physical dataset is the representation those tags describe.
                loaded_schema = read_schema(vector_group)
                loaded_store = HDF5Vectors2.open_store(data_group, loaded_schema)
                @test loaded_store.dims == schema.dims
                @test HDF5Vectors2.physical_length(loaded_store) == length(values)

            end

            @test eltype(file["tuples/data/values"]) === Int32
            @test eltype(file["static_arrays/data/values"]) === UInt8
            @test eltype(file["vectors/data/values"]) === Cstring

        end

    end

end

@testset "HDF5Vectors2 dense tails and validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "dense_validation.h5"), "w") do file

            schema = infer_schema(Vector{Char}; dims = (2,))
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors2.create_store(data_group, schema; chunk_length = 2)
            frames = [encode_value(schema, value) for value in [['a', 'b'], ['c', 'd']]]
            HDF5Vectors2.write_encoded_batch!(store, 1, frames)

            # An existing physical tail can be replaced, while a write cannot skip over
            # the next available position.
            replacement = encode_value(schema, ['e', 'f'])
            HDF5Vectors2.write_encoded!(store, 2, replacement)
            @test decode_value(schema, HDF5Vectors2.read_encoded(store, 2)) == ['e', 'f']
            @test_throws BoundsError HDF5Vectors2.write_encoded!(store, 4, replacement)

            HDF5Vectors2.truncate_store!(store, 1)
            @test HDF5Vectors2.physical_length(store) == 1
            @test_throws BoundsError HDF5Vectors2.read_encoded(store, 2)
            @test_throws BoundsError HDF5Vectors2.truncate_store!(store, 2)

            # Empty ranges preserve the encoded frame type and do not change storage.
            empty_frames = Array{Int32, 1}[]
            HDF5Vectors2.write_encoded_batch!(store, 2, empty_frames)
            @test HDF5Vectors2.read_encoded(store, 2:1) == empty_frames
            @test HDF5Vectors2.physical_length(store) == 1

            # Dynamic Arrays can have the correct element type and rank but the wrong
            # dimensions. Single and batch writes reject them before extending storage.
            wrong_frame = Int32[1, 2, 3]
            @test_throws DimensionMismatch HDF5Vectors2.write_encoded!(
                store,
                2,
                wrong_frame,
            )
            @test HDF5Vectors2.physical_length(store) == 1
            @test_throws DimensionMismatch HDF5Vectors2.write_encoded_batch!(
                store,
                2,
                [replacement, wrong_frame],
            )
            @test HDF5Vectors2.physical_length(store) == 1

            # Reopening detects each independent part of the dense layout contract: rank,
            # fixed leading dimensions, and encoded HDF5 datatype.
            wrong_rank_group = HDF5.create_group(file, "wrong_rank")
            wrong_rank_group["values"] = Int32[1, 2]
            @test_throws DimensionMismatch HDF5Vectors2.open_store(
                wrong_rank_group,
                schema,
            )

            wrong_dims_group = HDF5.create_group(file, "wrong_dims")
            wrong_dims_group["values"] = zeros(Int32, 3, 1)
            @test_throws DimensionMismatch HDF5Vectors2.open_store(
                wrong_dims_group,
                schema,
            )

            wrong_type_group = HDF5.create_group(file, "wrong_type")
            wrong_type_group["values"] = zeros(Float64, 2, 1)
            @test_throws ArgumentError HDF5Vectors2.open_store(wrong_type_group, schema)

        end

    end

end
