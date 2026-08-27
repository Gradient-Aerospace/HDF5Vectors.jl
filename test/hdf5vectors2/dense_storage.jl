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

                initial_values = values[1:(end - 1)]
                initial_batch = HDF5Vectors2.encode_batch(schema, initial_values)
                HDF5Vectors2.initialize_encoded!(store, initial_batch)
                final_frame = encode_value(schema, last(values))
                HDF5Vectors2.append_encoded!(store, length(values), final_frame)

                @test HDF5Vectors2.physical_length(store) == length(values)
                @test size(data_group["values"]) == (schema.dims..., length(values))
                @test HDF5Vectors2.dataset_matches_encoded_type(
                    data_group["values"],
                    encoded_type(schema),
                )

                first_encoded = HDF5Vectors2.read_encoded(store, 1)
                @test decode_value(schema, first_encoded) == first(values)
                encoded = HDF5Vectors2.read_encoded_batch(store, 1:length(values))
                @test HDF5Vectors2.decode_batch(schema, encoded) == values

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

@testset "HDF5Vectors2 dense batch and validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "dense_validation.h5"), "w") do file

            # The public copy path passes one already-stacked batch to physical storage.
            # Initialization writes that representation directly into an empty dataset.
            direct_schema = infer_schema(StaticArrays.SVector{2, Float64})
            direct_values = [
                StaticArrays.SVector(1.0, 2.0),
                StaticArrays.SVector(3.0, 4.0),
            ]
            direct_group = HDF5.create_group(file, "direct_batch")
            direct_store = HDF5Vectors2.create_store(
                direct_group,
                direct_schema;
                chunk_length = 2,
            )
            direct_batch = HDF5Vectors2.encode_batch(direct_schema, direct_values)
            HDF5Vectors2.initialize_encoded!(direct_store, direct_batch)
            stored_batch = HDF5Vectors2.read_encoded_batch(direct_store, 1:2)

            @test stored_batch == direct_batch
            @test HDF5Vectors2.decode_batch(direct_schema, stored_batch) == direct_values

            # An empty initialization preserves the stacked batch shape and leaves the
            # newly created dataset empty.
            empty_group = HDF5.create_group(file, "empty_batch")
            empty_store = HDF5Vectors2.create_store(
                empty_group,
                direct_schema;
                chunk_length = 2,
            )
            empty_batch = HDF5Vectors2.encode_batch(direct_schema, direct_values[1:0])
            @test size(empty_batch) == (2, 0)
            HDF5Vectors2.initialize_encoded!(empty_store, empty_batch)
            @test size(HDF5Vectors2.read_encoded_batch(empty_store, 1:0)) == (2, 0)
            @test iszero(HDF5Vectors2.physical_length(empty_store))

            # A malformed stacked batch is rejected before the empty dataset is extended.
            schema = infer_schema(Vector{Char}; dims = (2,))
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors2.create_store(data_group, schema; chunk_length = 2)
            wrong_batch = zeros(Int32, 3, 2)
            @test_throws DimensionMismatch HDF5Vectors2.initialize_encoded!(
                store,
                wrong_batch,
            )
            @test iszero(HDF5Vectors2.physical_length(store))

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
