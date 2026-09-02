module RecordStorageTests

include("_test_setup.jl")

struct PrototypeRecordWithDenseField
    point::PrototypePoint
    label::Symbol
    samples::NTuple{2, Char}
    marker::PrototypeSingleton1{:record}
end

@testset "HDF5Vectors record storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "record_storage.h5"), "w") do file

            # Field names belong to the schema that describes metadata and HDF5 paths. The
            # positional struct codec does not retain a duplicate copy of them.
            schema = infer_schema(PrototypeRecordWithDenseField)
            @test fieldcount(typeof(schema.codec)) == 0
            values = [
                PrototypeRecordWithDenseField(
                    PrototypePoint(1.0, 2),
                    :first,
                    ('a', 'b'),
                    PrototypeSingleton1{:record}(),
                ),
                PrototypeRecordWithDenseField(
                    PrototypePoint(3.0, 4),
                    :second,
                    ('c', 'd'),
                    PrototypeSingleton1{:record}(),
                ),
                PrototypeRecordWithDenseField(
                    PrototypePoint(5.0, 6),
                    :third,
                    ('e', 'f'),
                    PrototypeSingleton1{:record}(),
                ),
            ]

            vector_group = HDF5.create_group(file, "records")
            write_schema(vector_group, schema)
            data_group = HDF5.create_group(vector_group, "data")
            store = HDF5Vectors.create_store(data_group, schema; chunk_length = 2)

            initial_batch = HDF5Vectors.encode_batch(schema, values[1:(end - 1)])
            HDF5Vectors.initialize_encoded!(store, initial_batch)
            final_encoded = encode_value(schema, last(values))
            HDF5Vectors.append_encoded!(store, length(values), final_encoded)

            @test HDF5Vectors.physical_length(store) == length(values)
            @test decode_value(schema, HDF5Vectors.read_encoded(store, 1)) == first(values)
            encoded = HDF5Vectors.read_encoded_batch(store, 1:length(values))
            @test HDF5Vectors.decode_batch(schema, encoded) == values

            # The optimized path preserves columns throughout bulk encoding, storage, and
            # decoding. The nested point is itself a record batch, and the tuple field is
            # already stacked in its final dense HDF5 layout.
            batch = HDF5Vectors.encode_batch(schema, values)
            batch_group = HDF5.create_group(file, "record_batch")
            batch_store = HDF5Vectors.create_store(
                batch_group,
                schema;
                chunk_length = 2,
            )
            HDF5Vectors.initialize_encoded!(batch_store, batch)
            stored_batch = HDF5Vectors.read_encoded_batch(
                batch_store,
                1:length(values),
            )

            @test batch.columns[1] isa HDF5Vectors.RecordBatch
            @test size(batch.columns[3]) == (2, length(values))
            @test HDF5Vectors.decode_batch(schema, stored_batch) == values

            # Each child schema selects its physical representation recursively. Field
            # names make the layout meaningful to readers outside Julia, while the stored
            # field-name vector preserves declaration order independently of HDF5 links.
            child_names = Set(String(name) for name in keys(data_group))
            @test child_names == Set(["point", "label", "samples", "marker"])
            @test read(vector_group["metadata/schema/field_names"]) ==
                ["point", "label", "samples", "marker"]
            point_names = Set(String(name) for name in keys(data_group["point"]))
            @test point_names == Set(["x", "y"])
            @test size(data_group["point/x/values"]) == (length(values),)
            @test size(data_group["point/y/values"]) == (length(values),)
            @test size(data_group["label/values"]) == (length(values),)
            @test size(data_group["samples/values"]) == (2, length(values))
            @test isempty(keys(data_group["marker"]))

            # Opening reconstructs the recursively stored layout. One root physical-length
            # query then validates the lengths of every nonconstant column.
            loaded_schema = read_schema(vector_group)
            loaded_store = HDF5Vectors.open_store(data_group, loaded_schema)
            @test HDF5Vectors.physical_length(loaded_store) == length(values)
            loaded = HDF5Vectors.read_encoded_batch(
                loaded_store,
                1:length(values),
            )
            @test HDF5Vectors.decode_batch(loaded_schema, loaded) == values

        end

    end

end

@testset "HDF5Vectors record codec integration" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "record_codecs.h5"), "w") do file

            # Named tuples and static arrays use different logical reconstruction codecs,
            # but both become the same recursive physical record abstraction. The static
            # array case also places records inside an array-like container.
            named_values = [
                (point = PrototypePoint(1.0, 2), label = :first),
                (point = PrototypePoint(3.0, 4), label = :second),
            ]
            static_values = [
                StaticArrays.SVector(PrototypePoint(1.0, 2), PrototypePoint(3.0, 4)),
                StaticArrays.SVector(PrototypePoint(5.0, 6), PrototypePoint(7.0, 8)),
            ]
            cases = (
                ("named_tuples", named_values),
                ("static_record_arrays", static_values),
            )

            for (name, values) in cases

                schema = infer_schema(eltype(values))
                vector_group = HDF5.create_group(file, name)
                write_schema(vector_group, schema)
                data_group = HDF5.create_group(vector_group, "data")
                store = HDF5Vectors.create_store(
                    data_group,
                    schema;
                    chunk_length = 2,
                )

                encoded = HDF5Vectors.encode_batch(schema, values)
                HDF5Vectors.initialize_encoded!(store, encoded)

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

end

@testset "HDF5Vectors all-constant record storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "constant_record_storage.h5"), "w") do file

            # This is the singleton tuple that could not be reconstructed by the old
            # singleton implementation. It remains a record, but both of its columns are
            # constants and therefore have no physical length of their own.
            value = (PrototypeSingleton1{:a}(), PrototypeSingleton2{:b}())
            schema = infer_schema(typeof(value))
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors.create_store(data_group, schema; chunk_length = 2)
            initial_batch = HDF5Vectors.encode_batch(schema, fill(value, 2))
            HDF5Vectors.initialize_encoded!(store, initial_batch)
            encoded = encode_value(schema, value)
            HDF5Vectors.append_encoded!(store, 3, encoded)
            @test isnothing(HDF5Vectors.physical_length(store))
            @test all(child -> isempty(keys(child)), values(data_group))

            loaded = HDF5Vectors.read_encoded_batch(store, 1:3)
            @test HDF5Vectors.decode_batch(schema, loaded) == fill(value, 3)

        end

    end

end

@testset "HDF5Vectors record batch validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "record_validation.h5"), "w") do file

            # This explicit schema allows a malformed encoded column batch to be
            # constructed directly. Recursive preflight must reach every field before any
            # child dataset changes.
            logical_type = Tuple{Vector{Char}, Int64}
            schema = RecordSchema(
                logical_type,
                ("1", "2"),
                TupleCodec{logical_type}(),
                (
                    DenseSchema(Vector{Char}, (2,), CharCodec()),
                    ScalarSchema(IdentityCodec{Int64}()),
                ),
            )
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors.create_store(data_group, schema; chunk_length = 2)

            # A malformed column batch is rejected in full before the first field is
            # written. Here the dense field contains two records, but the scalar field
            # contains only one despite the batch's declared count of two.
            invalid_batch = HDF5Vectors.RecordBatch(
                (
                    Int32[Int('a') Int('c'); Int('b') Int('d')],
                    Int64[1],
                ),
                2,
            )
            @test_throws DimensionMismatch HDF5Vectors.initialize_encoded!(
                store,
                invalid_batch,
            )
            @test iszero(HDF5Vectors.physical_length(store))
            @test size(data_group["1/values"]) == (2, 0)
            @test isempty(data_group["2/values"])

            valid_values = [(['a', 'b'], Int64(1)), (['c', 'd'], Int64(2))]
            valid_batch = HDF5Vectors.encode_batch(schema, valid_values)
            HDF5Vectors.initialize_encoded!(store, valid_batch)
            @test HDF5Vectors.physical_length(store) == 2
            @test decode_value(schema, HDF5Vectors.read_encoded(store, 1)) ==
                first(valid_values)

            # A record group must contain exactly one child per schema field.
            missing_child_group = HDF5.create_group(file, "missing_child")
            HDF5.create_group(missing_child_group, "1")
            @test_throws ArgumentError HDF5Vectors.open_store(
                missing_child_group,
                schema,
            )

            # Child stores may be individually valid while disagreeing about record count.
            # Opening constructs the recursive store, while the root physical-length check
            # validates the complete multi-column layout once.
            unequal_group = HDF5.create_group(file, "unequal_lengths")
            unequal_dense_group = HDF5.create_group(unequal_group, "1")
            unequal_scalar_group = HDF5.create_group(unequal_group, "2")
            unequal_dense_group["values"] = zeros(Int32, 2, 1)
            unequal_scalar_group["values"] = Int64[1, 2]
            unequal_store = HDF5Vectors.open_store(unequal_group, schema)
            @test_throws DimensionMismatch HDF5Vectors.physical_length(unequal_store)

        end

    end

end

end # module RecordStorageTests
