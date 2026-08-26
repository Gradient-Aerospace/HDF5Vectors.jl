struct PrototypeRecordWithDenseField
    point::PrototypePoint
    label::Symbol
    samples::NTuple{2, Char}
    marker::PrototypeSingleton1{:record}
end

@testset "HDF5Vectors2 record storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "record_storage.h5"), "w") do file

            schema = infer_schema(PrototypeRecordWithDenseField)
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
            store = HDF5Vectors2.create_store(data_group, schema; chunk_length = 2)

            first_encoded = encode_value(schema, first(values))
            remaining_encoded = [
                encode_value(schema, value) for value in values[2:end]
            ]
            HDF5Vectors2.write_encoded!(store, 1, first_encoded)
            HDF5Vectors2.write_encoded_batch!(store, 2, remaining_encoded)

            @test HDF5Vectors2.physical_length(store) == length(values)
            @test decode_value(schema, HDF5Vectors2.read_encoded(store, 1)) == first(values)
            encoded = HDF5Vectors2.read_encoded(store, 1:length(values))
            @test [decode_value(schema, value) for value in encoded] == values

            # Fields use stable numeric groups, while each child schema selects its own
            # physical representation recursively. The fourth field is constant and needs
            # no value dataset.
            child_names = Set(String(name) for name in keys(data_group))
            @test child_names == Set(["1", "2", "3", "4"])
            @test Set(String(name) for name in keys(data_group["1"])) == Set(["1", "2"])
            @test size(data_group["1/1/values"]) == (length(values),)
            @test size(data_group["1/2/values"]) == (length(values),)
            @test size(data_group["2/values"]) == (length(values),)
            @test size(data_group["3/values"]) == (2, length(values))
            @test isempty(keys(data_group["4"]))

            # Loading uses the recursively stored schema and validates the lengths of all
            # nonconstant columns before returning the record store.
            loaded_schema = read_schema(vector_group)
            loaded_store = HDF5Vectors2.open_store(data_group, loaded_schema)
            @test HDF5Vectors2.physical_length(loaded_store) == length(values)
            loaded = HDF5Vectors2.read_encoded(loaded_store, 1:length(values))
            @test [decode_value(loaded_schema, value) for value in loaded] == values

        end

    end

end

@testset "HDF5Vectors2 record codec integration" begin

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
                store = HDF5Vectors2.create_store(
                    data_group,
                    schema;
                    chunk_length = 2,
                )

                encoded = [encode_value(schema, value) for value in values]
                HDF5Vectors2.write_encoded_batch!(store, 1, encoded)

                loaded_schema = read_schema(vector_group)
                loaded_store = HDF5Vectors2.open_store(data_group, loaded_schema)
                loaded = HDF5Vectors2.read_encoded(loaded_store, 1:length(values))
                @test [decode_value(loaded_schema, value) for value in loaded] == values

            end

        end

    end

end

@testset "HDF5Vectors2 all-constant record storage" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "constant_record_storage.h5"), "w") do file

            # This is the singleton tuple that could not be reconstructed by the old
            # singleton implementation. It remains a record, but both of its columns are
            # constants and therefore have no physical length of their own.
            value = (PrototypeSingleton1{:a}(), PrototypeSingleton2{:b}())
            schema = infer_schema(typeof(value))
            data_group = HDF5.create_group(file, "data")
            store = HDF5Vectors2.create_store(data_group, schema; chunk_length = 2)
            encoded = encode_value(schema, value)

            HDF5Vectors2.write_encoded!(store, 1, encoded)
            HDF5Vectors2.write_encoded_batch!(store, 2, fill(encoded, 2))
            @test isnothing(HDF5Vectors2.physical_length(store))
            @test all(child -> isempty(keys(child)), values(data_group))

            loaded = HDF5Vectors2.read_encoded(store, 1:3)
            @test [decode_value(schema, item) for item in loaded] == fill(value, 3)

        end

    end

end

@testset "HDF5Vectors2 record tails and validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "record_validation.h5"), "w") do file

            # This explicit schema gives one record field a runtime-variable Array shape.
            # It lets the test verify that recursive preflight reaches every record in a
            # batch before any child dataset changes.
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
            store = HDF5Vectors2.create_store(data_group, schema; chunk_length = 2)
            valid_value = (Int32[Int('a'), Int('b')], Int64(1))
            invalid_value = (Int32[Int('c')], Int64(2))

            @test_throws DimensionMismatch HDF5Vectors2.write_encoded_batch!(
                store,
                1,
                [valid_value, invalid_value],
            )
            @test HDF5Vectors2.physical_length(store) == 0
            @test size(data_group["1/values"]) == (2, 0)
            @test isempty(data_group["2/values"])

            HDF5Vectors2.write_encoded_batch!(store, 1, [valid_value, valid_value])
            HDF5Vectors2.truncate_store!(store, 1)
            @test HDF5Vectors2.physical_length(store) == 1
            @test HDF5Vectors2.read_encoded(store, 1) == valid_value
            @test_throws BoundsError HDF5Vectors2.truncate_store!(store, 2)

            # A record group must contain exactly one child per schema field.
            missing_child_group = HDF5.create_group(file, "missing_child")
            HDF5.create_group(missing_child_group, "1")
            @test_throws ArgumentError HDF5Vectors2.open_store(
                missing_child_group,
                schema,
            )

            # Child stores may be individually valid while disagreeing about record count.
            # Opening rejects that incomplete multi-column layout immediately.
            unequal_group = HDF5.create_group(file, "unequal_lengths")
            unequal_dense_group = HDF5.create_group(unequal_group, "1")
            unequal_scalar_group = HDF5.create_group(unequal_group, "2")
            unequal_dense_group["values"] = zeros(Int32, 2, 1)
            unequal_scalar_group["values"] = Int64[1, 2]
            @test_throws DimensionMismatch HDF5Vectors2.open_store(unequal_group, schema)

        end

    end

end
