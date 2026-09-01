function test_public_hdf5_vector(
    file,
    name,
    source::Vector{T};
    kwargs...,
) where {T}

    vector = HDF5Vectors.create_hdf5_vector(
        file["/"],
        name,
        T;
        chunk_length = 2,
        kwargs...,
    )
    for value in source
        push!(vector, value)
    end

    # An HDF5Vector presents the ordinary one-dimensional array behavior needed by logging
    # and analysis code, while range reads and collection use one recursive bulk read.
    @test vector isa HDF5Vectors.HDF5Vector{T}
    @test eltype(vector) === T
    @test length(vector) == length(source)
    @test size(vector) == (length(source),)
    @test IndexStyle(typeof(vector)) == IndexLinear()
    @test collect(vector) == source
    @test vector[:] == source
    @test [value for value in vector] == source
    @test map(identity, vector) == source
    @test identity.(vector) == source
    @test read(file[name * "/metadata/count"]) == length(source)

    # Logical bounds belong to HDF5Vector rather than its recursively nested physical
    # stores. Running these checks for every representation ensures an invalid request is
    # rejected before scalar, dense, record, blob, or constant storage receives it.
    @test_throws BoundsError vector[0]
    @test_throws BoundsError vector[length(vector) + 1]
    @test_throws BoundsError vector[0:0]
    @test_throws BoundsError vector[(length(vector) + 1):(length(vector) + 1)]

    if !isempty(source)
        @test vector[1] == first(source)
        @test vector[end] == last(source)
        indices = 1:min(2, length(source))
        @test vector[indices] == source[indices]
        @test vector[collect(indices)] == source[collect(indices)]
        mask = [isodd(index) for index in eachindex(source)]
        @test vector[mask] == source[mask]
        @test vector[BitVector(mask)] == source[BitVector(mask)]
    end

    loaded = HDF5Vectors.load_hdf5_vector(file[name])
    loaded_with_type = HDF5Vectors.load_hdf5_vector(file[name], T)
    @test collect(loaded) == source
    @test collect(loaded_with_type) == source

    copied = HDF5Vectors.copy_to_hdf5_vector(
        file["/"],
        name * "_copy",
        source;
        chunk_length = 2,
        kwargs...,
    )
    @test collect(copied) == source
    reloaded_copy = HDF5Vectors.load_hdf5_vector(file[name * "_copy"])
    @test collect(reloaded_copy) == source

    # A loaded vector has validated its physical layout and can continue appending from
    # its stored logical count. Running this for every nonempty representation checks the
    # optimized known-position append path after both creation and loading.
    if !isempty(source)
        push!(reloaded_copy, last(source))
        expected = copy(source)
        push!(expected, last(source))
        @test collect(reloaded_copy) == expected
    end

end

@testset "HDF5Vectors public interface" begin

    # The package exports only the ordinary vector interface. Schema and codec extension
    # points remain public, but callers access those specialized names explicitly. Julia
    # implicitly exports a module's own name, which is not part of this declared API list.
    exported_names = Set(
        name for name in names(HDF5Vectors; all = true)
        if Base.isexported(HDF5Vectors, name)
    )
    delete!(exported_names, nameof(HDF5Vectors))
    @test exported_names == Set((
        :HDF5Vector,
        :create_hdf5_vector,
        :load_hdf5_vector,
        :copy_to_hdf5_vector,
    ))
    for name in (
        :AbstractCodec,
        :AbstractRecordCodec,
        :AbstractSchema,
        :ScalarSchema,
        :DenseSchema,
        :RecordSchema,
        :BlobSchema,
        :ConstantSchema,
        :SchemaPolicy,
        :infer_schema,
        :json_schema,
        :serialization_schema,
        :encode_value,
        :decode_value,
        :write_schema,
        :read_schema,
    )
        @test Base.ispublic(HDF5Vectors, name)
    end

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "public_routing.h5"), "w") do file

            # The ordinary creation path returns the package's one concrete vector type and
            # supports standard AbstractVector operations directly.
            vector = HDF5Vectors.create_hdf5_vector(file["/"], "pushed", Int64)
            @test vector isa HDF5Vectors.HDF5Vector{Int64}
            push!(vector, 1)
            push!(vector, 2)
            @test collect(vector) == Int64[1, 2]

            # Copying and both loading forms should return HDF5Vector and use its versioned
            # schema metadata.
            source = [PrototypePoint(1.0, 2), PrototypePoint(3.0, 4)]
            copied = HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                "copied",
                source,
            )
            @test copied isa HDF5Vectors.HDF5Vector{PrototypePoint}
            @test read(file["copied/metadata/format_name"]) == "HDF5Vectors"
            @test collect(HDF5Vectors.load_hdf5_vector(file["copied"])) == source
            @test collect(
                HDF5Vectors.load_hdf5_vector(file["copied"], PrototypePoint),
            ) == source

            # Explicit schemas use the same public creation and loading functions as
            # inferred schemas.
            schema = infer_schema(PrototypeGrade)
            explicit = HDF5Vectors.create_hdf5_vector(
                file["/"],
                "explicit",
                schema,
            )
            push!(explicit, PrototypeGrade("A"))
            @test collect(HDF5Vectors.load_hdf5_vector(
                file["explicit"],
                schema,
            )) == [PrototypeGrade("A")]

        end

    end

end

@testset "HDF5Vectors public vector operations" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "public_vectors.h5"), "w") do file

            singleton = (PrototypeSingleton1{:a}(), PrototypeSingleton2{:b}())
            cases = (
                ("scalars", Float64[1.0, 2.0, 3.0], (;)),
                ("chars", collect("abc"), (;)),
                (
                    "application_codecs",
                    [PrototypeGrade("A"), PrototypeGrade("B")],
                    (;),
                ),
                (
                    "json_codecs",
                    [
                        PrototypeJSONValue("first", [1, 2]),
                        PrototypeJSONValue("second", [3, 4, 5]),
                    ],
                    (;),
                ),
                (
                    "dense_arrays",
                    [[1.0, 2.0], [3.0, 4.0]],
                    (; dims = (2,)),
                ),
                (
                    "records",
                    [PrototypePoint(1.0, 2), PrototypePoint(3.0, 4)],
                    (;),
                ),
                (
                    "records_with_json",
                    [
                        PrototypeJSONRecord(
                            PrototypeJSONValue("first", [1, 2]),
                            1.5,
                        ),
                        PrototypeJSONRecord(
                            PrototypeJSONValue("second", [3, 4, 5]),
                            2.5,
                        ),
                    ],
                    (;),
                ),
                (
                    "record_blobs",
                    [
                        PrototypeSample(PrototypePoint(1.0, 2), :a, [3.0]),
                        PrototypeSample(PrototypePoint(4.0, 5), :b, [6.0, 7.0]),
                    ],
                    (;),
                ),
                ("serialized_arrays", [[1.0], [2.0, 3.0]], (;)),
                ("constants", Nothing[nothing, nothing], (;)),
                ("constant_records", fill(singleton, 2), (;)),
            )

            for (name, source, kwargs) in cases
                test_public_hdf5_vector(file, name, source; kwargs...)
            end
            @test read(file["application_codecs/data/values"]) == UInt8['A', 'B']
            json_source = cases[4][2]
            @test read(file["json_codecs/data/values"]) == JSON3.write.(json_source)
            @test read(file["records_with_json/data/details/values"]) ==
                JSON3.write.(getproperty.(cases[7][2], :details))

            # Empty copies still need a concrete encoded batch type. Exercising every
            # schema shape here prevents empty records or blobs from degrading to an
            # untyped collection that cannot reach the physical bulk-write method.
            empty_cases = (
                ("empty_scalars", Float64[], (;)),
                ("empty_json", PrototypeJSONValue[], (;)),
                ("empty_dense", Vector{Vector{Float64}}(), (; dims = (2,))),
                ("empty_records", PrototypePoint[], (;)),
                ("empty_blobs", Vector{Vector{Float64}}(), (;)),
                ("empty_constants", Nothing[], (;)),
                ("empty_constant_records", typeof(singleton)[], (;)),
            )
            for (name, source, kwargs) in empty_cases
                test_public_hdf5_vector(file, name, source; kwargs...)
            end

            # An empty serialized copy has no byte or stop data, but it remains appendable
            # after loading because its schema and logical count are complete.
            reloaded_empty_blob = HDF5Vectors.load_hdf5_vector(file["empty_blobs_copy"])
            push!(reloaded_empty_blob, [1.0, 2.0])
            @test collect(reloaded_empty_blob) == [[1.0, 2.0]]

            # An explicit schema bypasses inference while remaining fully loadable because
            # the exact representation is persisted with the vector.
            native_schema = infer_schema(
                PrototypePoint;
                policy = SchemaPolicy(; portable = false),
            )
            explicit = HDF5Vectors.create_hdf5_vector(
                file["/"],
                "explicit_schema",
                native_schema;
                chunk_length = 2,
            )
            push!(explicit, PrototypePoint(1.0, 2))
            @test collect(HDF5Vectors.load_hdf5_vector(file["explicit_schema"])) ==
                [PrototypePoint(1.0, 2)]

        end

    end

end

@testset "HDF5Vectors public vector validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "public_validation.h5"), "w") do file

            # Chunk length is a vector-creation option rather than an independent store
            # concern. Both public creation paths reject it before making a destination.
            @test_throws ArgumentError HDF5Vectors.create_hdf5_vector(
                file["/"],
                "invalid_create_chunk",
                Int64;
                chunk_length = 0,
            )
            @test !haskey(file, "invalid_create_chunk")
            @test_throws ArgumentError HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                "invalid_copy_chunk",
                Int64[1];
                chunk_length = 0,
            )
            @test !haskey(file, "invalid_copy_chunk")
            @test_throws ArgumentError HDF5Vectors.create_hdf5_vector(
                file["/"],
                "noninteger_chunk",
                Int64;
                chunk_length = 2.5,
            )
            @test !haskey(file, "noninteger_chunk")

            # Declared dimensions must match both the array rank and any dimensions fixed
            # by the element type. These errors should also precede destination creation.
            @test_throws ArgumentError HDF5Vectors.create_hdf5_vector(
                file["/"],
                "noninteger_dimensions",
                Vector{Float64};
                dims = (2.0,),
            )
            @test_throws DimensionMismatch HDF5Vectors.create_hdf5_vector(
                file["/"],
                "wrong_static_dimensions",
                StaticArrays.SVector{2, Float64};
                dims = (3,),
            )
            @test !haskey(file, "noninteger_dimensions")
            @test !haskey(file, "wrong_static_dimensions")

            # Bulk copying encodes every value before creating its destination. A later
            # dimension error therefore leaves no partially created vector group.
            invalid_source = [[1.0, 2.0], [3.0]]
            @test_throws DimensionMismatch HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                "invalid_copy",
                invalid_source;
                dims = (2,),
            )
            @test !haskey(file, "invalid_copy")

            # A mismatched shape remains invalid even when Julia broadcasting could expand
            # it to the declared dimensions.
            broadcastable_source = [reshape([1.0, 2.0], 2, 1)]
            @test_throws DimensionMismatch HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                "broadcastable_invalid_copy",
                broadcastable_source;
                dims = (2, 2),
            )
            @test !haskey(file, "broadcastable_invalid_copy")

            vector = HDF5Vectors.create_hdf5_vector(
                file["/"],
                "validated_push",
                Vector{Float64};
                dims = (2,),
            )
            push!(vector, [1.0, 2.0])
            @test_throws DimensionMismatch push!(vector, [3.0])
            @test length(vector) == 1
            @test read(file["validated_push/metadata/count"]) == 1
            @test size(file["validated_push/data/values"]) == (2, 1)

            # The declared element type is an input contract. A rejected value or failed
            # HDF5 write must not advance the logical count in memory.
            scalar = HDF5Vectors.create_hdf5_vector(file["/"], "failed_scalar", Int64)
            @test_throws MethodError push!(scalar, Int32(1))
            @test isempty(scalar)
            close(scalar.store.dataset)
            @test_throws ErrorException push!(scalar, Int64(1))
            @test isempty(scalar)

            # The parent must be an HDF5 group, and names designate one immediate child of
            # that group. AbstractString implementations remain valid names.
            @test_throws MethodError HDF5Vectors.create_hdf5_vector(
                file,
                "file_parent",
                Int64,
            )
            @test_throws MethodError HDF5Vectors.create_hdf5_vector(
                file["/"],
                :symbol_name,
                Int64,
            )
            @test_throws ArgumentError HDF5Vectors.create_hdf5_vector(
                file["/"],
                "nested/vector",
                Int64,
            )
            @test_throws ArgumentError HDF5Vectors.create_hdf5_vector(
                file["/"],
                "validated_push",
                Int64,
            )
            full_name = "substring_name_suffix"
            substring_name = SubString(full_name, 1, 14)
            substring_vector = HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                substring_name,
                Int64[1, 2],
            )
            @test collect(substring_vector) == Int64[1, 2]

            # The logical count and physical length must agree whenever the schema has a
            # physical length. Constant-only vectors are the intentional exception.
            mismatched = HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                "mismatched_count",
                Int64[1, 2],
            )
            write(mismatched.count_dataset, Int64(1))
            @test_throws DimensionMismatch HDF5Vectors.load_hdf5_vector(
                file["mismatched_count"],
            )

            invalid_count = HDF5Vectors.copy_to_hdf5_vector(
                file["/"],
                "invalid_count",
                Int64[1],
            )
            count_dataset = invalid_count.count_dataset
            close(count_dataset)
            HDF5.delete_object(file["invalid_count/metadata"], "count")
            file["invalid_count/metadata/count"] = Float64(1)
            @test_throws ArgumentError HDF5Vectors.load_hdf5_vector(
                file["invalid_count"],
            )

        end

    end

end
