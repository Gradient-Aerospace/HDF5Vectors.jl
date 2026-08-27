function test_public_hdf5_vector(
    file,
    name,
    source::Vector{T};
    kwargs...,
) where {T}

    vector = HDF5Vectors2.create_hdf5_vector(
        file["/"],
        name,
        T;
        chunk_length = 2,
        kwargs...,
    )
    for value in source
        push!(vector, value)
    end

    # The prototype presents the ordinary one-dimensional array behavior needed by logging
    # and analysis code, while range reads and collection use one recursive bulk read.
    @test vector isa HDF5Vectors2.HDF5Vector{T}
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

    loaded = HDF5Vectors2.load_hdf5_vector(file[name])
    loaded_with_type = HDF5Vectors2.load_hdf5_vector(file[name], T)
    @test collect(loaded) == source
    @test collect(loaded_with_type) == source

    copied = HDF5Vectors2.copy_to_hdf5_vector(
        file["/"],
        name * "_copy",
        source;
        chunk_length = 2,
        kwargs...,
    )
    @test collect(copied) == source
    reloaded_copy = HDF5Vectors2.load_hdf5_vector(file[name * "_copy"])
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

@testset "HDF5Vectors2 public vector operations" begin

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

            # Empty copies still need a concrete encoded batch type. Exercising every
            # schema shape here prevents empty records or blobs from degrading to an
            # untyped collection that cannot reach the physical bulk-write method.
            empty_cases = (
                ("empty_scalars", Float64[], (;)),
                ("empty_dense", Vector{Vector{Float64}}(), (; dims = (2,))),
                ("empty_records", PrototypePoint[], (;)),
                ("empty_blobs", Vector{Vector{Float64}}(), (;)),
                ("empty_constants", Nothing[], (;)),
                ("empty_constant_records", typeof(singleton)[], (;)),
            )
            for (name, source, kwargs) in empty_cases
                test_public_hdf5_vector(file, name, source; kwargs...)
            end

            # An explicit schema bypasses inference while remaining fully loadable because
            # the exact representation is persisted with the vector.
            native_schema = infer_schema(
                PrototypePoint;
                policy = SchemaPolicy(; portable = false),
            )
            explicit = HDF5Vectors2.create_hdf5_vector(
                file["/"],
                "explicit_schema",
                native_schema;
                chunk_length = 2,
            )
            push!(explicit, PrototypePoint(1.0, 2))
            @test collect(HDF5Vectors2.load_hdf5_vector(file["explicit_schema"])) ==
                [PrototypePoint(1.0, 2)]

        end

    end

end

@testset "HDF5Vectors2 public vector validation" begin

    mktempdir() do directory

        HDF5.h5open(joinpath(directory, "public_validation.h5"), "w") do file

            # Bulk copying encodes every value before creating its destination. A later
            # dimension error therefore leaves no partially created vector group.
            invalid_source = [[1.0, 2.0], [3.0]]
            @test_throws DimensionMismatch HDF5Vectors2.copy_to_hdf5_vector(
                file["/"],
                "invalid_copy",
                invalid_source;
                dims = (2,),
            )
            @test !haskey(file, "invalid_copy")

            vector = HDF5Vectors2.create_hdf5_vector(
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

            # Names designate one immediate child of the supplied HDF5 group.
            @test_throws ArgumentError HDF5Vectors2.create_hdf5_vector(
                file["/"],
                "nested/vector",
                Int64,
            )
            @test_throws ArgumentError HDF5Vectors2.create_hdf5_vector(
                file["/"],
                "validated_push",
                Int64,
            )

            # The logical count and physical length must agree whenever the schema has a
            # physical length. Constant-only vectors are the intentional exception.
            mismatched = HDF5Vectors2.copy_to_hdf5_vector(
                file["/"],
                "mismatched_count",
                Int64[1, 2],
            )
            write(mismatched.count_dataset, Int64(1))
            @test_throws DimensionMismatch HDF5Vectors2.load_hdf5_vector(
                file["mismatched_count"],
            )

            invalid_count = HDF5Vectors2.copy_to_hdf5_vector(
                file["/"],
                "invalid_count",
                Int64[1],
            )
            count_dataset = invalid_count.count_dataset
            close(count_dataset)
            HDF5.delete_object(file["invalid_count/metadata"], "count")
            file["invalid_count/metadata/count"] = Float64(1)
            @test_throws ArgumentError HDF5Vectors2.load_hdf5_vector(
                file["invalid_count"],
            )

        end

    end

end
