# This benchmark tracks representative public operations without making performance a CI
# assertion. It can be run from the package root with:
#
#     julia --project=. benchmark/speed_and_allocations.jl

import HDF5
import HDF5Vectors
import StaticArrays

struct BenchmarkTimestamp
    weeks::Int32
    microseconds::Int64
end

struct BenchmarkMeasurement
    timestamp::BenchmarkTimestamp
    temperature::Float64
    position::StaticArrays.SVector{3, Float64}
end

function median(values)
    ordered = sort(values)
    return ordered[(length(ordered) + 1) ÷ 2]
end

function measure(action; samples)

    # The warm-up run ensures compilation and first-use HDF5 initialization do not dominate
    # the reported samples. Garbage collection before each sample reduces unrelated noise.
    action()
    times = Float64[]
    allocations = Int[]
    for _ in 1:samples
        GC.gc()
        result = @timed action()
        push!(times, result.time)
        push!(allocations, result.bytes)
    end
    return (; time = median(times), bytes = median(allocations))

end

function next_path(directory, prefix, counter)
    counter[] += 1
    return joinpath(directory, "$(prefix)_$(counter[]).h5")
end

function copy_values(path, source, options)
    return HDF5.h5open(path, "w") do file
        vector = HDF5Vectors.copy_to_hdf5_vector(
            file["/"],
            "values",
            source;
            chunk_length = 256,
            options...,
        )
        return length(vector)
    end
end

function push_values(path, source, options)
    return HDF5.h5open(path, "w") do file
        vector = HDF5Vectors.create_hdf5_vector(
            file["/"],
            "values",
            eltype(source);
            chunk_length = 256,
            options...,
        )
        for value in source
            push!(vector, value)
        end
        return length(vector)
    end
end

function load_and_collect(path)
    return HDF5.h5open(path, "r") do file
        return collect(HDF5Vectors.load_hdf5_vector(file["values"]))
    end
end

function benchmark_workload(
    directory,
    name,
    source,
    options;
    push_count,
    samples,
)

    counter = Ref(0)
    prefix = replace(name, ' ' => '_')
    collect_path = next_path(directory, prefix * "_collect", counter)
    copy_values(collect_path, source, options)

    # Correctness is checked before timing so an incorrect implementation cannot produce a
    # persuasive benchmark result.
    @assert load_and_collect(collect_path) == source

    push_source = source[1:min(push_count, length(source))]
    copy_result = measure(; samples) do
        path = next_path(directory, prefix * "_copy", counter)
        return copy_values(path, source, options)
    end
    push_result = measure(; samples) do
        path = next_path(directory, prefix * "_push", counter)
        return push_values(path, push_source, options)
    end
    load_and_collect_result = measure(() -> load_and_collect(collect_path); samples)

    # Keeping the file open separates value reading from schema loading. The end-to-end
    # measurement above remains useful for short-lived readers, while this one represents
    # repeated analysis of an already loaded vector.
    file = HDF5.h5open(collect_path, "r")
    collect_result = try
        vector = HDF5Vectors.load_hdf5_vector(file["values"])
        measure(() -> collect(vector); samples)
    finally
        close(file)
    end

    return (
        (; name, operation = "copy", measurement = copy_result),
        (; name, operation = "push", measurement = push_result),
        (; name, operation = "load + collect", measurement = load_and_collect_result),
        (; name, operation = "collect loaded", measurement = collect_result),
    )

end

function print_results(results, bulk_count, push_count, samples)

    println(
        "Bulk copy and collect use $bulk_count values; incremental push uses " *
        "$push_count.",
    )
    println("Each result is the median of $samples samples after one warm-up run.")
    println()
    println(
        rpad("workload", 22),
        rpad("operation", 17),
        lpad("time ms", 12),
        lpad("allocated MiB", 16),
    )

    for result in results
        milliseconds = 1_000 * result.measurement.time
        mebibytes = result.measurement.bytes / 2.0^20
        println(
            rpad(result.name, 22),
            rpad(result.operation, 17),
            lpad(round(milliseconds; digits = 2), 12),
            lpad(round(mebibytes; digits = 2), 16),
        )
    end

end

function main()

    bulk_count = 100_000
    push_count = 2_000
    samples = 7
    measurements = [
        BenchmarkMeasurement(
            BenchmarkTimestamp(Int32(index ÷ 100), Int64(index * 1_000)),
            273.15 + index / 100,
            StaticArrays.SVector(index / 10, index / 20, index / 30),
        ) for index in 1:bulk_count
    ]

    # These are the package's primary logging workloads. The test suite covers the wider
    # matrix of supported types and operations, including constants and serialization.
    workloads = (
        ("scalar", Float64.(1:bulk_count), (; portable = true)),
        (
            "dense SVector",
            [StaticArrays.SVector(value, 2value, 3value) for value in 1.0:bulk_count],
            (; portable = true),
        ),
        ("portable record", measurements, (; portable = true)),
    )

    results = NamedTuple[]

    # Every timed write receives a new path. End-to-end collection repeatedly opens the
    # same completed file, while loaded collection keeps one vector open. The temporary
    # directory is removed after all samples finish.
    mktempdir() do directory
        for (name, source, options) in workloads
            append!(
                results,
                benchmark_workload(
                    directory,
                    name,
                    source,
                    options;
                    push_count,
                    samples,
                ),
            )
        end
    end
    print_results(results, bulk_count, push_count, samples)

end

main()
