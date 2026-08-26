# This benchmark compares representative public operations without making performance a CI
# assertion. It can be run from the package root with:
#
#     julia --project=. benchmark/hdf5vectors2_comparison.jl

include(joinpath(@__DIR__, "..", "src", "HDF5Vectors2", "HDF5Vectors2.jl"))

import .HDF5Vectors2
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

function copy_existing(path, source, options)
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

function copy_prototype(path, source, options)
    return HDF5.h5open(path, "w") do file
        vector = HDF5Vectors2.copy_to_hdf5_vector(
            file["/"],
            "values",
            source;
            chunk_length = 256,
            options...,
        )
        return length(vector)
    end
end

function push_existing(path, source, options)
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

function push_prototype(path, source, options)
    return HDF5.h5open(path, "w") do file
        vector = HDF5Vectors2.create_hdf5_vector(
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

function collect_existing(path)
    return HDF5.h5open(path, "r") do file
        return collect(HDF5Vectors.load_hdf5_vector(file["values"]))
    end
end

function collect_prototype(path)
    return HDF5.h5open(path, "r") do file
        return collect(HDF5Vectors2.load_hdf5_vector(file["values"]))
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
    existing_collect_path = next_path(directory, prefix * "_existing_collect", counter)
    prototype_collect_path = next_path(directory, prefix * "_prototype_collect", counter)
    copy_existing(existing_collect_path, source, options)
    copy_prototype(prototype_collect_path, source, options)

    # Correctness is checked before timing so a fast but incorrect implementation cannot
    # produce a persuasive benchmark result.
    existing_values = collect_existing(existing_collect_path)
    prototype_values = collect_prototype(prototype_collect_path)
    @assert existing_values == source
    @assert prototype_values == source

    push_source = source[1:min(push_count, length(source))]
    existing_copy = measure(; samples) do
        path = next_path(directory, prefix * "_existing_copy", counter)
        return copy_existing(path, source, options)
    end
    prototype_copy = measure(; samples) do
        path = next_path(directory, prefix * "_prototype_copy", counter)
        return copy_prototype(path, source, options)
    end
    existing_push = measure(; samples) do
        path = next_path(directory, prefix * "_existing_push", counter)
        return push_existing(path, push_source, options)
    end
    prototype_push = measure(; samples) do
        path = next_path(directory, prefix * "_prototype_push", counter)
        return push_prototype(path, push_source, options)
    end
    existing_collect = measure(() -> collect_existing(existing_collect_path); samples)
    prototype_collect = measure(() -> collect_prototype(prototype_collect_path); samples)

    return (
        (; name, operation = "copy", existing = existing_copy, prototype = prototype_copy),
        (; name, operation = "push", existing = existing_push, prototype = prototype_push),
        (;
            name,
            operation = "collect",
            existing = existing_collect,
            prototype = prototype_collect,
        ),
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
        rpad("operation", 12),
        lpad("existing ms", 14),
        lpad("prototype ms", 14),
        lpad("time ratio", 14),
        lpad("existing MiB", 14),
        lpad("prototype MiB", 14),
        lpad("alloc ratio", 14),
    )

    for result in results
        existing_ms = 1_000 * result.existing.time
        prototype_ms = 1_000 * result.prototype.time
        existing_mib = result.existing.bytes / 2.0^20
        prototype_mib = result.prototype.bytes / 2.0^20
        println(
            rpad(result.name, 22),
            rpad(result.operation, 12),
            lpad(round(existing_ms; digits = 2), 14),
            lpad(round(prototype_ms; digits = 2), 14),
            lpad(round(prototype_ms / existing_ms; digits = 2), 14),
            lpad(round(existing_mib; digits = 2), 14),
            lpad(round(prototype_mib; digits = 2), 14),
            lpad(round(prototype_mib / existing_mib; digits = 2), 14),
        )
    end

end

function main()

    bulk_count = 5_000
    push_count = 500
    samples = 3
    measurements = [
        BenchmarkMeasurement(
            BenchmarkTimestamp(Int32(index ÷ 100), Int64(index * 1_000)),
            273.15 + index / 100,
            StaticArrays.SVector(index / 10, index / 20, index / 30),
        ) for index in 1:bulk_count
    ]

    # These workloads cover every physical storage shape and both representations of
    # ordinary bits structs. The public comparison tests contain the wider type matrix.
    workloads = (
        ("scalar", Float64.(1:bulk_count), (; portable = true)),
        (
            "dense SVector",
            [StaticArrays.SVector(value, 2value, 3value) for value in 1.0:bulk_count],
            (; portable = true),
        ),
        ("portable record", measurements, (; portable = true)),
        ("native record", measurements, (; portable = false)),
        (
            "serialized array",
            [
                Float64[index + offset for offset in 1:(index % 8 + 1)]
                for index in 1:bulk_count
            ],
            (; portable = true),
        ),
        ("constant", fill(Val(:ready), bulk_count), (; portable = true)),
    )

    results = NamedTuple[]

    # Every timed write receives a new path, while collect repeatedly opens the same
    # completed file. The temporary directory is removed after all samples finish.
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
