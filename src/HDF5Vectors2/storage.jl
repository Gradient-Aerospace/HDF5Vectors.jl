###################
# Store Interface #
###################

# A store contains open HDF5 objects but no logical type conversion. Its methods read and
# write values that have already passed through the corresponding schema. This boundary
# keeps validation and codec failures out of the physical mutation layer.
abstract type AbstractStore end

# Store mutation deliberately follows the two ways an HDF5Vector can grow. A bulk copy
# calls `initialize_encoded!` exactly once on a newly created empty store. Later `push!`
# calls pass the next logical index to `append_encoded!`. Stores do not provide replacement,
# arbitrary-position batch writes, or truncation because the public vector is append-only.

function validate_chunk_length(chunk_length)
    if !(chunk_length isa Integer) || chunk_length isa Bool || chunk_length <= 0
        throw(ArgumentError(
            "The HDF5 chunk length must be a positive integer; got $chunk_length.",
        ))
    end
    return Int(chunk_length)
end

function validate_store_children(group::HDF5.Group, expected_names)
    stored_names = Set(String(name) for name in keys(group))
    expected_names = Set(String(name) for name in expected_names)
    if stored_names != expected_names
        throw(ArgumentError(
            "Stored data children $stored_names do not match $expected_names.",
        ))
    end
    return nothing
end

# HDF5.jl reports variable-length strings as Cstring and compound structs as structural
# NamedTuples when `eltype` is queried. Comparing HDF5 datatype objects validates their
# actual physical representations without mistaking either case for a schema mismatch.
function dataset_matches_encoded_type(dataset::HDF5.Dataset, encoded_type::Type)
    stored_datatype = HDF5.datatype(dataset)
    expected_datatype = HDF5.datatype(encoded_type)
    try
        return stored_datatype == expected_datatype
    finally
        close(stored_datatype)
        close(expected_datatype)
    end
end

# Most stores already return a vector of encoded logical values for a range. Dense storage
# specializes this operation because its natural batch is one higher-dimensional array.
function read_encoded_batch(store::AbstractStore, indices::UnitRange{Int})
    return read_encoded(store, indices)
end

function validate_encoded_column_count(values, expected_count)
    if length(values) != expected_count
        throw(DimensionMismatch(
            "An encoded column has $(length(values)) values instead of $expected_count.",
        ))
    end
    return nothing
end
