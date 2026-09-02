# Each included file owns a test module and its module-local setup. The same files can
# therefore be run directly with `julia --project=test test/<file>.jl`.
include("schema_and_codecs.jl")
include("format.jl")
include("scalar_and_constant_storage.jl")
include("dense_storage.jl")
include("record_storage.jl")
include("blob_storage.jl")
include("public_vector.jl")
include("supported_types_and_operations.jl")
