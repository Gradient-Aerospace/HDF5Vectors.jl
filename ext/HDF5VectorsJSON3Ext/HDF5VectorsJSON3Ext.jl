module HDF5VectorsJSON3Ext

################################
# Types that Serialize to JSON #
################################

import JSON3
import HDF5Vectors

function HDF5Vectors.encode_value(
    ::HDF5Vectors.JSONCodec{T},
    value::T,
) where {T}
    return JSON3.write(value)
end

function HDF5Vectors.decode_value(
    ::HDF5Vectors.JSONCodec{T},
    value::String,
) where {T}
    return JSON3.read(value, T)
end

end
