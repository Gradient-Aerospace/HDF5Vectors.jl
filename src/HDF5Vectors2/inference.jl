####################
# Schema Inference #
####################

const hdf5_scalar_types = Union{
    Bool,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

"""
Controls default schema inference without becoming part of storage execution.

`portable` selects field-oriented records instead of native HDF5 representations for
nonzero-size bits types. Arrays without declared dimensions and nonconcrete declared types
can use Julia serialization when their corresponding policy fields are true.
"""
struct SchemaPolicy
    portable::Bool
    serialize_arrays::Bool
    serialize_nonconcrete::Bool
end

function SchemaPolicy(;
    portable = true,
    serialize_arrays = true,
    serialize_nonconcrete = true,
)
    return SchemaPolicy(portable, serialize_arrays, serialize_nonconcrete)
end

struct SchemaContext{D}
    policy::SchemaPolicy
    dims::D
end

function unsupported_schema(type, reason)
    throw(ArgumentError("HDF5Vectors2 cannot infer a schema for $type: $reason"))
end

function validate_dims(dims, expected_rank)
    if isnothing(dims)
        return nothing
    elseif !(dims isa Tuple) || length(dims) != expected_rank
        throw(DimensionMismatch(
            "Expected $expected_rank declared dimensions, but got $dims.",
        ))
    elseif !all(dimension -> dimension isa Integer && !(dimension isa Bool), dims)
        throw(ArgumentError("Declared dimensions must be integers; got $dims."))
    elseif !all(dimension -> dimension > 0, dims)
        throw(ArgumentError("Declared dimensions must be positive; got $dims."))
    end
    return Tuple(Int(dimension) for dimension in dims)
end

function reject_dims(type, dims)
    if !isnothing(dims)
        throw(ArgumentError(
            "Dimensions cannot be declared for scalar or record type $type.",
        ))
    end
end

"""
    infer_schema(type::Type; dims = nothing, policy = SchemaPolicy())

Builds a complete storage schema without opening or modifying an HDF5 file. The result is
the representation plan used by later creation, writing, and loading operations.
"""
function infer_schema(
    type::Type;
    dims = nothing,
    policy = SchemaPolicy(),
)
    return infer_schema(type, SchemaContext(policy, dims))
end

"""
    serialization_schema(type::Type)

Builds an explicit Julia-serialization schema for `type`.
"""
function serialization_schema(::Type{T}) where {T}
    return BlobSchema(SerializationCodec{T}())
end

function infer_schema(type::Type{<:hdf5_scalar_types}, context::SchemaContext)
    reject_dims(type, context.dims)
    return ScalarSchema(IdentityCodec{type}())
end

function infer_schema(::Type{String}, context::SchemaContext)
    reject_dims(String, context.dims)
    return ScalarSchema(IdentityCodec{String}())
end

function infer_schema(::Type{Char}, context::SchemaContext)
    reject_dims(Char, context.dims)
    return ScalarSchema(CharCodec())
end

function infer_schema(::Type{Symbol}, context::SchemaContext)
    reject_dims(Symbol, context.dims)
    return ScalarSchema(SymbolCodec())
end

function infer_schema(type::Type{T}, context::SchemaContext) where {H, T <: Enum{H}}
    reject_dims(type, context.dims)
    if !(H <: hdf5_scalar_types)
        return unsupported_schema(type, "its enum base type $H is not HDF5-native.")
    end
    return ScalarSchema(EnumCodec{T, H}())
end

function infer_schema(::Type{Tuple{}}, context::SchemaContext)
    reject_dims(Tuple{}, context.dims)
    value = ()
    return ConstantSchema(ConstantCodec{Tuple{}}(value))
end

function infer_schema(type::Type{NTuple{N, E}}, context::SchemaContext) where {N, E}

    dims = isnothing(context.dims) ? (N,) : validate_dims(context.dims, 1)
    if dims != (N,)
        throw(DimensionMismatch("The dimensions $dims do not match the tuple length $N."))
    elseif iszero(N)
        return ConstantSchema(ConstantCodec{type}(()))
    end

    child_context = SchemaContext(context.policy, nothing)
    element_schema = infer_schema(E, child_context)
    if element_schema isa ScalarSchema
        return DenseSchema(type, dims, element_schema.codec)
    end
    return record_schema(type, context)

end

function infer_schema(type::Type{<:StaticArrays.StaticArray}, context::SchemaContext)

    expected_dims = Tuple(StaticArrays.Size(type))
    dims = isnothing(context.dims) ? expected_dims : validate_dims(
        context.dims,
        length(expected_dims),
    )
    if dims != expected_dims
        throw(DimensionMismatch(
            "The dimensions $dims do not match the static dimensions $expected_dims.",
        ))
    elseif iszero(prod(expected_dims))
        value = type()
        return ConstantSchema(ConstantCodec{type}(value))
    end

    child_context = SchemaContext(context.policy, nothing)
    element_schema = infer_schema(eltype(type), child_context)
    if element_schema isa ScalarSchema
        return DenseSchema(type, dims, element_schema.codec)
    end
    return record_schema(type, SchemaContext(context.policy, nothing))

end

function infer_schema(type::Type{<:Array{E, N}}, context::SchemaContext) where {E, N}

    dims = validate_dims(context.dims, N)
    if isnothing(dims)
        if context.policy.serialize_arrays
            return serialization_schema(type)
        end
        return unsupported_schema(type, "its dimensions were not declared.")
    end

    child_context = SchemaContext(context.policy, nothing)
    element_schema = infer_schema(E, child_context)
    if element_schema isa ScalarSchema
        return DenseSchema(type, dims, element_schema.codec)
    elseif context.policy.serialize_arrays
        return serialization_schema(type)
    end
    return unsupported_schema(type, "its element type does not have a scalar encoding.")

end

function infer_schema(type::Type, context::SchemaContext)

    reject_dims(type, context.dims)

    if !isconcretetype(type)
        if context.policy.serialize_nonconcrete
            return serialization_schema(type)
        end
        return unsupported_schema(type, "the declared type is not concrete.")
    elseif isprimitivetype(type)
        return unsupported_schema(type, "HDF5 does not provide a native datatype for it.")
    end

    names = fieldnames(type)
    if isempty(names)
        return infer_constant_schema(type)
    elseif Base.issingletontype(type)
        # A field-bearing singleton is still a record. Its fields provide a public and
        # inspectable representation even when the type has no zero-argument constructor.
        return record_schema(type, context)
    elseif isbitstype(type) && !context.policy.portable
        return ScalarSchema(IdentityCodec{type}())
    end
    return record_schema(type, context)

end

function infer_constant_schema(type::Type)

    if !Base.issingletontype(type)
        return unsupported_schema(type, "it has no fields but does not have one value.")
    elseif type === NamedTuple{(), Tuple{}}
        value = (;)
    elseif applicable(type)
        value = type()
    else
        return unsupported_schema(
            type,
            "its constant value cannot be reconstructed through a supported interface.",
        )
    end

    if !(value isa type)
        return unsupported_schema(
            type,
            "its zero-argument constructor returned $(typeof(value)).",
        )
    end
    return ConstantSchema(ConstantCodec{type}(value))

end

function record_schema(type::Type, context::SchemaContext)

    field_names = fieldnames(type)
    names = Tuple(string(field_name) for field_name in field_names)
    types = fieldtypes(type)
    children = Tuple(
        infer_schema(field_type, SchemaContext(context.policy, nothing))
        for field_type in types
    )

    codec = if type <: NamedTuple
        NamedTupleCodec{type}()
    elseif type <: Tuple
        TupleCodec{type}()
    elseif type <: StaticArrays.StaticArray
        StaticArrayCodec{type}()
    else
        StructCodec{type, length(field_names)}(field_names)
    end
    return RecordSchema(type, names, codec, children)

end
