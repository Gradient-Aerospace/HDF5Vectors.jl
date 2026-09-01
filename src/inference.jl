####################
# Schema Inference #
####################

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

    for (name, value) in (
        ("portable", portable),
        ("serialize_arrays", serialize_arrays),
        ("serialize_nonconcrete", serialize_nonconcrete),
    )
        if !(value isa Bool)
            throw(ArgumentError("The $name option must be Bool; got $(repr(value))."))
        end
    end
    return SchemaPolicy(portable, serialize_arrays, serialize_nonconcrete)

end

struct SchemaContext{D}
    policy::SchemaPolicy
    dims::D
end

function unsupported_schema(type, reason)
    throw(ArgumentError("HDF5Vectors cannot infer a schema for $type: $reason"))
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

Applications can extend this function for a logical type and return a schema composed from
the built-in physical representations. Recursive inference always returns through this
public method, so the extension applies equally to a root vector, a record field, or a
dense element type.

For example, a `Grade` stored as one `UInt8` needs only a codec and this selection method:

```julia
struct GradeCodec <: HDF5Vectors.AbstractCodec{Grade, UInt8} end

HDF5Vectors.infer_schema(::Type{Grade}; kwargs...) =
    HDF5Vectors.ScalarSchema(GradeCodec())
```

The codec implements [`encode_value`](@ref) and [`decode_value`](@ref). Its concrete type is
stored with the schema, so no package-owned codec-name registry is needed when loading.
"""
function infer_schema(
    type::Type;
    dims = nothing,
    policy = SchemaPolicy(),
)
    return infer_builtin_schema(type, SchemaContext(policy, dims))
end

# Recursive inference deliberately returns through the public, one-argument interface.
# This permits a codec selected for an application type to work in exactly the same way at
# the root of a vector, in a record field, or as the element codec of dense storage.
function infer_child_schema(type::Type, context::SchemaContext)
    return infer_schema(
        type;
        dims = context.dims,
        policy = context.policy,
    )
end

function infer_builtin_schema(type::Type, context::SchemaContext)

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
        return native_scalar_schema(type)
    end
    return record_schema(type, context)

end
