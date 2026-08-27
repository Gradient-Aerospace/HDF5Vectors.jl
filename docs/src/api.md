# API Reference

```@meta
CurrentModule = HDF5Vectors
```

## Creating, Copying, and Loading

These functions form the ordinary user interface.

```@docs
create_hdf5_vector
copy_to_hdf5_vector
load_hdf5_vector
```

## Working With HDF5 Vectors

```@docs
AbstractHDF5Vector
HDF5Vector
iterable
```

HDF5 vector implementations also provide the applicable `AbstractVector` operations described in [Common Vector Operations](index.md#Common-Vector-Operations).

## Customizing Element Storage

These hooks select an existing representation or convert between a Julia value and its stored representation.

```@docs
storage_style
construct
deconstruct
```

### Built-In Storage Styles

```@docs
ElementalStorageStyle
ArrayStorageStyle
CompositeStorageStyle
SingletonStorageStyle
ByteArrayStorageStyle
JSONStorageStyle
```

## Implementing a Storage Backend

These interfaces are intended for packages that provide an entirely new HDF5 vector representation.

```@docs
AbstractHDF5VectorStorageStyle
store_metadata
supports_setindex
```
