# Change Log for HDF5Vectors

## Unreleased

### Bug Fixes

* Fixed excessive memory use when saving serialized values, including arrays nested in records, with small logical chunk lengths. Blob byte chunks now have a minimum size of 65536 bytes; logical offset chunks and existing-file compatibility are unchanged.

## v1.0.1

### Bug Fixes

* Fixed a bug where types were recorded as their display names, which changes depending on how they're imported, rather than fully-qualified types.

## v1.0.0

* Added initial functionality, supporting append-only vectors of basic types, structs of basic types, and structs of structs.
