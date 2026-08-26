# When Writing to HDF5 Fails

HDF5Vectors arranges ordinary validation and bookkeeping so many expected errors occur before the file changes. It cannot make every HDF5 write all-or-nothing, and it does not undo file changes that completed before a later operation failed.

## Errors Detected Before Writing

Values passed to `push!` must already have the vector's declared element type, so a value of the wrong type is rejected before the method begins. Dynamically sized array elements are also checked against their declared dimensions before their dataset is extended.

The specialized bulk-copy implementations for elemental values, fixed-size arrays, and Julia-serialized values prepare or serialize the complete input collection before creating the destination HDF5 group. A conversion, dimension, or serialization error during that preparation therefore does not leave a destination with the requested name. The generic and composite bulk-copy implementations perform multiple writes and cannot provide the same protection for every input error.

## In-Memory Lengths

An HDF5 vector updates its in-memory length only after its value has been written successfully. If a write throws an error, `length(vector)` therefore continues to report the number of completed elements known to that Julia object.

This does not mean the HDF5 file is unchanged. Extensible datasets must sometimes be enlarged before the new value is written. If conversion or the HDF5 write then fails, the dataset can retain that larger extent even though the in-memory length was not increased. Closing and loading that vector again may expose an unwritten fill value or fail while validating its datasets.

## Composite Values Can Be Partially Written

Field-oriented composite storage writes each field to a separate nested HDF5 vector. Suppose a value has fields `time` and `payload`. If writing `time` succeeds and writing `payload` fails, the `time` dataset contains one more value while the `payload` dataset does not. The outer vector's in-memory length is not increased, but the child datasets now have different lengths.

Loading field-oriented storage checks that every field has the same number of values. In this example, [`load_hdf5_vector`](@ref) throws a `DimensionMismatch` rather than constructing elements from mismatched fields. HDF5Vectors does not remove the extra `time` value automatically.

Replacing a composite element also writes its fields one at a time. HDF5Vectors first checks bounds and confirms that every field supports replacement, but a low-level HDF5 failure can still occur after one field was replaced and before the next one was changed.

## Serialized Values Can Be Partially Written

Julia byte serialization stores element bytes and their cumulative ending positions in separate nested vectors. A failure while appending bytes can leave extra bytes without a corresponding ending position. Loading checks that the final ending position agrees with the byte count and throws a `DimensionMismatch` when they differ.

## Process and HDF5 Failures

A process crash, forced termination, loss of power, full disk, filesystem error, or unrecoverable HDF5 error can interrupt any write. For example, the process could stop after an HDF5 dataset is extended but before its new value is written, or after one field of a composite value is stored but before the remaining fields are stored. The resulting HDF5 vector may contain an extra fill value, mismatched field lengths, incomplete serialized bytes, or storage that HDF5 itself cannot open.

HDF5Vectors cannot catch an error when the Julia process no longer exists, and it does not keep a second copy of previous HDF5 state from which to restore the file.

## What to Do After a Write Failure

Validate application-specific constraints before calling `push!`, and prefer [`copy_to_hdf5_vector`](@ref) when a complete collection already exists and fits in memory. Specialized bulk-copy paths can detect more ordinary input errors before creating HDF5 storage.

After a caught write or HDF5 error, do not assume that the affected vector is safe to keep using. Close the file normally when possible, reopen it, and call [`load_hdf5_vector`](@ref) to validate the stored layout. If loading fails or the values do not match the application's expectations, discard and recreate that vector from a trusted source or checkpoint.

For outputs that must survive interruption, keep the source data or periodic checkpoints needed to rebuild the current vector. At the application level, another option is to write a new HDF5 file and replace the previous completed file only after the new file has closed successfully.
