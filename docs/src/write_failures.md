# When Writing to HDF5 Fails

HDF5Vectors performs type checks, schema inference, and logical encoding before changing physical storage. This prevents many ordinary input errors from leaving partial data behind. HDF5 itself does not make a group of dataset writes all-or-nothing, however, and HDF5Vectors does not attempt to undo writes that completed before a later low-level operation failed.

## Errors Detected Before Physical Writes

The declared element type is enforced by dispatch, so a value of the wrong type is rejected before `push!` begins. `push!` then encodes the complete logical value before extending any dataset. This includes dimension checks, custom codecs, record decomposition, and Julia serialization.

`copy_to_hdf5_vector` validates the destination and schema and encodes the complete input vector before creating the destination group. A codec, dimension, or serialization error during this preparation therefore leaves no child with the requested name.

These guarantees cover errors in ordinary Julia conversion and validation. They cannot prevent a low-level failure after HDF5 mutation begins.

## Logical Counts

Every vector records its logical length in `metadata/count`. A successful append writes the encoded value first, persists the new count second, and updates the Julia object's in-memory count last.

If the value write fails, `length(vector)` continues to report the previous count. The physical dataset may nevertheless have been extended before the failure. If the value succeeds but writing `metadata/count` fails, the physical data can contain one more value than the recorded logical count. Loading checks these lengths and rejects a mismatch.

## Records Can Be Partially Written

Field-oriented records write each field to a separate physical store. Encoding prepares every field first, but the HDF5 writes still happen one child at a time.

Suppose a record contains `time` and `payload`. If the `time` write succeeds and the `payload` write encounters an HDF5 error, those child stores have different lengths. The outer count is not advanced. Loading checks the child lengths and throws a `DimensionMismatch` instead of combining mismatched fields.

The same limitation applies to a bulk copy after its destination has been created. All input values have already been encoded, but a filesystem or HDF5 failure can interrupt initialization between child stores.

## Blob Data Can Be Partially Written

Blob storage uses separate `bytes` and `stops` datasets. A failure can leave appended bytes without their cumulative ending position, or an ending position without an updated logical count. Loading checks that the final stop agrees with the total byte count and that the physical value count agrees with `metadata/count`.

## Process and Filesystem Failures

A process crash, forced termination, loss of power, full disk, filesystem error, or unrecoverable HDF5 error can interrupt any physical write. For example, the process might stop after a dataset is extended but before the new value is written, or after one record field is stored but before its siblings are stored. The result may contain an extra fill value, mismatched field lengths, incomplete blob data, or an HDF5 object that cannot be opened.

HDF5Vectors cannot catch an error after the Julia process has stopped, and it does not keep a second copy of prior HDF5 state from which to restore the file.

## Recovering After a Failure

After a caught HDF5 write error, the affected vector should not be assumed safe for further use. When possible, the file can be closed and reopened, and [`load_hdf5_vector`](@ref) can validate the stored schema, datasets, and logical count. If loading fails or the stored values do not meet the application's expectations, the safest recovery is to recreate that vector from a trusted source or checkpoint.

Applications that must survive interruption can retain their source data or periodic checkpoints. Another useful pattern is to write a new HDF5 file and replace the previous completed file only after the new file has closed successfully.
