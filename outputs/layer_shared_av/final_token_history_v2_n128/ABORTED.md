# Aborted experimental run

The token-history correction was implemented, but full-query coarse QK used
query_chunk-dependent GEMM row shapes. A previous controlled comparison found
large routing/pose changes when query_chunk changed from128 to1024. Therefore
this run was stopped, not promoted as final validation, and its incomplete
files must not be mixed with completed results. Existing files are retained.

Next version uses canonical coarse-QK row tiles independent of mask/attention
batching and verifies GPU chunk invariance before rerunning validation.
