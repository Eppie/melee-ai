Goal: move feature transforms offline so training reads pre-transformed data while still retaining raw features in Zarr for debugging/validation. Follow TDD: start by writing tests before code changes.

Plan

1) Testing first (unit + integration)
- Add unit tests covering: (a) new dataset metadata indicating which transforms were baked in and the config used; (b) loader selecting the pre-transformed array and refusing to apply runtime transforms when available; (c) hard failures when transformed data is missing or spec hash mismatches.
- Add integration tests for window loading: generate a tiny synthetic Zarr shard with raw features + expected transformed features; verify WindowDataset yields the transformed values and raises if the transformed array is removed or metadata is wrong.
- Add regression tests around dataset build/inference parity: run the transform functions on sample data, compare to the offline-stored arrays, and ensure round-trip equality with the previous online path.

2) Dataset format updates (design, then implement)
- Decide on storage layout: keep raw features in existing X; add a new array (e.g., X_proc/X_transformed) for transformed features plus metadata in meta.json capturing the transform spec hash and version.
- Define versioning and compatibility rules: how training detects presence/validity of transformed data and refuses to proceed when it is absent or outdated.
- Use strict typing from data_types.py to distinguish raw vs. transformed arrays at the type level (e.g., RawNumpyArray vs. ProcessedNumpyArray) throughout the codepath.

3) Pipeline updates: preprocessing/build
- Update the dataset generation pipeline (Zarr creation) to compute transforms once and store them alongside raw data, writing meta to record the applied spec.
- Ensure memory efficiency: stream transforms per shard/window rather than full dataset in memory.

4) Training/dataloader changes
- Modify WindowDataset to require the pre-transformed array and fail fast/loudly if it is absent or invalid; remove any online-transform path.
- Ensure feature names stay aligned (raw vs transformed) and that shapes/dtypes match previous runtime output.
- Apply strict typing (Raw/Processed wrappers) in the dataloader pipeline to make raw vs transformed handling explicit.

5) Validation/inference scripts
- Update validation/analyze scripts to respect the new pre-transformed data while retaining the ability to inspect raw features for debugging.

6) Migration (no backward compatibility)
- Assume regeneration of datasets/checkpoints; drop online transform fallback entirely.
- Extend the existing zarr_storage.py pipeline to build datasets with offline transforms.

7) Observability and safeguards
- Add sanity checks: verify meta hash of transform spec matches code-config; warn or fail when mismatched.
- Log that training is using offline transforms; if offline data is missing/stale, abort loudly.

8) Documentation
- Update DATA_FLOW.md / README to describe the offline transform flow, new dataset fields, and how to toggle fallback paths.

9) Final validation
- Run the full test suite plus focused dataloader/throughput checks to confirm speed benefits and correctness.
