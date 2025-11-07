# Training Data Flow

This document describes the data flow for the Nano-Melee training pipeline, from raw Slippi replay files to the data that is fed to the model.

## 1. Slippi Replay Files (`.slp`)

The training process starts with a collection of Slippi replay files (`.slp`). These files contain the complete game state and controller inputs for every frame of a Melee match. These files are parsed using libraries like `py-slippi` to extract the raw data.

## 2. Preprocessing and Zarr Storage

The parsed data from the `.slp` files is preprocessed and stored in the Zarr format. This is handled by scripts that are not part of the core training loop but are a necessary prerequisite.

The output of this stage is a Zarr corpus, which is a directory containing:
- `shard_*.zarr`: Directories containing the actual data, sharded for efficient access.
- `meta.json`: A file containing metadata about the corpus, such as the sequence length (`seq_len`), feature names, and target names.
- `lengths.npy`: A NumPy array containing the number of frames in each episode.
- `index.jsonl`: A JSONL file that maps each episode to its shard and provides other metadata.

Within each shard, the data is organized into episodes. Each episode now stores both raw and transformed matrices:
- `X_raw` / `Y_raw`: Float32 arrays that preserve the unnormalized features and targets exactly as they were extracted from the replay.
- `X_transformed`: Float32 features with every configured transform (scales, offsets, palette snaps) baked in at build time.
- `Y_quantized`: A single float32 matrix that packs the quantized main-stick indices, C-stick indices, optional shoulder indices, and button probabilities. Offsets and palette sizes are recorded in `meta.json` so readers know how to slice the matrix back into logical components.

## 3. Data Loading and Windowing (`window_dataset.py`)

The training loop loads data from the Zarr corpus using a `torch.utils.data.DataLoader`. The `make_dataloader` function in `window_dataset.py` is the entry point for this process.

### `WindowDataset`

The `WindowDataset` class is a `torch.utils.data.Dataset` that provides access to sliding windows of data from the Zarr corpus.

- **`ZarrCorpusIndex`**: This helper class is used to index the entire Zarr corpus. It reads the metadata files (`meta.json`, `lengths.npy`, `index.jsonl`) to build a mapping from a global window index to a specific episode and a starting frame offset within that episode. This allows for efficient random access to any window in the dataset.

- **`__getitem__(i)`**: This method is called by the `DataLoader` to retrieve a single window of data.
    1. It calls `self.index.window_to_episode(i)` to get the episode and frame offset for the requested window index `i`.
    2. It opens the transformed feature array and the packed quantized target matrix for that episode using `self.index.open_episode_arrays`.
    3. It slices a window of `seq_len` frames from the feature matrix, producing `Xw`.
    4. It slices the aligned quantized target matrix once, then uses the metadata offsets to recover `main_idx`, `c_idx`, `buttons`, and optionally `shoulder_idx`, packaging them into the same dictionary structure that `quantize_controller_targets` historically returned.
    5. Both the feature slice and the quantized target tensors are converted to `torch.Tensor`s and returned to the DataLoader.

## 4. Batching and Training (`train.py`)

The `DataLoader` takes the windows from `WindowDataset` and groups them into batches. The `RandomWindowSampler` is used to ensure that the windows are sampled randomly.

The main training loop in `train.py` then processes these batches:

- **`_prepare_batch`**: This function moves the `X` tensor and the already-quantized `target_info` tensors in the batch to the appropriate device (CPU or GPU).

- **`build_model_inputs`**: This function takes the `X` tensor from the batch and creates a `TensorDict` with a structured representation of the model's inputs. It uses the `ColumnMap` to identify which columns in the `X` tensor correspond to which features.
    - **Categorical Features**: Features like `stage`, `character`, and `action` are cast to `torch.long` to be used with embedding layers.
    - **Continuous Features**: Features like `gamestate` and `controller` (the previous controller state) remain as `torch.float`.

### Data Fed to the Model

The `TensorDict` produced by `build_model_inputs` is what is directly fed to the model. This `TensorDict` contains:
- **Categorical Features**: As integer indices, ready for embedding.
- **Continuous Features**: As floating-point values.
- **Transformed Stick Inputs**: The controller stick inputs in the `X` tensor have been quantized to a discrete palette by the preprocessing pipeline.

Quantized targets arrive directly from the dataset as part of `target_info`, so the loss functions can consume them immediately without re-running controller quantization in the training loop.

## Recommendations for Improvement

Based on the analysis of the data flow, here are some recommendations for potential improvements:

### 1. Explicit Input Domain Metadata

Now that target quantization happens during dataset construction, the corpus metadata is the single source of truth for palette sizes and domains. We should consider persisting the exact `input_domain` that was used (`unit01` vs `unit11`) so downstream consumers (evaluation, inference) never have to guess.

### 2. Profile Indexing/Caching

The `window_dataset.py` file contains a `_LRUEpisodeCache` and a `ZarrCorpusIndex` with `O(log E)` performance. The code includes TODO comments questioning the effectiveness of the cache and the possibility of achieving constant time for the index. If data loading is a bottleneck, it would be beneficial to profile these components to quantify their impact. If the cache is not providing a significant speedup, it could be simplified or removed. Similarly, if indexing is a bottleneck, exploring alternative indexing strategies (e.g., a direct lookup table) could be worthwhile, at the cost of increased memory usage.
