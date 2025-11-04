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

Within each shard, the data is organized into episodes. Each episode has two main arrays:
- `X`: A NumPy array of shape `(num_frames, num_features)` containing the input features for each frame.
- `Y`: A NumPy array of shape `(num_frames, num_targets)` containing the target controller inputs for each frame.

## 3. Data Loading and Windowing (`window_dataset.py`)

The training loop loads data from the Zarr corpus using a `torch.utils.data.DataLoader`. The `make_dataloader` function in `window_dataset.py` is the entry point for this process.

### `WindowDataset`

The `WindowDataset` class is a `torch.utils.data.Dataset` that provides access to sliding windows of data from the Zarr corpus.

- **`ZarrCorpusIndex`**: This helper class is used to index the entire Zarr corpus. It reads the metadata files (`meta.json`, `lengths.npy`, `index.jsonl`) to build a mapping from a global window index to a specific episode and a starting frame offset within that episode. This allows for efficient random access to any window in the dataset.

- **`__getitem__(i)`**: This method is called by the `DataLoader` to retrieve a single window of data.
    1. It calls `self.index.window_to_episode(i)` to get the episode and frame offset for the requested window index `i`.
    2. It opens the `X` and `Y` Zarr arrays for that episode using `self.index.open_episode_arrays`.
    3. It slices a window of `seq_len` frames from the `X` and `Y` arrays, resulting in `Xw` and `Yw`.
    4. **Feature Transformation**: It calls `_apply_feature_transforms` on `Xw` (the input features). This function applies a series of transformations defined in `config.py`, such as:
        - `stick_palette`: This transform takes the continuous `[0, 1]` stick values, converts them to the `[-1, 1]` domain, and then snaps them to the nearest value in a discrete palette (e.g., `CONTROL_STICK_QUANTIZED`). This is a crucial step for preparing the input for the model.
    5. The `Yw` tensor (targets) is **not** transformed. It remains in its original continuous domain (typically `[0, 1]` for controller inputs).
    6. The `Xw` and `Yw` NumPy arrays are converted to `torch.Tensor`s.

## 4. Batching and Training (`train.py`)

The `DataLoader` takes the windows from `WindowDataset` and groups them into batches. The `RandomWindowSampler` is used to ensure that the windows are sampled randomly.

The main training loop in `train.py` then processes these batches:

- **`_prepare_batch`**: This function moves the `X` and `Y` tensors in the batch to the appropriate device (CPU or GPU).

- **`build_model_inputs`**: This function takes the `X` tensor from the batch and creates a `TensorDict` with a structured representation of the model's inputs. It uses the `ColumnMap` to identify which columns in the `X` tensor correspond to which features.
    - **Categorical Features**: Features like `stage`, `character`, and `action` are cast to `torch.long` to be used with embedding layers.
    - **Continuous Features**: Features like `gamestate` and `controller` (the previous controller state) remain as `torch.float`.

- **`quantize_controller_targets`**: This function takes the `Y` tensor from the batch and quantizes the continuous target controller inputs into discrete indices.
    - It calls `controller_quantization.quantize_targets`, which expects the stick values in the `Y` tensor to be continuous (in the `[-1, 1]` or `[0, 1]` domain).
    - This function is responsible for converting the continuous target stick values into the discrete indices that are used for calculating the cross-entropy loss.

### Data Fed to the Model

The `TensorDict` produced by `build_model_inputs` is what is directly fed to the model. This `TensorDict` contains:
- **Categorical Features**: As integer indices, ready for embedding.
- **Continuous Features**: As floating-point values.
- **Transformed Stick Inputs**: The controller stick inputs in the `X` tensor have been quantized to a discrete palette by the `stick_palette` transform.

The quantized targets from `quantize_controller_targets` are used to compute the loss against the model's output logits.

## Recommendations for Improvement

Based on the analysis of the data flow, here are some recommendations for potential improvements:

### 1. Explicit Input Domain for Quantization

In `controller_quantization.py`, the `quantize_targets` function has an `input_domain="auto"` option, which tries to infer the domain of the input stick values. The code itself has a TODO comment: `auto should not be needed`. Relying on automatic detection can be brittle. It would be more robust to explicitly set the `input_domain` to either `"unit01"` or `"unit11"` based on how the data is stored in the Zarr corpus. This would make the code more predictable and less prone to errors.

### 2. Consolidate Target Quantization

Currently, feature transformations (including `stick_palette`) are applied in `WindowDataset`, and then target quantization is handled separately in `quantize_controller_targets`. While the issue of applying `stick_palette` to targets has been fixed, it highlights a potential area for simplification. A clearer design would be to have a single, well-defined place where all target processing occurs. This could involve moving all target-related logic into `quantize_controller_targets` and ensuring that `WindowDataset` only deals with loading the raw data.

### 3. Profile Caching and Indexing

The `window_dataset.py` file contains a `_LRUEpisodeCache` and a `ZarrCorpusIndex` with `O(log E)` performance. The code includes TODO comments questioning the effectiveness of the cache and the possibility of achieving constant time for the index. If data loading is a bottleneck, it would be beneficial to profile these components to quantify their impact. If the cache is not providing a significant speedup, it could be simplified or removed. Similarly, if indexing is a bottleneck, exploring alternative indexing strategies (e.g., a direct lookup table) could be worthwhile, at the cost of increased memory usage.
