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
- `X`: A NumPy array of shape `(num_frames, num_features)` containing the **preprocessed** input features for each frame (sticks quantized to palettes, scaling applied).
- `Y`: A NumPy array of shape `(num_frames, num_targets)` containing the **pre-quantized** target controller labels for each frame:
  - `p1_main_stick_idx`, `p1_c_stick_idx`, `p1_shoulder_idx` (int-like, stored as float32)
  - `p1_button_a`, `p1_button_b`, `p1_button_xy`, `p1_button_z`, `p1_button_lr` (float32 in [0,1])

## 3. Data Loading and Windowing (`window_dataset.py`)

The training loop loads data from the Zarr corpus using a `torch.utils.data.DataLoader`. The `make_dataloader` function in `window_dataset.py` is the entry point for this process.

### `WindowDataset`

The `WindowDataset` class is a `torch.utils.data.Dataset` that provides access to sliding windows of data from the Zarr corpus.

- **`ZarrCorpusIndex`**: This helper class is used to index the entire Zarr corpus. It reads the metadata files (`meta.json`, `lengths.npy`, `index.jsonl`) to build a mapping from a global window index to a specific episode and a starting frame offset within that episode. This allows for efficient random access to any window in the dataset.

- **`__getitem__(i)`**: This method is called by the `DataLoader` to retrieve a single window of data.
    1. It calls `self.index.window_to_episode(i)` to get the episode and frame offset for the requested window index `i`.
    2. It opens the preprocessed `X` and `Y` Zarr arrays for that episode using `self.index.open_episode_arrays`.
    3. It slices a window of `seq_len` frames from the `X` and `Y` arrays, resulting in `Xw` and `Yw`.
    4. **No runtime transforms**: `Xw` and `Yw` are already preprocessed/quantized and are converted directly to `torch.Tensor`s. If preprocessing metadata is missing, dataset construction fails fast.

## 4. Batching and Training (`train.py`)

The `DataLoader` takes the windows from `WindowDataset` and groups them into batches. The `RandomWindowSampler` is used to ensure that the windows are sampled randomly.

The main training loop in `train.py` then processes these batches:

- **`_prepare_batch`**: This function moves the `X` and `Y` tensors in the batch to the appropriate device (CPU or GPU).

- **`build_model_inputs`**: This function takes the `X` tensor from the batch and creates a `TensorDict` with a structured representation of the model's inputs. It uses the `ColumnMap` to identify which columns in the `X` tensor correspond to which features.
    - **Categorical Features**: Features like `stage`, `character`, and `action` are cast to `torch.long` to be used with embedding layers.
    - **Continuous Features**: Features like `gamestate` and `controller` (the previous controller state) remain as `torch.float`.

- **Targets**: Batches already contain quantized target indices and button labels from the dataset. The training/validation loops no longer call `controller_quantization.quantize_targets`; they consume the stored labels directly and will raise if the dataset is not preprocessed.

### Data Fed to the Model

The `TensorDict` produced by `build_model_inputs` is what is directly fed to the model. This `TensorDict` contains:
- **Categorical Features**: As integer indices, ready for embedding.
- **Continuous Features**: As floating-point values.
- **Transformed Stick Inputs**: The controller stick inputs in the `X` tensor have been quantized to a discrete palette by the `stick_palette` transform.

The quantized targets stored in the dataset are used to compute the loss against the model's output logits without further transformation.
