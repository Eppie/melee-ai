# ML Pipeline Data Flow Documentation

This document traces the complete data flow through the Super Smash Bros Melee ML pipeline, from raw .slp replay files to real-time inference.

## 1. Raw Data Source: .slp Files

### 1.1 File Format
- **Location**: `/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX/` (configured in `config.py` line 31)
- **Format**: UBJSON (Universal Binary JSON) specification Draft 12
- **Structure**: Contains discrete events describing game state changes
- **Key Events**:
  - `0x35`: Event Payloads (enumerates possible events)
  - `0x36`: Game Start (game initialization data)
  - `0x37`: Pre-Frame Update (controller inputs used for current frame)
  - `0x38`: Post-Frame Update (game state results after processing frame)
  - `0x39`: Game End (game termination)

### 1.2 Temporal Structure per Frame
Each frame contains two main data collection points:
1. **Pre-Frame Update** (Event 0x37): Controller inputs applied at frame start
2. **Post-Frame Update** (Event 0x38): Game state results after physics/collision processing

## 2. SLP File Processing (`zarr_storage.py`)

### 2.1 File Discovery and Loading
```python
# Function: main() in zarr_storage.py:382
slp_files = sorted(glob.glob(os.path.join(config.zarr.input_root, "master-master*.slp")))[:1]
```

- Uses `glob` to find SLP files matching pattern `master-master*.slp`
- Currently processes only first file (`[:1]`) for testing

### 2.2 Episode Processing Pipeline
```python
# Function: process_one_episode() in zarr_storage.py:98
def process_one_episode(raw_path: str) -> List[Row]:
    console = Console(is_dolphin=False, allow_old_version=True, path=raw_path)
    # ... frame processing loop
```

**Step-by-step processing:**

1. **Console Initialization**:
   - Creates `Console` object pointing to SLP file
   - `is_dolphin=False` indicates file-based replay parsing
   - `allow_old_version=True` enables compatibility with older replay formats

2. **Frame Processing Loop**:
   ```python
   # Lines 107-133 in process_one_episode()
   while True:
       gamestate = console.step()
       if gamestate is None:
           break
       # ... frame validation and extraction
   ```

   - `console.step()` advances to next frame in replay file
   - Returns `GameState` object containing current frame's data
   - Loop continues until end of replay (`gamestate is None`)

3. **Frame Validation**:
   ```python
   # Lines 113-118 in process_one_episode()
   if len(gamestate.players) < 2:
       continue
   if gamestate.frame < 0:
       continue
   ```

   - Skips frames with insufficient players (< 2)
   - Skips negative frames (pre-game setup frames)

4. **Data Extraction**:
   ```python
   # Lines 119-123 in process_one_episode()
   try:
       row = extract(gamestate)
       rows.append(row)
   except (ValueError, KeyError, AttributeError):
       continue
   ```

   - Calls `extract()` function to convert `GameState` to `Row` object
   - Continues processing if extraction fails (handles corrupted frames)

### 2.3 GameState to Row Conversion
```python
# Function: extract() in zarr_storage.py:23
def extract(game_state: GameState) -> Row:
    # Common fields
    stage: np.int32 = _preprocess_stage(game_state.stage)
    # ... player data extraction
```

**Detailed extraction process:**

1. **Player Port Identification**:
   ```python
   # Lines 27-33 in extract()
   players: list[int] = sorted(game_state.players.keys())
   p1_port: int = players[0]  # First player becomes p1 (ego)
   p2_port: int = players[1]  # Second player becomes p2 (opponent)
   ```

2. **Per-Player Data Extraction**:
   ```python
   # Lines 35-71 in extract()
   def _extract_player_data(port: int) -> dict[str, np.int32 | np.float32 | bool]:
       pl: PlayerState = game_state.players[port]
       cs: ControllerState = pl.controller_state
       # ... field extraction
   ```

   **Extracted fields per player (from PLAYER_SPEC in schema.py):**
   - **Character**: `pl.character` (preprocessed via `_preprocess_character()`)
   - **Position**: `pl.position.x`, `pl.position.y`
   - **Game State**: `pl.percent`, `pl.stock`, `pl.shield_strength`
   - **Action**: `pl.action` (preprocessed via `_preprocess_action()`)
   - **Physics**: `pl.jumps_left`, `pl.facing`, `pl.on_ground`, `pl.invulnerable`
   - **Controller Inputs**: `cs.main_stick[0/1]`, `cs.c_stick[0/1]`, `cs.l_shoulder`
   - **Buttons**: `cs.button` enum values for A, B, X, Y, Z, L, R (processed via helper functions)

3. **Schema Construction**:
   ```python
   # Lines 77-89 in extract()
   row_fields = {
       "stage": stage,
       **{f"p1_{key}": value for key, value in p1_data.items()},
       **{f"p2_{key}": value for key, value in p2_data.items()},
   }
   return Row(**row_fields)
   ```

   - Creates dictionary with stage + p1_ and p2_ prefixed fields
   - Constructs `Row` dataclass instance (defined in schema.py)

### 2.4 Temporal Window Creation
```python
# Function: _rows_to_dense() in zarr_storage.py:213
def _rows_to_dense(rows: Sequence[object], schema: Schema) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    T = len(rows)  # Total frames in episode
    T_out = T - 1  # Output windows (need pairs for temporal prediction)
    F = len(schema.features)  # Feature dimensions
    Yd = len(schema.targets)  # Target dimensions
```

**Temporal shifting mechanism:**
```python
# Lines 235-239 in _rows_to_dense()
for j, name in enumerate(schema.features):
    X[:, j] = np.fromiter((getattr(rows[i], name) for i in range(T_out)), count=T_out, dtype=np.float32)

for j, name in enumerate(schema.targets):
    Y[:, j] = np.fromiter((getattr(rows[i + 1], name) for i in range(T_out)), count=T_out, dtype=np.float32)
```

This creates the temporal prediction task:
- **X[i]**: Features from frame i (current game state)
- **Y[i]**: Targets from frame i+1 (next controller inputs for player 1)

## 3. Zarr Storage System

### 3.1 Episode Storage Format
```python
# Function: write_episode() in zarr_storage.py:167
def write_episode(self, episode_id: int, X: np.ndarray, Y: np.ndarray) -> str:
    ep_name = f"ep_{episode_id:06d}"
    epg = self.root.require_group(ep_name)
    # ... array creation and data writing
```

**Storage structure:**
- **X array**: Shape `(T-1, F)` - features for each window
- **Y array**: Shape `(T-1, Yd)` - targets for each window
- **Chunking**: Configurable chunk sizes for efficient storage and loading

### 3.2 Shard Organization
```python
# Function: _process_shard() in zarr_storage.py:244
def _process_shard(shard_id: int, raw_paths: Sequence[str], schema: Schema) -> ShardResult:
    shard_path = Path(config.zarr.out_root) / f"shard_{shard_id:05d}.zarr"
    writer = EpisodeWriter(schema, str(shard_path))
    # ... episode processing loop
```

- Episodes grouped into shards (default: 100 episodes per shard)
- Each shard is a separate Zarr hierarchy
- Metadata tracking episode IDs, frame counts, data types

### 3.3 Metadata System
```python
# Function: _merge_and_write_metadata() in zarr_storage.py:284
def _merge_and_write_metadata(results: List[ShardResult], schema: Schema) -> None:
    # Index file: maps episode_id -> (shard_id, frames)
    # Lengths file: frames per episode for window calculation
    # Meta file: schema, configuration, timestamps
```

## 4. Dataset Loading (`window_dataset.py`)

### 4.1 Corpus Index
```python
# Class: ZarrCorpusIndex in window_dataset.py:54
def __init__(self, data_dir: str | Path) -> None:
    # Load metadata files
    self.meta = json.load(meta_path.open("r"))
    self.lengths = np.load(lengths_path)  # (E,) frames per episode
    self.wins = np.load(wins_path)  # (E,) windows per episode
```

**Index calculation:**
```python
# Lines 93-95 in ZarrCorpusIndex
self._wins = wins.astype(np.int64)
self._cumwins = np.cumsum(self._wins, dtype=np.int64)
self.total_windows: int = int(self._cumwins[-1])
```

- **Window count per episode**: `max(frames - seq_len + 1, 0)`
- **Cumulative windows**: Enables O(log E) episode lookup
- **Total windows**: Sum across all episodes for dataset size

### 4.2 Window-to-Episode Mapping
```python
# Function: window_to_episode() in window_dataset.py:106
def window_to_episode(self, global_win_idx: int) -> Tuple[int, int]:
    ep_idx = int(np.searchsorted(self._cumwins, global_win_idx, side="right"))
    base = 0 if ep_idx == 0 else int(self._cumwins[ep_idx - 1])
    offset = int(global_win_idx - base)
    return ep_idx, offset
```

- Binary search to find which episode contains target window
- Offset calculation for position within episode

### 4.3 Episode Array Loading
```python
# Function: open_episode_arrays() in window_dataset.py:125
def open_episode_arrays(self, ep: EpisodeInfo, *, cache=None) -> Tuple[zarr.Array, Optional[zarr.Array]]:
    shard_path = self._shard_paths.get(ep.shard_id)
    root = zarr.open_group(str(shard_path), mode="r", path=None)
    ep_name = f"ep_{ep.episode_id:06d}"
    epg = root[ep_name]
    X = epg["X"]  # shape (T, F), float32
    Y = epg.get("Y", None)  # shape (T, Yd) or missing
```

- **LRU Cache**: Per-worker episode caching (default 8 episodes)
- **Memory mapping**: Zarr arrays accessed without full loading

### 4.4 Window Dataset
```python
# Class: WindowDataset in window_dataset.py:212
class WindowDataset(Dataset):
    def __getitem__(self, i: int) -> Dict[str, object]:
        ep_idx, offset = self.index.window_to_episode(i)
        ep = self.index.episodes[ep_idx]
        start = offset
        L = self.seq_len  # 256 by default
        Xa, Ya = self.index.open_episode_arrays(ep, cache=self._cache)
        Xw = Xa[start:start + L, :]  # (L, F)
        Yw = None if Ya is None else Ya[start:start + L, :]  # (L, Yd)
```

**Window slicing:**
- **Xw**: Features for frames [start, start+L-1]
- **Yw**: Targets for frames [start, start+L-1] (shifted by 1 from features)

## 5. Training Pipeline (`train.py`)

### 5.1 DataLoader Creation
```python
# Function: make_dataloader() in window_dataset.py:400
def make_dataloader() -> Tuple[torch.utils.data.DataLoader, WindowDataset, Sampler[int]]:
    ds = WindowDataset(config.zarr.out_root, feature_keep=config.train.feature_keep, target_keep=config.train.target_keep)
    sampler = RandomWindowSampler(ds, replacement=config.train.replacement, num_samples=effective_num_samples)
    loader = torch.utils.data.DataLoader(ds, batch_size=config.train.batch_size, sampler=sampler, ...)
```

### 5.2 Model Input Construction
```python
# Function: build_inputs_for_gptv7() in train.py:185
def build_inputs_for_gptv7(batch_X: torch.FloatTensor, colmap: ColumnMap) -> TensorDict:
    # Categorical embeddings
    stage = batch_X[..., colmap.stage_idx].to(torch.long).unsqueeze(-1)
    ego_character = batch_X[..., colmap.ego_char_idx].to(torch.long).unsqueeze(-1)
    # ... other categorical fields
    gamestate = batch_X[..., colmap.gamestate_idxs]  # [B,L,Gg]
    controller = batch_X[..., colmap.controller_idxs]  # [B,L,Gc]
```

**TensorDict structure:**
- **stage**: Stage embedding indices
- **ego_character/opponent_character**: Character embedding indices
- **ego_action/opponent_action**: Action embedding indices
- **gamestate**: Numeric game state features (position, damage, etc.)
- **controller**: Controller input features (sticks, buttons)

### 5.3 Target Quantization
```python
# Function: quantize_targets() in controller_quantization.py (imported in train.py:19)
target_info = quantize_targets(Y, colmap, config.train.shoulder_centers)
```

- **Main/C-stick**: Quantized to discrete positions (FOX_STICK_64, C_STICK_XY_CLUSTER_CENTERS_V0_1)
- **Buttons**: Binary classification (pressed/not pressed)
- **Shoulder**: Quantized to discrete analog levels

### 5.4 Model Forward Pass
```python
# Function: train_loop() in train.py:652
pred: TensorDict = model(inputs_td)  # keys: buttons, main_stick, c_stick, (shoulder)
```

**GPTv7 Architecture (`gpt.py`):**
1. **Input Embedding**: Categorical embeddings + linear projection of numeric features
2. **Transformer Blocks**: Self-attention with RoPE positional encoding
3. **Output Heads**: Separate heads for each controller component with progressive conditioning

### 5.5 Loss Computation
```python
# Lines 720-766 in train_loop()
# Main stick CE loss
logits_main = pred["main_stick"].reshape(B * L, -1)
target_main = target_info["main_idx"].reshape(B * L)
loss_main = F.cross_entropy(logits_main, target_main, weight=main_weights)

# C-stick CE loss
loss_c = F.cross_entropy(logits_c, target_c, weight=c_weights)

# Buttons BCE loss
loss_btn = F.binary_cross_entropy_with_logits(logits_btn, target_btn, pos_weight=pos_weight)

# Optional shoulder CE loss
if shoulder_enabled:
    loss_s = F.cross_entropy(logits_s, target_s)
```

## 6. Inference Pipeline (`model_interface.py`)

### 6.1 Real-time State Collection
```python
# Function: collect_raw_inputs_from_gamestate() in model_interface.py:195
def collect_raw_inputs_from_gamestate(gamestate: GameState, bot_port: int, opp_port: int) -> Dict[str, float]:
    ego_player = gamestate.players.get(bot_port)
    opp_player = gamestate.players.get(opp_port)
    features = {"stage": gamestate.stage.value}
    features.update(_player_fields(ego_player, "p1"))
    features.update(_player_fields(opp_player, "p2"))
    return features
```

**Real-time data extraction:**
- Same field extraction as training (`extract()` function)
- Player port assignment: bot_port → p1, opp_port → p2
- Real-time constraint: Must complete within ~16ms frame time

### 6.2 Model Input Preparation
```python
# Class: GPTInferenceEngine in model_interface.py:268
def prepare_inputs(self, raw_inputs: Dict[str, float]) -> Optional[TensorDict]:
    frame = self._frame_to_tensor(raw_inputs)
    self.buffer.append(frame)
    batch = self._stack_frames()
    if batch is None:
        return None
    return self._build_inputs(batch)
```

**Buffer management:**
- **History buffer**: Maintains `seq_len` (256) most recent frames
- **Warmup period**: Requires `warmup_frames` (128) frames before prediction
- **Sliding window**: Newest frame replaces oldest in buffer

### 6.3 Prediction and Controller Output
```python
# Function: predict_from_raw() in model_interface.py:406
def predict_from_raw(self, raw_inputs: Dict[str, float]) -> ControllerState:
    inputs_td = self.prepare_inputs(raw_inputs)
    if inputs_td is None or len(self.buffer) < self.warmup_frames:
        return ControllerState.neutral()  # Neutral inputs during warmup
    with torch.inference_mode():
        outputs = self.model(inputs_td)
    return self._decode_outputs(outputs)
```

**Output decoding:**
- **Sticks**: Argmax → discrete position → analog coordinates
- **Buttons**: Sigmoid → threshold → boolean states
- **Shoulder**: Argmax → quantized analog value

### 6.4 Game Integration
```python
# Function: apply_model_outputs_to_game() in model_interface.py:444
def apply_model_outputs_to_game(controller: Controller, model_outputs: ControllerState) -> None:
    controller.release_all()  # Clear previous inputs
    # Apply button presses based on model_outputs
    # Set analog stick positions
    # Set shoulder analog value
```

## 7. Key Temporal Relationships

### 7.1 Training Data Structure
For each window of length L:
- **X[i]**: Game state at frame i (includes controller inputs used for frame i)
- **Y[i]**: Controller inputs for frame i+1 (what should be applied next)

**Temporal prediction task**: "Given current game state (including what inputs I just used), predict what inputs I should use next."

### 7.2 Inference Data Flow
During gameplay:
1. **Frame N**: Game state observed → added to buffer
2. **Buffer contains frames [N-seq_len+1, N]**
3. **Model predicts controller inputs for frame N+1**
4. **Controller inputs applied for frame N+1**
5. **Frame N+1 game state observed** (results of predicted inputs)

## 8. Data Schema Details

### 8.1 Features (X) - All Players
- **Stage**: Current stage ID
- **Player 1 & 2**: Character, position, damage, stock, action, physics state, controller inputs

### 8.2 Targets (Y) - Player 1 Only
- **Controller inputs**: Main stick, C-stick, buttons, shoulder analog
- **Quantized**: Continuous inputs discretized for classification

### 8.3 Temporal Alignment
- **Causal structure**: X[i] can be used to predict Y[i]
- **No future leakage**: Features only contain information available at prediction time

This completes the comprehensive data flow documentation from raw SLP files through the entire ML pipeline.
