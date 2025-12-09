  ---
  Refined Future Position Prediction Model Design

  Major Revisions

  1. Horizon Conditioning Architecture (CRITICAL FIX)

  Original weakness: Concatenating horizon embedding to final hidden state, then using shallow MLP to extrapolate 1-60 frames. This asks too much of the MLP.

  Refined approach: Deeper, Horizon-Aware Output Head

  ┌─────────────────────────────────────────────────────────────┐
  │                   TEMPORAL ENCODER                           │
  │                  (unchanged from before)                     │
  │                           │                                  │
  │                  [last frame: h_context]                     │
  └───────────────────────────┼──────────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────────┐
  │              TRAJECTORY OUTPUT HEAD (ResNet-style)           │
  │                                                              │
  │  Input: h_context [192] + horizon_embed [32] = [224]        │
  │                           │                                  │
  │         ┌─────────────────▼─────────────────┐               │
  │         │   Linear(224 → 384) + ReLU        │               │
  │         └─────────────────┬─────────────────┘               │
  │                           │                                  │
  │         ┌─────────────────▼─────────────────┐               │
  │         │   ResBlock(384): MLP + Residual   │               │
  │         │   • Linear(384 → 768) + ReLU      │               │
  │         │   • Linear(768 → 384)             │               │
  │         │   • Skip connection + ReLU        │               │
  │         └─────────────────┬─────────────────┘               │
  │                           │                                  │
  │         ┌─────────────────▼─────────────────┐               │
  │         │   ResBlock(384): Same structure   │               │
  │         └─────────────────┬─────────────────┘               │
  │                           │                                  │
  │         ┌─────────────────▼─────────────────┐               │
  │         │   Output Projection               │               │
  │         │   → P1: 6×5=30 mixture params     │               │
  │         │   → P2: 6×5=30 mixture params     │               │
  │         └───────────────────────────────────┘               │
  └──────────────────────────────────────────────────────────────┘

  Why this works:
  - The ResBlocks give depth for complex trajectory extrapolation
  - Skip connections stabilize gradient flow
  - The network can learn horizon-specific physics (short-term: momentum, long-term: gravity dominates)

  Updated parameter count:
  - Projection: 224 → 384: ~86K
  - ResBlock 1: 384 → 768 → 384: ~590K
  - ResBlock 2: ~590K
  - Output: 384 → 60: ~23K
  - Total output head: ~1.3M params
  - Full model: ~2.2M params

  Still comfortably <3ms inference on RTX 4090.

  ---
  2. MDN Stability Improvements

  Coordinate System Decision:
  - Use raw game units (e.g., X ∈ [-85, 85] for Final Destination)
  - Predict position DELTA (offset from current position) rather than absolute position
    - Easier to learn: "Fox falls 15 units in 30 frames" vs "Fox will be at y=-25"
    - Multi-modal deltas are smaller (modes cluster around mean movement)
    - Better generalization across stage positions

  Sigma Parameterization (Stability Fix):

  # OLD (unstable):
  sigma = torch.exp(log_sigma).clamp(min=1.0, max=50.0)

  # NEW (stable):
  sigma = F.softplus(log_sigma) + 0.5  # softplus(x) = log(1 + exp(x))
  # Then clamp: sigma.clamp(min=1.0, max=40.0)

  Why softplus is better:
  - exp(-10) = 4.5e-5 → near-zero sigma → NaN
  - softplus(-10) = 4.5e-5, but softplus(-5) = 0.0067, softplus(0) = 0.69
  - Gradient of softplus is sigmoid(x), which is well-behaved
  - Adding epsilon=0.5 ensures minimum sigma even if network outputs -∞

  Normalization Strategy:
  # Input normalization (for positions, velocities):
  # - Stage-relative: subtract stage center, divide by stage half-width
  # - X: normalize by ~90 (typical stage half-width)
  # - Y: normalize by ~70 (typical useful height)
  # This puts most values in [-1, 1]

  # For mixture output:
  # - Predict normalized deltas
  # - Sigma in normalized units: min=0.01 (tiny), max=0.5 (large uncertainty)
  # - Denormalize for visualization

  ---
  3. Trajectory Consistency: Multi-Horizon Prediction

  Problem: Independent predictions for h=10, h=30, h=50 can violate physics (e.g., go left at h=30, right at h=50).

  Solution: Predict keyframe horizons simultaneously

  Architecture change:
  class TrajectoryOutputHead(nn.Module):
      def __init__(self):
          self.horizons = [5, 10, 20, 30, 40, 50, 60]  # 7 keyframes
          # Shared ResNet trunk
          self.trunk = nn.Sequential(...)
          # Separate head per horizon (could also be shared)
          self.p1_heads = nn.ModuleList([
              MixtureDensityHead() for _ in self.horizons
          ])
          self.p2_heads = nn.ModuleList([
              MixtureDensityHead() for _ in self.horizons
          ])

      def forward(self, h_context):
          # h_context: [B, 192]
          features = self.trunk(h_context)  # [B, 384]

          p1_trajectories = []
          p2_trajectories = []
          for i, h in enumerate(self.horizons):
              # Inject horizon-specific embedding
              h_embed = self.horizon_embed(torch.tensor([h]))
              h_cond = torch.cat([features, h_embed.expand(B, -1)], dim=-1)

              p1_mix = self.p1_heads[i](h_cond)
              p2_mix = self.p2_heads[i](h_cond)
              p1_trajectories.append(p1_mix)
              p2_trajectories.append(p2_mix)

          return p1_trajectories, p2_trajectories  # List of 7 mixture params

  Training:
  # Each sample has 7 targets (ground truth at each keyframe horizon)
  loss = 0
  for i, h in enumerate(horizons):
      loss += mixture_nll(p1_traj[i], target_p1[:, h])
      loss += mixture_nll(p2_traj[i], target_p2[:, h])
  loss = loss / len(horizons)

  Benefits:
  1. Single forward pass predicts entire trajectory arc
  2. Shared trunk learns common physics
  3. Horizon-specific heads specialize (short-term: inputs matter, long-term: gravity dominates)
  4. Can add consistency loss between adjacent horizons (optional regularization)

  Parameter impact:
  - 7 heads × 2 players × 30 params = 420 output params (vs 60 before)
  - But shared trunk amortizes cost
  - Negligible inference slowdown

  ---
  4. Input Features Revision

  Features to REMOVE:

  - ❌ Action Frame Counter - Noisy, hard to interpret across 400+ action states
  - ❌ Hitstun Remaining - Noisy, redundant with action state + history

  Features to ADD:

  - ✅ Shoulder Analog - max(L_analog, R_analog) for wavedash/powershield timing
  - ✅ Button Press Onsets - Distinguish tap vs hold
  button_fresh_press = (buttons[t] > 0) & (buttons[t-1] == 0)  # 5 binary flags
  - ✅ Input Deltas - Change in analog inputs from previous frame
  joystick_x_delta = joystick_x[t] - joystick_x[t-1]
  joystick_y_delta = joystick_y[t] - joystick_y[t-1]
  cstick_x_delta = cstick_x[t] - cstick_x[t-1]
  cstick_y_delta = cstick_y[t] - cstick_y[t-1]
  shoulder_delta = shoulder[t] - shoulder[t-1]
  - ✅ Distance to Stage Features
  distance_to_left_ledge = x - left_ledge_x
  distance_to_right_ledge = right_ledge_x - x
  distance_to_stage_bottom = y - 0  # stage surface is y=0
  distance_to_top_blastzone = top_blastzone - y

  Revised Feature List (Per Player, Per Frame)

  | Feature                                                                           | Count | Notes                        |
  |-----------------------------------------------------------------------------------|-------|------------------------------|
  | Position/Motion                                                                   |       |                              |
  | X, Y position                                                                     | 2     | Normalized by stage bounds   |
  | X, Y velocity                                                                     | 2     | Derived from position deltas |
  | Facing direction                                                                  | 1     | -1 or +1                     |
  | Action State                                                                      |       |                              |
  | Action state ID                                                                   | 1     | Embedded (dim 24)            |
  | Grounded/airborne                                                                 | 1     | Binary                       |
  | Combat State                                                                      |       |                              |
  | Percent                                                                           | 1     | Normalized / 300             |
  | Shield size                                                                       | 1     | Normalized / 60              |
  | Jumps remaining                                                                   | 1     | Normalized by max jumps      |
  | Controller Inputs                                                                 |       |                              |
  | Joystick X, Y                                                                     | 2     | [-1, 1]                      |
  | C-stick X, Y                                                                      | 2     | [-1, 1]                      |
  | Shoulder analog (max)                                                             | 1     | [0, 1]                       |
  | Buttons (A, B, X/Y, Z, L/R)                                                       | 5     | Binary                       |
  | Input Changes                                                                     |       |                              |
  | Button press onsets                                                               | 5     | Binary (0→1 transition)      |
  | Joystick delta X, Y                                                               | 2     | Change from t-1              |
  | C-stick delta X, Y                                                                | 2     | Change from t-1              |
  | Shoulder delta                                                                    | 1     | Change from t-1              |
  | Stage Proximity                                                                   |       |                              |
  | Dist to left ledge                                                                | 1     | Can be negative (offstage)   |
  | Dist to right ledge                                                               | 1     | Can be negative              |
  | Dist to stage floor                                                               | 1     | y position essentially       |
  | Dist to top blastzone                                                             | 1     | Remaining vertical space     |
  | Character Physics (constants)                                                     |       |                              |
  | Gravity, Terminal Vel, FastFall, AirSpeed, AirFriction, MaxWalk, Friction, InitDJ | 8     | From characterdata.csv       |

  Continuous features per player: ~44
  Embedded: character (16) + action_state (24) = 40
  Total per player per frame: ~84

  Global per frame:
  - Stage ID (embedded: 8)
  - Relative position (p2 - p1): 2
  - Distance between players: 1

  Grand total per frame: ~180 dims after embedding

  For 16 frames: 16 × 180 = 2,880 input features

  ---
  5. Data Storage: NPZ Instead of Zarr

  User suggestion: Use compressed npy/npz

  Revised storage format:

  data/
  ├── episodes/
  │   ├── episode_00000.npz
  │   ├── episode_00001.npz
  │   └── ...
  └── index.json

  Per-episode .npz structure:
  np.savez_compressed(
      'episode_00000.npz',
      features=features,        # [n_frames, feat_dim], float32 or bf16
      future_deltas=deltas,     # [n_frames, 7, 4], float32 (7 horizons)
      valid_mask=mask,          # [n_frames, 7], bool (mark end-of-episode)
      metadata={
          'stage': stage_id,
          'p1_char': char_id,
          'p2_char': char_id,
          'num_frames': n,
      }
  )

  Pros of NPZ over Zarr:
  - Simpler: each episode is self-contained
  - Fast random access: load entire episode into RAM (most are <5K frames = <5MB)
  - Compressed: saves disk space
  - Easier debugging: can inspect single files

  Cons:
  - Less efficient for partial reads (loads entire episode)
  - No chunking

  For Melee replays (~2-5K frames each), NPZ is ideal. Load full episodes into RAM, they're tiny.

  DataLoader strategy:
  class NPZDataset(Dataset):
      def __init__(self, data_dir):
          self.episodes = list(Path(data_dir).glob('*.npz'))
          # Pre-compute sample indices
          self.samples = []
          for ep_path in self.episodes:
              ep = np.load(ep_path)
              n_frames = len(ep['features'])
              for frame in range(16, n_frames - 60):  # Valid context + future
                  self.samples.append((ep_path, frame))

      def __getitem__(self, idx):
          ep_path, frame = self.samples[idx]
          ep = np.load(ep_path)  # Cached by OS if hot

          context = ep['features'][frame-16:frame]  # [16, dim]
          targets = ep['future_deltas'][frame]      # [7, 4]
          valid = ep['valid_mask'][frame]           # [7]

          return torch.from_numpy(context), torch.from_numpy(targets), valid

  Optimization: Pre-load hot episodes
  # In DataLoader init:
  self.cache = {}
  for ep_path in most_common_episodes[:100]:  # Cache top 100
      self.cache[ep_path] = np.load(ep_path)

  ---
  6. Training: BF16 Without GradScaler

  User specification: Use bf16, NO gradscaler

  def train():
      model = FuturePositionPredictor(config).cuda()
      model = torch.compile(model)

      optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
      scheduler = CosineAnnealingLR(optimizer, T_max=100_000)

      for batch in dataloader:
          context, targets, valid_mask = batch
          context = context.to('cuda', dtype=torch.bfloat16)
          targets = targets.to('cuda', dtype=torch.bfloat16)

          # No autocast wrapper needed, inputs already bf16
          p1_traj, p2_traj = model(context)

          loss = 0
          for i in range(7):  # 7 horizons
              if valid_mask[:, i].any():
                  loss += mixture_nll(p1_traj[i], targets[:, i, :2], valid_mask[:, i])
                  loss += mixture_nll(p2_traj[i], targets[:, i, 2:4], valid_mask[:, i])
          loss = loss / 14  # Average over 7 horizons × 2 players

          # No scaler, direct backward
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          optimizer.zero_grad(set_to_none=True)
          scheduler.step()

  Why bf16 without scaler works:
  - BF16 range: ~10^-38 to 10^38 (same exponent range as FP32)
  - Loss scaling unnecessary
  - Gradients rarely underflow
  - Simpler code

  ---
  Updated Implementation Plan

  Phase 1: Data Pipeline (Days 1-3)

  Step 1.1: Replay Parser (data/parse_slp.py)
  def parse_slp_to_features(slp_path: Path) -> Dict[str, np.ndarray]:
      """Parse .slp file into raw features."""
      console = Console(is_dolphin=False, path=str(slp_path))

      frames = []
      prev_state = None

      while gamestate := console.step():
          if not is_valid_frame(gamestate):
              continue

          # Extract raw features
          p1 = gamestate.players[1]
          p2 = gamestate.players[2]

          # Compute derived features (velocity, input deltas)
          if prev_state:
              p1_vel_x = p1.position.x - prev_state['p1_x']
              p1_vel_y = p1.position.y - prev_state['p1_y']
              # ... button press onsets, input deltas
          else:
              p1_vel_x = 0
              # ... (first frame initialization)

          frame_features = build_feature_vector(
              p1, p2, gamestate,
              prev_velocities=(p1_vel_x, p1_vel_y, ...),
              prev_inputs=prev_state['inputs'] if prev_state else None
          )
          frames.append(frame_features)
          prev_state = save_state(p1, p2, gamestate)

      return {
          'features': np.array(frames, dtype=np.float32),
          'stage': gamestate.stage.value,
          'p1_char': p1.character.value,
          'p2_char': p2.character.value,
      }

  Step 1.2: Future Delta Computation (data/compute_targets.py)
  def compute_future_deltas(positions: np.ndarray, horizons=[5,10,20,30,40,50,60]):
      """
      positions: [n_frames, 4] (p1_x, p1_y, p2_x, p2_y)
      Returns: [n_frames, n_horizons, 4] deltas
      """
      n_frames = len(positions)
      deltas = np.zeros((n_frames, len(horizons), 4), dtype=np.float32)
      valid = np.zeros((n_frames, len(horizons)), dtype=bool)

      for t in range(n_frames):
          for i, h in enumerate(horizons):
              if t + h < n_frames:
                  deltas[t, i] = positions[t + h] - positions[t]
                  valid[t, i] = True

      return deltas, valid

  Step 1.3: NPZ Writer (data/build_dataset.py)
  from multiprocessing import Pool
  from pathlib import Path
  import numpy as np

  def process_replay(slp_path: Path, output_dir: Path):
      """Process single replay to .npz"""
      try:
          data = parse_slp_to_features(slp_path)
          features = data['features']
          positions = features[:, [POS_X_P1, POS_Y_P1, POS_X_P2, POS_Y_P2]]

          deltas, valid = compute_future_deltas(positions)

          episode_id = slp_path.stem
          output_path = output_dir / f"{episode_id}.npz"

          np.savez_compressed(
              output_path,
              features=features.astype(np.float32),
              future_deltas=deltas.astype(np.float32),
              valid_mask=valid,
              stage=np.array([data['stage']], dtype=np.uint8),
              p1_char=np.array([data['p1_char']], dtype=np.uint8),
              p2_char=np.array([data['p2_char']], dtype=np.uint8),
          )
          return True
      except Exception as e:
          print(f"Failed {slp_path}: {e}")
          return False

  def build_dataset(slp_dir: Path, output_dir: Path, n_workers=8):
      slp_files = list(slp_dir.glob('**/*.slp'))
      output_dir.mkdir(parents=True, exist_ok=True)

      with Pool(n_workers) as pool:
          results = pool.starmap(process_replay,
                                [(slp, output_dir) for slp in slp_files])

      print(f"Processed {sum(results)}/{len(results)} replays")

      # Build index
      index = []
      for npz_path in output_dir.glob('*.npz'):
          ep = np.load(npz_path)
          index.append({
              'path': str(npz_path),
              'num_frames': len(ep['features']),
              'stage': int(ep['stage'][0]),
              'p1_char': int(ep['p1_char'][0]),
              'p2_char': int(ep['p2_char'][0]),
          })

      import json
      with open(output_dir / 'index.json', 'w') as f:
          json.dump(index, f, indent=2)

  Phase 2: Model Implementation (Days 4-6)

  Step 2.1: ResNet Output Head (model/output_head.py)
  class ResBlock(nn.Module):
      def __init__(self, dim: int, expansion: int = 2):
          super().__init__()
          hidden = dim * expansion
          self.mlp = nn.Sequential(
              nn.Linear(dim, hidden, bias=False),
              nn.ReLU(),
              nn.Linear(hidden, dim, bias=False),
          )
          self.norm = nn.RMSNorm(dim)

      def forward(self, x):
          return F.relu(x + self.mlp(self.norm(x)))

  class TrajectoryOutputHead(nn.Module):
      def __init__(self, d_model=192, n_horizons=7, n_components=6):
          super().__init__()
          self.horizons = [5, 10, 20, 30, 40, 50, 60]

          # Horizon embeddings
          self.horizon_embed = nn.Embedding(61, 32)  # 0-60

          # Shared trunk
          self.project = nn.Linear(d_model + 32, 384)
          self.res1 = ResBlock(384)
          self.res2 = ResBlock(384)

          # Per-horizon heads (could share, but specialized is better)
          self.p1_heads = nn.ModuleList([
              MixtureDensityHead(384, n_components)
              for _ in range(n_horizons)
          ])
          self.p2_heads = nn.ModuleList([
              MixtureDensityHead(384, n_components)
              for _ in range(n_horizons)
          ])

      def forward(self, h_context):
          # h_context: [B, d_model]
          B = h_context.size(0)

          p1_trajectories = []
          p2_trajectories = []

          for i, h in enumerate(self.horizons):
              # Embed horizon
              h_embed = self.horizon_embed(
                  torch.full((B,), h, device=h_context.device, dtype=torch.long)
              )  # [B, 32]

              # Concatenate and process
              x = torch.cat([h_context, h_embed], dim=-1)  # [B, d_model+32]
              x = F.relu(self.project(x))  # [B, 384]
              x = self.res1(x)
              x = self.res2(x)

              # Predict mixtures
              p1_mix = self.p1_heads[i](x)
              p2_mix = self.p2_heads[i](x)

              p1_trajectories.append(p1_mix)
              p2_trajectories.append(p2_mix)

          return p1_trajectories, p2_trajectories

  class MixtureDensityHead(nn.Module):
      def __init__(self, d_input, n_components=6):
          super().__init__()
          self.n_components = n_components
          # Output: weights[K], mu_x[K], mu_y[K], log_sigma_x[K], log_sigma_y[K]
          self.proj = nn.Linear(d_input, n_components * 5)

      def forward(self, x):
          # x: [B, d_input]
          out = self.proj(x)  # [B, K*5]
          out = out.reshape(x.size(0), self.n_components, 5)

          weights_logits = out[..., 0]       # [B, K]
          mu_x = out[..., 1]                 # [B, K]
          mu_y = out[..., 2]                 # [B, K]
          log_sigma_x = out[..., 3]          # [B, K]
          log_sigma_y = out[..., 4]          # [B, K]

          # Stable sigma via softplus
          sigma_x = (F.softplus(log_sigma_x) + 0.5).clamp(min=1.0, max=40.0)
          sigma_y = (F.softplus(log_sigma_y) + 0.5).clamp(min=1.0, max=40.0)

          return {
              'weights': F.softmax(weights_logits, dim=-1),  # [B, K]
              'mu_x': mu_x,
              'mu_y': mu_y,
              'sigma_x': sigma_x,
              'sigma_y': sigma_y,
          }

  Step 2.2: Loss Function (model/loss.py)
  def mixture_nll_loss(mixture_params, target_delta, valid_mask=None):
      """
      mixture_params: dict with keys [weights, mu_x, mu_y, sigma_x, sigma_y]
                      each [B, K]
      target_delta: [B, 2] true (dx, dy) in normalized units
      valid_mask: [B] bool mask for valid samples
      """
      weights = mixture_params['weights']      # [B, K]
      mu_x = mixture_params['mu_x']            # [B, K]
      mu_y = mixture_params['mu_y']            # [B, K]
      sigma_x = mixture_params['sigma_x']      # [B, K]
      sigma_y = mixture_params['sigma_y']      # [B, K]

      target_x = target_delta[:, 0:1]  # [B, 1]
      target_y = target_delta[:, 1:2]  # [B, 1]

      # Log probability under each component
      log_px = gaussian_log_prob(target_x, mu_x, sigma_x)  # [B, K]
      log_py = gaussian_log_prob(target_y, mu_y, sigma_y)  # [B, K]
      log_pxy = log_px + log_py  # Independence assumption

      # Mixture log probability
      log_weights = torch.log(weights + 1e-8)
      log_mixture = torch.logsumexp(log_weights + log_pxy, dim=-1)  # [B]

      # Negative log likelihood
      nll = -log_mixture

      # Apply mask if provided
      if valid_mask is not None:
          nll = nll * valid_mask
          return nll.sum() / valid_mask.sum().clamp(min=1)
      else:
          return nll.mean()

  def gaussian_log_prob(x, mu, sigma):
      """
      x: [B, 1]
      mu: [B, K]
      sigma: [B, K]
      Returns: [B, K] log probabilities
      """
      return -0.5 * (
          torch.log(2 * torch.pi * sigma**2) +
          ((x - mu) / sigma)**2
      )

  Phase 3: Training (Days 7-9)

  # train.py
  import torch
  from torch.utils.data import DataLoader
  from pathlib import Path

  def train():
      # Dataset
      dataset = NPZDataset(Path('data/episodes'), context_length=16)
      dataloader = DataLoader(
          dataset,
          batch_size=256,
          shuffle=True,
          num_workers=8,
          pin_memory=True,
          prefetch_factor=4,
          persistent_workers=True,
      )

      # Model
      model = FuturePositionPredictor(d_model=192, n_layers=3).cuda()
      model = torch.compile(model, mode='max-autotune')

      # Optimizer
      optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, fused=True)
      scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-5)

      step = 0
      for epoch in range(100):
          for context, targets, valid in dataloader:
              # BF16 conversion
              context = context.cuda(non_blocking=True).to(torch.bfloat16)
              targets = targets.cuda(non_blocking=True).to(torch.bfloat16)
              valid = valid.cuda(non_blocking=True)

              # Forward
              p1_traj, p2_traj = model(context)

              # Loss over all horizons
              loss = 0
              for i in range(7):
                  loss += mixture_nll_loss(
                      p1_traj[i], targets[:, i, :2], valid[:, i]
                  )
                  loss += mixture_nll_loss(
                      p2_traj[i], targets[:, i, 2:4], valid[:, i]
                  )
              loss = loss / 14

              # Backward (no scaler)
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              optimizer.step()
              optimizer.zero_grad(set_to_none=True)
              scheduler.step()

              if step % 100 == 0:
                  print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

              step += 1

  ---
  Summary of Changes

  | Aspect                 | Original             | Refined                                                       |
  |------------------------|----------------------|---------------------------------------------------------------|
  | Output head            | Shallow MLP          | ResNet with 2-3 blocks                                        |
  | Horizon handling       | Single horizon input | Predict 7 keyframe horizons jointly                           |
  | Coordinates            | Absolute positions   | Position deltas (offsets)                                     |
  | Sigma parameterization | exp().clamp()        | softplus() + epsilon                                          |
  | Features removed       | -                    | Action frame counter, hitstun remaining                       |
  | Features added         | -                    | Shoulder analog, button onsets, input deltas, stage proximity |
  | Storage                | Zarr                 | Compressed NPZ per episode                                    |
  | Precision              | BF16 + gradscaler    | BF16, no scaler                                               |
  | Parameters             | ~1.5M                | ~2.2M (deeper output head)                                    |

  This design now has:
  1. ✅ Deep horizon-aware processing (fixes extrapolation bottleneck)
  2. ✅ Stable MDN training (softplus, delta prediction)
  3. ✅ Trajectory consistency (joint multi-horizon prediction)
  4. ✅ Cleaner features (removed noise, added discriminative signals)
  5. ✅ Simpler data pipeline (NPZ instead of Zarr)
  6. ✅ BF16 native training (no scaler complexity)


  Directory Structure

  future_position/
  ├── README.md
  ├── constants.py
  ├── config.py
  ├── data/
  │   ├── __init__.py
  │   ├── parse_slp.py
  │   ├── features.py
  │   ├── compute_targets.py
  │   ├── build_dataset.py
  │   └── dataset.py
  ├── model/
  │   ├── __init__.py
  │   ├── embeddings.py
  │   ├── encoder.py
  │   ├── output_head.py
  │   ├── predictor.py
  │   └── loss.py
  ├── train/
  │   ├── __init__.py
  │   ├── train.py
  │   └── validate.py
  └── inference/
      ├── __init__.py
      ├── engine.py
      └── visualize.py

  ---
  future_position/README.md

  # Future Position Prediction Model

  A standalone model for predicting future character positions in Super Smash Bros. Melee.

  ## Overview

  This model predicts where both players will be at multiple future time horizons (5, 10, 20, 30, 40, 50, 60 frames ahead) using a Transformer-based architecture with Mixture Density
  Network outputs.

  ## Architecture

  - **Input**: 16 frames of game state context (positions, velocities, action states, inputs, character physics)
  - **Encoder**: 3-layer Transformer with rotary positional embeddings
  - **Output**: ResNet-style heads predicting 6-component Gaussian mixtures for position deltas at 7 keyframe horizons
  - **Parameters**: ~2.2M
  - **Inference**: <5ms on RTX 4090 with bf16

  ## Usage

  ### 1. Data Preprocessing
  ```bash
  python -m future_position.data.build_dataset \
      --slp-dir /path/to/replays \
      --output-dir data/processed \
      --n-workers 8

  2. Training

  python -m future_position.train.train \
      --data-dir data/processed \
      --output-dir checkpoints \
      --batch-size 256

  3. Inference

  python -m future_position.inference.visualize \
      --replay /path/to/replay.slp \
      --checkpoint checkpoints/best.pt \
      --output visualization.mp4

  Design

  See the detailed design document for architecture rationale and implementation plan.

  ---

  ## `future_position/constants.py`

  ```python
  """Shared constants for future position prediction model."""

  from typing import List

  # Horizons to predict (in frames, 60 FPS)
  HORIZONS: List[int] = [5, 10, 20, 30, 40, 50, 60]

  # Context window length
  CONTEXT_LENGTH: int = 16

  # Stage normalization
  STAGE_HALF_WIDTH: float = 90.0
  STAGE_HALF_HEIGHT: float = 70.0

  # Character physics fields from characterdata.csv
  CHARACTER_PHYSICS_FIELDS: List[str] = [
      "Gravity",
      "TerminalVelocity",
      "FastFallSpeed",
      "AirSpeed",
      "AirFriction",
      "MaxWalkSpeed",
      "Friction",
      "InitDJSpeed",
  ]

  # Embedding dimensions
  CHARACTER_EMBED_DIM: int = 16
  ACTION_STATE_EMBED_DIM: int = 24
  STAGE_EMBED_DIM: int = 8
  HORIZON_EMBED_DIM: int = 32

  # Counts
  MAX_ACTION_STATES: int = 512
  N_CHARACTERS: int = 26
  N_STAGES: int = 8
  N_BUTTONS: int = 5

  # Model architecture
  D_MODEL: int = 192
  N_LAYERS: int = 3
  N_HEADS: int = 6
  MLP_RATIO: int = 3
  OUTPUT_HIDDEN_DIM: int = 384
  N_RESBLOCKS: int = 2
  N_MIXTURE_COMPONENTS: int = 6

  # Training
  MIN_SIGMA: float = 1.0
  MAX_SIGMA: float = 40.0
  SIGMA_EPSILON: float = 0.5
  MAX_GRAD_NORM: float = 1.0

  # NPZ field names
  NPZ_FEATURES_KEY: str = "features"
  NPZ_FUTURE_DELTAS_KEY: str = "future_deltas"
  NPZ_VALID_MASK_KEY: str = "valid_mask"
  NPZ_STAGE_KEY: str = "stage"
  NPZ_P1_CHAR_KEY: str = "p1_char"
  NPZ_P2_CHAR_KEY: str = "p2_char"

  ---
  future_position/config.py

  """Configuration dataclasses."""

  from dataclasses import dataclass, field
  from pathlib import Path
  from typing import List, Optional, Tuple


  @dataclass
  class DataConfig:
      """Data processing configuration."""
      slp_dir: Optional[Path] = None
      output_dir: Optional[Path] = None
      n_workers: int = 8
      context_length: int = 16
      horizons: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 60])
      val_fraction: float = 0.1
      random_seed: int = 42


  @dataclass
  class ModelConfig:
      """Model architecture configuration."""
      d_model: int = 192
      n_layers: int = 3
      n_heads: int = 6
      mlp_ratio: int = 3
      dropout: float = 0.0
      character_embed_dim: int = 16
      action_state_embed_dim: int = 24
      stage_embed_dim: int = 8
      horizon_embed_dim: int = 32
      output_hidden_dim: int = 384
      n_resblocks: int = 2
      n_mixture_components: int = 6
      min_sigma: float = 1.0
      max_sigma: float = 40.0
      sigma_epsilon: float = 0.5


  @dataclass
  class TrainingConfig:
      """Training configuration."""
      batch_size: int = 256
      learning_rate: float = 1e-3
      weight_decay: float = 0.01
      max_epochs: int = 100
      warmup_steps: int = 1000
      scheduler_type: str = "cosine"
      min_lr: float = 1e-5
      max_grad_norm: float = 1.0
      num_workers: int = 8
      prefetch_factor: int = 4
      pin_memory: bool = True
      persistent_workers: bool = True
      save_every: int = 1000
      keep_top_k: int = 5
      log_every: int = 100
      device: str = "cuda"
      use_compile: bool = True
      compile_mode: str = "max-autotune"


  @dataclass
  class InferenceConfig:
      """Inference configuration."""
      checkpoint_path: Optional[Path] = None
      device: str = "cuda"
      batch_size: int = 1
      use_compile: bool = False


  @dataclass
  class VisualizationConfig:
      """Visualization configuration."""
      fps: int = 60
      dpi: int = 100
      resolution: Tuple[int, int] = (1200, 1000)
      heatmap_resolution: int = 64
      heatmap_alpha: float = 0.8
      show_ground_truth: bool = True
      show_prediction: bool = True
      display_horizons: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60])


  @dataclass
  class Config:
      """Top-level configuration."""
      data: DataConfig = field(default_factory=DataConfig)
      model: ModelConfig = field(default_factory=ModelConfig)
      training: TrainingConfig = field(default_factory=TrainingConfig)
      inference: InferenceConfig = field(default_factory=InferenceConfig)
      visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

      def to_dict(self) -> dict:
          """Convert to dictionary for serialization."""
          # TODO: Implement
          raise NotImplementedError

      @classmethod
      def from_dict(cls, d: dict) -> 'Config':
          """Load from dictionary."""
          # TODO: Implement
          raise NotImplementedError

      def save(self, path: Path) -> None:
          """Save to JSON."""
          # TODO: Implement
          raise NotImplementedError

      @classmethod
      def load(cls, path: Path) -> 'Config':
          """Load from JSON."""
          # TODO: Implement
          raise NotImplementedError

  ---
  future_position/data/__init__.py

  """Data processing pipeline."""

  from .dataset import NPZDataset
  from .build_dataset import build_dataset

  __all__ = ['NPZDataset', 'build_dataset']

  ---
  future_position/data/parse_slp.py

  """Parse .slp replay files to extract raw features.

  Uses libmelee to parse replay files and extract frame-by-frame game state.
  """

  from pathlib import Path
  from typing import Dict, Optional
  import numpy as np


  def parse_slp_to_features(slp_path: Path) -> Dict[str, np.ndarray]:
      """Parse .slp file and extract all relevant features.

      Args:
          slp_path: Path to .slp replay file

      Returns:
          Dictionary containing:
              - 'features': [n_frames, feature_dim] array of per-frame features
              - 'stage': Stage ID
              - 'p1_char': Player 1 character ID
              - 'p2_char': Player 2 character ID
              - 'positions': [n_frames, 4] (p1_x, p1_y, p2_x, p2_y)

      Notes:
          - Uses libmelee.Console for parsing
          - Computes derived features (velocity, input deltas, button onsets)
          - Handles first frame initialization
          - Only uses libmelee, no dependencies on parent repo
      """
      # TODO: Implement using libmelee.Console
      # 1. Open console connection to replay file
      # 2. Iterate through game states
      # 3. Extract raw features per frame
      # 4. Compute derived features (velocity = pos[t] - pos[t-1])
      # 5. Compute input changes (button onsets, analog deltas)
      # 6. Return structured numpy arrays
      raise NotImplementedError


  def is_valid_frame(gamestate) -> bool:
      """Check if game state is valid for feature extraction.

      Args:
          gamestate: libmelee.GameState object

      Returns:
          True if frame should be processed, False otherwise

      Notes:
          - Skip menu states, paused frames, etc.
          - Require both players to be active
      """
      # TODO: Implement validation logic
      # - Check gamestate.menu_state
      # - Check player states
      # - Skip invalid frames
      raise NotImplementedError


  def extract_player_features(
      player,
      prev_player_state: Optional[Dict],
      prev_inputs: Optional[Dict],
      character_physics: Dict[str, float],
      stage_bounds: Dict[str, float],
  ) -> np.ndarray:
      """Extract features for a single player in a single frame.

      Args:
          player: libmelee.PlayerState object
          prev_player_state: Previous frame's player state (for velocity computation)
          prev_inputs: Previous frame's inputs (for delta/onset computation)
          character_physics: Physics constants from characterdata.csv
          stage_bounds: Stage-specific bounds (ledges, blast zones)

      Returns:
          Feature vector [feature_dim] for this player

      Features extracted:
          - Position (X, Y)
          - Velocity (X, Y) - computed from position delta
          - Facing direction
          - Action state ID
          - Grounded/airborne
          - Percent, shield size, jumps remaining
          - Controller inputs (joystick, c-stick, shoulder, buttons)
          - Input changes (button onsets, analog deltas)
          - Stage proximity (distance to ledges, blast zones)
          - Character physics constants (8 values)
      """
      # TODO: Implement feature extraction
      # 1. Extract position from player.position.x, player.position.y
      # 2. Compute velocity if prev_player_state exists
      # 3. Extract action state: player.action.value
      # 4. Extract combat state: player.percent, player.shield_strength, etc.
      # 5. Extract controller: player.controller_state
      # 6. Compute input changes (compare with prev_inputs)
      # 7. Compute stage proximity
      # 8. Append character physics constants
      # 9. Return concatenated feature vector
      raise NotImplementedError


  def compute_stage_proximity(
      x: float,
      y: float,
      stage_bounds: Dict[str, float],
  ) -> np.ndarray:
      """Compute distance to stage features (ledges, blast zones).

      Args:
          x: X position
          y: Y position
          stage_bounds: Dict with keys: left_ledge, right_ledge, top_blastzone, etc.

      Returns:
          [4] array: [dist_to_left_ledge, dist_to_right_ledge,
                      dist_to_floor, dist_to_top_blastzone]
      """
      # TODO: Implement
      # - Distance can be negative (offstage)
      # - Normalize by stage size
      raise NotImplementedError


  def compute_button_onsets(
      current_buttons: np.ndarray,
      prev_buttons: Optional[np.ndarray],
  ) -> np.ndarray:
      """Compute button press onsets (0->1 transitions).

      Args:
          current_buttons: [5] binary button states
          prev_buttons: [5] previous frame button states (or None for first frame)

      Returns:
          [5] binary onset flags
      """
      # TODO: Implement
      # onset = (current > 0) & (prev == 0)
      # Handle first frame (prev is None)
      raise NotImplementedError


  def compute_input_deltas(
      current_inputs: Dict[str, float],
      prev_inputs: Optional[Dict[str, float]],
  ) -> np.ndarray:
      """Compute analog input changes from previous frame.

      Args:
          current_inputs: Dict with keys: joystick_x, joystick_y, cstick_x, cstick_y, shoulder
          prev_inputs: Previous frame inputs (or None)

      Returns:
          [5] array of deltas for each analog input
      """
      # TODO: Implement
      # delta = current - prev
      # Handle first frame (return zeros)
      raise NotImplementedError


  def load_character_physics(character_id: int) -> Dict[str, float]:
      """Load character physics constants from characterdata.csv.

      Args:
          character_id: Character ID (from libmelee.Character enum)

      Returns:
          Dict mapping physics field names to values

      Notes:
          - Reads from libmelee/melee/characterdata.csv
          - Returns 8 constants defined in constants.CHARACTER_PHYSICS_FIELDS
      """
      # TODO: Implement
      # 1. Load characterdata.csv (use pandas or csv module)
      # 2. Find row matching character_id
      # 3. Extract required fields
      # 4. Return as dict
      raise NotImplementedError


  def get_stage_bounds(stage_id: int) -> Dict[str, float]:
      """Get stage-specific bounds (ledges, blast zones).

      Args:
          stage_id: Stage ID (from libmelee.Stage enum)

      Returns:
          Dict with keys: left_ledge, right_ledge, stage_floor, top_blastzone, etc.

      Notes:
          - May need to hardcode common stage bounds or extract from libmelee
          - Used for stage proximity features
      """
      # TODO: Implement
      # Hardcode bounds for legal stages:
      # - Final Destination
      # - Battlefield
      # - Fountain of Dreams
      # - Yoshi's Story
      # - Dream Land
      # - Pokemon Stadium
      raise NotImplementedError

  ---
  future_position/data/features.py

  """Feature extraction utilities.

  Helper functions for building feature vectors from parsed game state.
  """

  import numpy as np
  from typing import Dict, List


  def normalize_position(x: float, y: float, stage_half_width: float, stage_half_height: float) -> tuple:
      """Normalize position coordinates to [-1, 1] range.

      Args:
          x: X position in game units
          y: Y position in game units
          stage_half_width: Half-width of stage for normalization
          stage_half_height: Half-height of stage for normalization

      Returns:
          (normalized_x, normalized_y)
      """
      # TODO: Implement
      raise NotImplementedError


  def denormalize_position(x_norm: float, y_norm: float, stage_half_width: float, stage_half_height: float) -> tuple:
      """Convert normalized coordinates back to game units.

      Args:
          x_norm: Normalized X in [-1, 1]
          y_norm: Normalized Y in [-1, 1]
          stage_half_width: Half-width of stage
          stage_half_height: Half-height of stage

      Returns:
          (x, y) in game units
      """
      # TODO: Implement
      raise NotImplementedError


  def build_feature_vector(
      p1_features: np.ndarray,
      p2_features: np.ndarray,
      global_features: np.ndarray,
      relational_features: np.ndarray,
  ) -> np.ndarray:
      """Concatenate all feature components into single vector.

      Args:
          p1_features: Player 1 features
          p2_features: Player 2 features
          global_features: Stage, frame count, etc.
          relational_features: Distance between players, relative position

      Returns:
          Concatenated feature vector
      """
      # TODO: Implement
      raise NotImplementedError


  def compute_relational_features(
      p1_x: float,
      p1_y: float,
      p2_x: float,
      p2_y: float,
  ) -> np.ndarray:
      """Compute features describing relationship between players.

      Args:
          p1_x, p1_y: Player 1 position
          p2_x, p2_y: Player 2 position

      Returns:
          [3] array: [distance, relative_x, relative_y]
              where relative = p2 - p1
      """
      # TODO: Implement
      # distance = sqrt((p2_x - p1_x)^2 + (p2_y - p1_y)^2)
      # relative_x = p2_x - p1_x
      # relative_y = p2_y - p1_y
      raise NotImplementedError


  class FeatureNormalizer:
      """Handles feature normalization statistics.

      Computes and applies mean/std normalization to continuous features.
      Saves normalization stats for inference time.
      """

      def __init__(self):
          """Initialize normalizer."""
          self.mean: Optional[np.ndarray] = None
          self.std: Optional[np.ndarray] = None

      def fit(self, features: np.ndarray) -> None:
          """Compute normalization statistics from data.

          Args:
              features: [n_samples, feature_dim] array
          """
          # TODO: Implement
          # self.mean = features.mean(axis=0)
          # self.std = features.std(axis=0) + 1e-8
          raise NotImplementedError

      def transform(self, features: np.ndarray) -> np.ndarray:
          """Apply normalization to features.

          Args:
              features: [n_samples, feature_dim] array

          Returns:
              Normalized features
          """
          # TODO: Implement
          # return (features - self.mean) / self.std
          raise NotImplementedError

      def inverse_transform(self, features: np.ndarray) -> np.ndarray:
          """Reverse normalization.

          Args:
              features: [n_samples, feature_dim] normalized array

          Returns:
              Original scale features
          """
          # TODO: Implement
          # return features * self.std + self.mean
          raise NotImplementedError

      def save(self, path: Path) -> None:
          """Save normalization stats to file."""
          # TODO: Implement
          raise NotImplementedError

      @classmethod
      def load(cls, path: Path) -> 'FeatureNormalizer':
          """Load normalization stats from file."""
          # TODO: Implement
          raise NotImplementedError

  ---
  future_position/data/compute_targets.py

  """Compute ground truth future position deltas.

  For each frame, compute the actual position deltas at each horizon.
  """

  import numpy as np
  from typing import List, Tuple


  def compute_future_deltas(
      positions: np.ndarray,
      horizons: List[int],
  ) -> Tuple[np.ndarray, np.ndarray]:
      """Compute future position deltas for all frames.

      Args:
          positions: [n_frames, 4] array (p1_x, p1_y, p2_x, p2_y)
          horizons: List of frame offsets to predict (e.g., [5, 10, 20, ...])

      Returns:
          deltas: [n_frames, n_horizons, 4] position deltas
          valid: [n_frames, n_horizons] bool mask (False for end-of-episode)

      Notes:
          - delta[t, h] = position[t + h] - position[t]
          - Mark invalid if t + h >= n_frames (episode ends)
      """
      # TODO: Implement
      # 1. Initialize output arrays
      # 2. For each frame t and horizon h:
      #    - If t + h < n_frames:
      #        deltas[t, h] = positions[t + h] - positions[t]
      #        valid[t, h] = True
      #    - Else:
      #        deltas[t, h] = 0 (or NaN)
      #        valid[t, h] = False
      # 3. Return deltas and mask
      raise NotImplementedError


  def normalize_deltas(
      deltas: np.ndarray,
      stage_half_width: float,
      stage_half_height: float,
  ) -> np.ndarray:
      """Normalize position deltas to standard scale.

      Args:
          deltas: [n_frames, n_horizons, 4] position deltas
          stage_half_width: Normalization constant for X
          stage_half_height: Normalization constant for Y

      Returns:
          Normalized deltas where X deltas / stage_half_width, Y deltas / stage_half_height
      """
      # TODO: Implement
      # deltas_norm = deltas.copy()
      # deltas_norm[..., [0, 2]] /= stage_half_width  # p1_x, p2_x
      # deltas_norm[..., [1, 3]] /= stage_half_height  # p1_y, p2_y
      raise NotImplementedError


  def denormalize_deltas(
      deltas_norm: np.ndarray,
      stage_half_width: float,
      stage_half_height: float,
  ) -> np.ndarray:
      """Convert normalized deltas back to game units.

      Args:
          deltas_norm: [n_frames, n_horizons, 4] normalized deltas
          stage_half_width: Normalization constant for X
          stage_half_height: Normalization constant for Y

      Returns:
          Deltas in game units
      """
      # TODO: Implement (inverse of normalize_deltas)
      raise NotImplementedError

  ---
  future_position/data/build_dataset.py

  """Build dataset from .slp replay files.

  Processes replays in parallel and saves as compressed .npz files.
  """

  from pathlib import Path
  from typing import Optional
  from multiprocessing import Pool
  import numpy as np
  import json
  from tqdm import tqdm

  from .parse_slp import parse_slp_to_features
  from .compute_targets import compute_future_deltas, normalize_deltas
  from ..constants import HORIZONS, NPZ_FEATURES_KEY, NPZ_FUTURE_DELTAS_KEY, NPZ_VALID_MASK_KEY, NPZ_STAGE_KEY, NPZ_P1_CHAR_KEY, NPZ_P2_CHAR_KEY


  def process_single_replay(slp_path: Path, output_dir: Path, horizons: list) -> bool:
      """Process a single replay file to .npz format.

      Args:
          slp_path: Path to .slp file
          output_dir: Directory to save .npz output
          horizons: List of horizons to compute

      Returns:
          True if successful, False if failed

      Notes:
          - Parses replay to features
          - Computes future deltas
          - Saves as compressed .npz
          - One .npz per episode
      """
      # TODO: Implement
      # 1. Call parse_slp_to_features(slp_path)
      # 2. Extract positions array
      # 3. Call compute_future_deltas(positions, horizons)
      # 4. Save to output_dir / f"{episode_id}.npz"
      # 5. Handle exceptions, return success status
      raise NotImplementedError


  def build_dataset(
      slp_dir: Path,
      output_dir: Path,
      n_workers: int = 8,
      horizons: list = HORIZONS,
  ) -> None:
      """Build dataset from directory of .slp files.

      Args:
          slp_dir: Directory containing .slp replay files (recursive search)
          output_dir: Output directory for .npz files
          n_workers: Number of parallel workers
          horizons: List of horizons to compute

      Notes:
          - Processes replays in parallel
          - Creates index.json with episode metadata
          - Shows progress bar
      """
      # TODO: Implement
      # 1. Find all .slp files recursively
      # 2. Create output_dir
      # 3. Process in parallel with Pool
      # 4. Show progress with tqdm
      # 5. Build index from processed files
      # 6. Save index.json
      raise NotImplementedError


  def build_episode_index(output_dir: Path) -> list:
      """Build index of all episodes in output directory.

      Args:
          output_dir: Directory containing .npz files

      Returns:
          List of dicts with episode metadata:
              - path: str
              - num_frames: int
              - stage: int
              - p1_char: int
              - p2_char: int

      Notes:
          - Scans all .npz files
          - Loads metadata from each
          - Returns list for easy JSON serialization
      """
      # TODO: Implement
      # 1. Find all .npz files in output_dir
      # 2. For each file:
      #    - Load with np.load()
      #    - Extract metadata
      #    - Add to index list
      # 3. Return index
      raise NotImplementedError


  def save_index(index: list, output_path: Path) -> None:
      """Save episode index to JSON file.

      Args:
          index: List of episode metadata dicts
          output_path: Path to save JSON file
      """
      # TODO: Implement
      # with open(output_path, 'w') as f:
      #     json.dump(index, f, indent=2)
      raise NotImplementedError


  def load_index(index_path: Path) -> list:
      """Load episode index from JSON file.

      Args:
          index_path: Path to index JSON file

      Returns:
          List of episode metadata dicts
      """
      # TODO: Implement
      # with open(index_path) as f:
      #     return json.load(f)
      raise NotImplementedError


  if __name__ == '__main__':
      """Command-line interface for dataset building."""
      # TODO: Implement argument parsing
      # - --slp-dir
      # - --output-dir
      # - --n-workers
      # Call build_dataset()
      raise NotImplementedError

  ---
  future_position/data/dataset.py

  """PyTorch Dataset for loading preprocessed .npz data."""

  import numpy as np
  import torch
  from torch.utils.data import Dataset
  from pathlib import Path
  from typing import List, Tuple, Optional

  from ..constants import (
      NPZ_FEATURES_KEY,
      NPZ_FUTURE_DELTAS_KEY,
      NPZ_VALID_MASK_KEY,
      CONTEXT_LENGTH,
  )


  class NPZDataset(Dataset):
      """PyTorch Dataset for future position prediction.

      Loads preprocessed .npz files containing features and ground truth deltas.
      """

      def __init__(
          self,
          data_dir: Path,
          context_length: int = CONTEXT_LENGTH,
          cache_episodes: int = 100,
      ):
          """Initialize dataset.

          Args:
              data_dir: Directory containing .npz episode files and index.json
              context_length: Number of past frames to use as context
              cache_episodes: Number of hot episodes to cache in memory

          Notes:
              - Loads index.json to enumerate all episodes
              - Pre-computes valid sample indices (frame, episode) tuples
              - Caches frequently accessed episodes
          """
          # TODO: Implement
          # 1. Load index from data_dir / 'index.json'
          # 2. Build list of valid samples:
          #    - For each episode, for each frame where:
          #      - frame >= context_length (have enough history)
          #      - frame + max_horizon < n_frames (have future targets)
          # 3. Initialize cache dict
          raise NotImplementedError

          self.data_dir: Path = None
          self.context_length: int = None
          self.episodes: List[dict] = []
          self.samples: List[Tuple[int, int]] = []  # (episode_idx, frame_idx)
          self.cache: dict = {}

      def __len__(self) -> int:
          """Return number of valid samples."""
          # TODO: Implement
          raise NotImplementedError

      def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
          """Get a single training sample.

          Args:
              idx: Sample index

          Returns:
              context: [context_length, feature_dim] tensor
              targets: [n_horizons, 4] position deltas
              valid: [n_horizons] bool mask

          Notes:
              - Loads episode from cache or disk
              - Extracts context window and target deltas
              - Returns as torch tensors
          """
          # TODO: Implement
          # 1. Get (episode_idx, frame_idx) from self.samples[idx]
          # 2. Load episode (from cache or disk)
          # 3. Extract context: features[frame_idx - context_length : frame_idx]
          # 4. Extract targets: future_deltas[frame_idx]
          # 5. Extract valid mask: valid_mask[frame_idx]
          # 6. Convert to torch tensors
          # 7. Return (context, targets, valid)
          raise NotImplementedError

      def _load_episode(self, episode_idx: int) -> dict:
          """Load episode from cache or disk.

          Args:
              episode_idx: Index of episode in self.episodes

          Returns:
              Dict with keys: features, future_deltas, valid_mask
          """
          # TODO: Implement
          # 1. Check cache
          # 2. If not in cache, load from disk with np.load()
          # 3. Optionally add to cache
          # 4. Return episode data
          raise NotImplementedError

      def _build_cache(self, top_n: int) -> None:
          """Pre-load top N most-sampled episodes into cache.

          Args:
              top_n: Number of episodes to cache
          """
          # TODO: Implement
          # 1. Count samples per episode
          # 2. Sort episodes by sample count
          # 3. Load top N into cache
          raise NotImplementedError

  ---


  ---
  future_position/model/__init__.py

  """Model architecture for future position prediction."""

  from .predictor import FuturePositionPredictor
  from .loss import mixture_nll_loss, gaussian_log_prob

  __all__ = ['FuturePositionPredictor', 'mixture_nll_loss', 'gaussian_log_prob']

  ---
  future_position/model/embeddings.py

  """Embedding layers for categorical features."""

  import torch
  import torch.nn as nn
  from typing import Dict

  from ..constants import (
      N_CHARACTERS,
      CHARACTER_EMBED_DIM,
      MAX_ACTION_STATES,
      ACTION_STATE_EMBED_DIM,
      N_STAGES,
      STAGE_EMBED_DIM,
      HORIZON_EMBED_DIM,
      D_MODEL,
  )


  class FeatureEmbedder(nn.Module):
      """Embeds categorical features and projects to d_model.

      Handles:
      - Character ID embeddings
      - Action state ID embeddings
      - Stage ID embeddings
      - Continuous feature concatenation
      - Linear projection to d_model
      """

      def __init__(
          self,
          d_model: int = D_MODEL,
          character_embed_dim: int = CHARACTER_EMBED_DIM,
          action_state_embed_dim: int = ACTION_STATE_EMBED_DIM,
          stage_embed_dim: int = STAGE_EMBED_DIM,
          n_continuous_features: int = None,  # TODO: Define based on feature extraction
      ):
          """Initialize embeddings.

          Args:
              d_model: Output dimension after projection
              character_embed_dim: Character embedding dimension
              action_state_embed_dim: Action state embedding dimension
              stage_embed_dim: Stage embedding dimension
              n_continuous_features: Number of continuous features (after removing categoricals)
          """
          super().__init__()
          # TODO: Implement
          # 1. Create nn.Embedding layers for character, action_state, stage
          # 2. Compute total input dimension:
          #    n_continuous + 2*character_embed + 2*action_state_embed + stage_embed
          #    (2x for P1 and P2)
          # 3. Create linear projection to d_model
          raise NotImplementedError

      def forward(self, features: torch.Tensor) -> torch.Tensor:
          """Embed and project features.

          Args:
              features: [B, T, feature_dim] raw features

          Returns:
              [B, T, d_model] embedded and projected features

          Notes:
              - Extract categorical indices from features
              - Embed categoricals
              - Concatenate with continuous features
              - Project to d_model
          """
          # TODO: Implement
          # 1. Split features into continuous and categorical indices
          # 2. Embed categoricals
          # 3. Concatenate all
          # 4. Linear projection
          raise NotImplementedError


  class SinusoidalEmbedding(nn.Module):
      """Sinusoidal positional embeddings for horizons.

      Uses sin/cos functions of different frequencies for position encoding.
      """

      def __init__(self, max_len: int = 61, embed_dim: int = HORIZON_EMBED_DIM):
          """Initialize sinusoidal embeddings.

          Args:
              max_len: Maximum sequence length (61 for horizons 0-60)
              embed_dim: Embedding dimension
          """
          super().__init__()
          # TODO: Implement
          # 1. Pre-compute sinusoidal embeddings
          # 2. Register as buffer (not trained)
          # Formula:
          #   PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
          #   PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
          raise NotImplementedError

      def forward(self, positions: torch.Tensor) -> torch.Tensor:
          """Get embeddings for positions.

          Args:
              positions: [B] or [B, T] integer positions

          Returns:
              [B, embed_dim] or [B, T, embed_dim] embeddings
          """
          # TODO: Implement
          # Index into pre-computed embeddings
          raise NotImplementedError


  class RotaryPositionalEmbedding(nn.Module):
      """Rotary positional embeddings (RoPE) for Transformer.

      Applies rotations to query and key vectors based on position.
      """

      def __init__(self, dim: int, max_len: int = 1024):
          """Initialize RoPE.

          Args:
              dim: Dimension per head (should be even)
              max_len: Maximum sequence length
          """
          super().__init__()
          # TODO: Implement
          # 1. Pre-compute rotation frequencies
          # 2. Register as buffer
          # Formula from RoFormer paper
          raise NotImplementedError

      def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
          """Apply rotary embeddings to queries and keys.

          Args:
              q: [B, n_heads, T, head_dim] queries
              k: [B, n_heads, T, head_dim] keys
              seq_len: Sequence length

          Returns:
              (q_rot, k_rot) with rotary embeddings applied
          """
          # TODO: Implement
          # Apply rotation matrices to q and k
          raise NotImplementedError

      @staticmethod
      def rotate_half(x: torch.Tensor) -> torch.Tensor:
          """Rotate half the hidden dims of the input."""
          # TODO: Implement helper for rotation
          raise NotImplementedError

  ---
  future_position/model/encoder.py

  """Transformer encoder for temporal modeling."""

  import torch
  import torch.nn as nn
  from typing import Optional

  from ..constants import D_MODEL, N_LAYERS, N_HEADS, MLP_RATIO
  from .embeddings import RotaryPositionalEmbedding


  class TransformerEncoder(nn.Module):
      """Stack of Transformer encoder layers with RoPE.

      Uses:
      - Rotary positional embeddings
      - RMSNorm (instead of LayerNorm)
      - No bias in linear layers
      - Causal attention mask (optional)
      """

      def __init__(
          self,
          d_model: int = D_MODEL,
          n_layers: int = N_LAYERS,
          n_heads: int = N_HEADS,
          mlp_ratio: int = MLP_RATIO,
          dropout: float = 0.0,
          causal: bool = False,
      ):
          """Initialize encoder.

          Args:
              d_model: Model dimension
              n_layers: Number of transformer layers
              n_heads: Number of attention heads
              mlp_ratio: MLP hidden dimension = d_model * mlp_ratio
              dropout: Dropout probability
              causal: Whether to use causal attention mask
          """
          super().__init__()
          # TODO: Implement
          # 1. Create list of TransformerEncoderLayer
          # 2. Create RoPE module
          # 3. Create final RMSNorm
          raise NotImplementedError

      def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
          """Forward pass through encoder.

          Args:
              x: [B, T, d_model] input sequence
              mask: Optional [T, T] attention mask

          Returns:
              [B, T, d_model] encoded sequence
          """
          # TODO: Implement
          # 1. Apply each layer
          # 2. Apply final norm
          raise NotImplementedError


  class TransformerEncoderLayer(nn.Module):
      """Single transformer encoder layer.

      Structure:
      - Multi-head self-attention with RoPE
      - RMSNorm
      - MLP with ReLU
      - RMSNorm
      - Residual connections
      """

      def __init__(
          self,
          d_model: int,
          n_heads: int,
          mlp_ratio: int,
          dropout: float = 0.0,
      ):
          """Initialize layer.

          Args:
              d_model: Model dimension
              n_heads: Number of attention heads
              mlp_ratio: MLP expansion ratio
              dropout: Dropout probability
          """
          super().__init__()
          # TODO: Implement
          # 1. Multi-head attention
          # 2. RMSNorm
          # 3. MLP
          # 4. RMSNorm
          raise NotImplementedError

      def forward(
          self,
          x: torch.Tensor,
          rope: RotaryPositionalEmbedding,
          mask: Optional[torch.Tensor] = None,
      ) -> torch.Tensor:
          """Forward pass.

          Args:
              x: [B, T, d_model]
              rope: RoPE module for attention
              mask: Optional attention mask

          Returns:
              [B, T, d_model]
          """
          # TODO: Implement
          # 1. Self-attention with residual
          # 2. MLP with residual
          raise NotImplementedError


  class MultiHeadAttention(nn.Module):
      """Multi-head attention with RoPE support."""

      def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
          """Initialize attention.

          Args:
              d_model: Model dimension
              n_heads: Number of heads
              dropout: Dropout probability
          """
          super().__init__()
          # TODO: Implement
          # 1. Q, K, V projections (no bias)
          # 2. Output projection
          # 3. Dropout
          raise NotImplementedError

      def forward(
          self,
          x: torch.Tensor,
          rope: RotaryPositionalEmbedding,
          mask: Optional[torch.Tensor] = None,
      ) -> torch.Tensor:
          """Attention forward pass.

          Args:
              x: [B, T, d_model]
              rope: RoPE module
              mask: Optional [T, T] mask

          Returns:
              [B, T, d_model]
          """
          # TODO: Implement
          # 1. Project to Q, K, V
          # 2. Split into heads
          # 3. Apply RoPE to Q, K
          # 4. Compute attention scores
          # 5. Apply mask if provided
          # 6. Attention weights and values
          # 7. Concat heads and project
          raise NotImplementedError


  class RMSNorm(nn.Module):
      """Root Mean Square Layer Normalization.

      More efficient than LayerNorm, no mean centering.
      """

      def __init__(self, dim: int, eps: float = 1e-6):
          """Initialize RMSNorm.

          Args:
              dim: Normalization dimension
              eps: Epsilon for numerical stability
          """
          super().__init__()
          # TODO: Implement
          # 1. Scale parameter (learnable)
          # 2. Store eps
          raise NotImplementedError

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          """Apply RMSNorm.

          Args:
              x: [..., dim] input

          Returns:
              [..., dim] normalized output

          Formula:
              RMS(x) = sqrt(mean(x^2) + eps)
              output = x / RMS(x) * scale
          """
          # TODO: Implement
          raise NotImplementedError


  class MLP(nn.Module):
      """Feedforward MLP with ReLU activation."""

      def __init__(self, d_model: int, mlp_ratio: int, dropout: float = 0.0):
          """Initialize MLP.

          Args:
              d_model: Input/output dimension
              mlp_ratio: Hidden dimension = d_model * mlp_ratio
              dropout: Dropout probability
          """
          super().__init__()
          # TODO: Implement
          # 1. Linear(d_model, d_model * mlp_ratio, bias=False)
          # 2. ReLU
          # 3. Dropout
          # 4. Linear(d_model * mlp_ratio, d_model, bias=False)
          # 5. Dropout
          raise NotImplementedError

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          """Forward pass.

          Args:
              x: [B, T, d_model]

          Returns:
              [B, T, d_model]
          """
          # TODO: Implement
          raise NotImplementedError

  ---
  future_position/model/output_head.py

  """Output heads for trajectory prediction with mixture density networks."""

  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from typing import Dict, List

  from ..constants import (
      OUTPUT_HIDDEN_DIM,
      N_RESBLOCKS,
      N_MIXTURE_COMPONENTS,
      HORIZON_EMBED_DIM,
      MIN_SIGMA,
      MAX_SIGMA,
      SIGMA_EPSILON,
      HORIZONS,
  )


  class ResBlock(nn.Module):
      """Residual block for output head.

      Structure:
      - RMSNorm
      - Linear expansion
      - ReLU
      - Linear contraction
      - Residual connection
      """

      def __init__(self, dim: int, expansion: int = 2):
          """Initialize ResBlock.

          Args:
              dim: Input/output dimension
              expansion: Hidden layer expansion factor
          """
          super().__init__()
          # TODO: Implement
          # 1. RMSNorm
          # 2. Linear(dim, dim * expansion, bias=False)
          # 3. ReLU
          # 4. Linear(dim * expansion, dim, bias=False)
          raise NotImplementedError

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          """Forward with residual connection.

          Args:
              x: [B, dim]

          Returns:
              [B, dim]
          """
          # TODO: Implement
          # return x + self.mlp(self.norm(x))
          raise NotImplementedError


  class MixtureDensityHead(nn.Module):
      """Predicts mixture of Gaussians for position deltas.

      Outputs:
      - weights: [B, K] mixture weights (sum to 1)
      - mu_x, mu_y: [B, K] means for X and Y
      - sigma_x, sigma_y: [B, K] standard deviations

      Where K = n_mixture_components
      """

      def __init__(self, d_input: int, n_components: int = N_MIXTURE_COMPONENTS):
          """Initialize MDN head.

          Args:
              d_input: Input feature dimension
              n_components: Number of mixture components (K)
          """
          super().__init__()
          self.n_components = n_components
          # TODO: Implement
          # Linear projection to K * 5 outputs
          # (weights, mu_x, mu_y, log_sigma_x, log_sigma_y)
          raise NotImplementedError

      def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
          """Predict mixture parameters.

          Args:
              x: [B, d_input] input features

          Returns:
              Dict with keys:
                  'weights': [B, K] (softmax, sum to 1)
                  'mu_x': [B, K]
                  'mu_y': [B, K]
                  'sigma_x': [B, K] (positive, bounded)
                  'sigma_y': [B, K] (positive, bounded)
          """
          # TODO: Implement
          # 1. Project to K * 5
          # 2. Reshape to [B, K, 5]
          # 3. Split into components
          # 4. Apply softmax to weights
          # 5. Apply stable sigma transformation:
          #    sigma = (F.softplus(log_sigma) + SIGMA_EPSILON).clamp(MIN_SIGMA, MAX_SIGMA)
          # 6. Return dict
          raise NotImplementedError


  class TrajectoryOutputHead(nn.Module):
      """Output head for multi-horizon trajectory prediction.

      Predicts position deltas at multiple horizons using:
      - Shared ResNet trunk
      - Horizon-specific MDN heads
      """

      def __init__(
          self,
          d_model: int,
          horizon_embed_dim: int = HORIZON_EMBED_DIM,
          hidden_dim: int = OUTPUT_HIDDEN_DIM,
          n_resblocks: int = N_RESBLOCKS,
          n_components: int = N_MIXTURE_COMPONENTS,
          horizons: List[int] = HORIZONS,
      ):
          """Initialize trajectory output head.

          Args:
              d_model: Input dimension from encoder
              horizon_embed_dim: Dimension of horizon embeddings
              hidden_dim: Hidden dimension for processing
              n_resblocks: Number of residual blocks
              n_components: Number of mixture components per prediction
              horizons: List of horizons to predict
          """
          super().__init__()
          self.horizons = horizons
          # TODO: Implement
          # 1. Horizon embeddings (can be nn.Embedding or SinusoidalEmbedding)
          # 2. Initial projection: d_model + horizon_embed_dim -> hidden_dim
          # 3. ResBlocks (n_resblocks)
          # 4. Per-horizon MDN heads for P1 and P2
          #    - self.p1_heads = nn.ModuleList([MixtureDensityHead(...) for _ in horizons])
          #    - self.p2_heads = nn.ModuleList([MixtureDensityHead(...) for _ in horizons])
          raise NotImplementedError

      def forward(self, h_context: torch.Tensor) -> tuple:
          """Predict trajectories for all horizons.

          Args:
              h_context: [B, d_model] context representation from encoder

          Returns:
              (p1_trajectories, p2_trajectories)
              Each is a list of length n_horizons containing mixture parameter dicts
          """
          # TODO: Implement
          # For each horizon:
          #   1. Get horizon embedding
          #   2. Concatenate with h_context
          #   3. Project to hidden_dim
          #   4. Apply ResBlocks
          #   5. Predict P1 and P2 mixtures
          # Return lists of predictions
          raise NotImplementedError

  ---
  future_position/model/predictor.py

  """Main model: combines embeddings, encoder, and output heads."""

  import torch
  import torch.nn as nn
  from typing import List, Dict

  from .embeddings import FeatureEmbedder
  from .encoder import TransformerEncoder
  from .output_head import TrajectoryOutputHead
  from ..config import ModelConfig


  class FuturePositionPredictor(nn.Module):
      """Full model for future position prediction.

      Architecture:
      1. Feature embeddings (categorical + continuous)
      2. Transformer encoder (temporal modeling)
      3. Trajectory output head (multi-horizon MDN predictions)
      """

      def __init__(self, config: ModelConfig):
          """Initialize model.

          Args:
              config: Model configuration
          """
          super().__init__()
          self.config = config

          # TODO: Implement
          # 1. Feature embedder
          # 2. Transformer encoder
          # 3. Trajectory output head
          raise NotImplementedError

      def forward(self, context: torch.Tensor) -> tuple:
          """Forward pass.

          Args:
              context: [B, T, feature_dim] context window

          Returns:
              (p1_trajectories, p2_trajectories)
              Each is a list of mixture parameter dicts, one per horizon
          """
          # TODO: Implement
          # 1. Embed features: x = self.embedder(context)  # [B, T, d_model]
          # 2. Encode: x = self.encoder(x)  # [B, T, d_model]
          # 3. Extract last frame: h = x[:, -1]  # [B, d_model]
          # 4. Predict trajectories: p1_traj, p2_traj = self.output_head(h)
          # 5. Return trajectories
          raise NotImplementedError

      @torch.inference_mode()
      def predict(
          self,
          context: torch.Tensor,
          return_samples: bool = False,
          n_samples: int = 100,
      ) -> Dict:
          """Inference mode prediction with optional sampling.

          Args:
              context: [B, T, feature_dim] or [T, feature_dim]
              return_samples: Whether to sample from mixture
              n_samples: Number of samples to draw per prediction

          Returns:
              Dict with predictions for each horizon
          """
          # TODO: Implement
          # 1. Handle single sample (add batch dim)
          # 2. Forward pass
          # 3. Optionally sample from mixture
          # 4. Return structured predictions
          raise NotImplementedError

      def sample_from_mixture(
          self,
          mixture_params: Dict[str, torch.Tensor],
          n_samples: int = 100,
      ) -> torch.Tensor:
          """Sample from mixture of Gaussians.

          Args:
              mixture_params: Dict with mixture parameters
              n_samples: Number of samples to draw

          Returns:
              [B, n_samples, 2] samples (x, y)
          """
          # TODO: Implement
          # 1. Sample component indices from categorical(weights)
          # 2. Sample from selected Gaussians
          # 3. Return samples
          raise NotImplementedError

      def count_parameters(self) -> Dict[str, int]:
          """Count parameters in each module.

          Returns:
              Dict mapping module names to parameter counts
          """
          # TODO: Implement
          # Return counts for:
          # - embedder
          # - encoder
          # - output_head
          # - total
          raise NotImplementedError

  ---
  future_position/model/loss.py

  """Loss functions for mixture density networks."""

  import torch
  import torch.nn.functional as F
  from typing import Dict, Optional


  def mixture_nll_loss(
      mixture_params: Dict[str, torch.Tensor],
      target_delta: torch.Tensor,
      valid_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
      """Negative log-likelihood loss for mixture of Gaussians.

      Args:
          mixture_params: Dict with keys:
              'weights': [B, K] mixture weights
              'mu_x': [B, K] mean X
              'mu_y': [B, K] mean Y
              'sigma_x': [B, K] std X
              'sigma_y': [B, K] std Y
          target_delta: [B, 2] ground truth (delta_x, delta_y)
          valid_mask: [B] optional boolean mask for valid samples

      Returns:
          Scalar loss (mean over valid samples)

      Notes:
          - Assumes X and Y are independent within each component
          - Uses log-sum-exp for numerical stability
      """
      # TODO: Implement
      # 1. Extract parameters from dict
      # 2. Compute log p(x | component k) for each component
      # 3. Compute log p(y | component k) for each component
      # 4. Combine: log p(x, y | k) = log p(x | k) + log p(y | k)
      # 5. Add log weights: log w_k + log p(x, y | k)
      # 6. Mixture log prob: log_sum_exp over k
      # 7. Negative log likelihood: -log_prob
      # 8. Apply mask if provided
      # 9. Return mean
      raise NotImplementedError


  def gaussian_log_prob(
      x: torch.Tensor,
      mu: torch.Tensor,
      sigma: torch.Tensor,
  ) -> torch.Tensor:
      """Log probability of x under Gaussian(mu, sigma).

      Args:
          x: [B, 1] target values
          mu: [B, K] means
          sigma: [B, K] standard deviations

      Returns:
          [B, K] log probabilities

      Formula:
          log p(x | mu, sigma) = -0.5 * (log(2*pi*sigma^2) + ((x - mu) / sigma)^2)
      """
      # TODO: Implement
      raise NotImplementedError


  def mixture_mode_accuracy(
      mixture_params: Dict[str, torch.Tensor],
      target_delta: torch.Tensor,
      threshold: float = 5.0,
  ) -> torch.Tensor:
      """Compute accuracy: is target within threshold of highest-weight mode?

      Args:
          mixture_params: Mixture parameters
          target_delta: [B, 2] ground truth
          threshold: Distance threshold (in normalized units)

      Returns:
          Scalar accuracy (fraction of samples within threshold)
      """
      # TODO: Implement
      # 1. Find highest-weight component per sample
      # 2. Get mu_x, mu_y of that component
      # 3. Compute distance to target
      # 4. Check if distance < threshold
      # 5. Return mean
      raise NotImplementedError


  def mixture_calibration_error(
      mixture_params: Dict[str, torch.Tensor],
      target_delta: torch.Tensor,
      n_bins: int = 10,
  ) -> torch.Tensor:
      """Expected calibration error for mixture predictions.

      Measures if predicted uncertainty matches empirical error.

      Args:
          mixture_params: Mixture parameters
          target_delta: [B, 2] ground truth
          n_bins: Number of bins for calibration

      Returns:
          Scalar calibration error
      """
      # TODO: Implement
      # 1. Compute predicted confidence (e.g., max weight or inverse sigma)
      # 2. Compute actual error (distance to target)
      # 3. Bin by confidence
      # 4. Compute |avg_confidence - avg_accuracy| per bin
      # 5. Weighted average over bins
      raise NotImplementedError

  ---



  ---
  future_position/train/__init__.py

  """Training and validation utilities."""

  from .train import train
  from .validate import validate

  __all__ = ['train', 'validate']

  ---
  future_position/train/train.py

  """Training loop for future position prediction model."""

  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, random_split
  from torch.optim import AdamW
  from torch.optim.lr_scheduler import CosineAnnealingLR
  from pathlib import Path
  from typing import Optional
  import json
  from tqdm import tqdm

  from ..model import FuturePositionPredictor, mixture_nll_loss
  from ..data import NPZDataset
  from ..config import Config
  from .validate import validate


  def train(config: Config, resume_from: Optional[Path] = None) -> None:
      """Main training loop.

      Args:
          config: Training configuration
          resume_from: Optional checkpoint path to resume from

      Notes:
          - Uses bf16 precision (no grad scaler)
          - torch.compile for optimization
          - Saves checkpoints every N steps
          - Logs metrics every N steps
      """
      # TODO: Implement
      # 1. Setup device, random seed
      # 2. Load dataset and split train/val
      # 3. Create dataloaders
      # 4. Initialize model
      # 5. Apply torch.compile if config.use_compile
      # 6. Setup optimizer and scheduler
      # 7. Load checkpoint if resume_from
      # 8. Training loop:
      #    - For each epoch:
      #      - For each batch:
      #        - Forward pass (bf16)
      #        - Compute loss over all horizons
      #        - Backward pass
      #        - Gradient clipping
      #        - Optimizer step
      #        - Scheduler step
      #        - Log metrics
      #        - Save checkpoint
      #      - Validate at end of epoch
      raise NotImplementedError


  def setup_dataloaders(config: Config) -> tuple:
      """Setup train and validation dataloaders.

      Args:
          config: Configuration

      Returns:
          (train_loader, val_loader)
      """
      # TODO: Implement
      # 1. Load full dataset
      # 2. Split into train/val
      # 3. Create DataLoaders with config params
      raise NotImplementedError


  def setup_model(config: Config, device: str) -> nn.Module:
      """Initialize and setup model.

      Args:
          config: Model configuration
          device: Device to place model on

      Returns:
          Model (optionally compiled)
      """
      # TODO: Implement
      # 1. Create FuturePositionPredictor
      # 2. Move to device
      # 3. Convert to bf16
      # 4. Apply torch.compile if requested
      raise NotImplementedError


  def setup_optimizer(model: nn.Module, config: Config):
      """Setup optimizer and scheduler.

      Args:
          model: Model to optimize
          config: Training configuration

      Returns:
          (optimizer, scheduler)
      """
      # TODO: Implement
      # 1. Create AdamW with fused=True for speed
      # 2. Create scheduler (cosine annealing)
      raise NotImplementedError


  def train_step(
      model: nn.Module,
      batch: tuple,
      optimizer: torch.optim.Optimizer,
      scheduler: torch.optim.lr_scheduler._LRScheduler,
      config: Config,
  ) -> dict:
      """Single training step.

      Args:
          model: Model
          batch: (context, targets, valid_mask)
          optimizer: Optimizer
          scheduler: LR scheduler
          config: Config

      Returns:
          Dict with loss and metrics
      """
      # TODO: Implement
      # 1. Unpack batch, move to device and convert to bf16
      # 2. Forward pass
      # 3. Compute loss over all horizons
      # 4. Backward
      # 5. Gradient clipping
      # 6. Optimizer step
      # 7. Scheduler step
      # 8. Return metrics dict
      raise NotImplementedError


  def compute_total_loss(
      p1_trajectories: list,
      p2_trajectories: list,
      targets: torch.Tensor,
      valid_mask: torch.Tensor,
  ) -> torch.Tensor:
      """Compute total loss over all horizons.

      Args:
          p1_trajectories: List of P1 mixture params (one per horizon)
          p2_trajectories: List of P2 mixture params
          targets: [B, n_horizons, 4] ground truth deltas
          valid_mask: [B, n_horizons] validity mask

      Returns:
          Scalar loss (averaged over horizons and players)
      """
      # TODO: Implement
      # For each horizon:
      #   loss += mixture_nll_loss(p1_traj[i], targets[:, i, :2], valid_mask[:, i])
      #   loss += mixture_nll_loss(p2_traj[i], targets[:, i, 2:4], valid_mask[:, i])
      # Return loss / (n_horizons * 2)
      raise NotImplementedError


  def save_checkpoint(
      model: nn.Module,
      optimizer: torch.optim.Optimizer,
      scheduler: torch.optim.lr_scheduler._LRScheduler,
      step: int,
      loss: float,
      config: Config,
      checkpoint_dir: Path,
  ) -> None:
      """Save training checkpoint.

      Args:
          model: Model to save
          optimizer: Optimizer state
          scheduler: Scheduler state
          step: Current training step
          loss: Current loss
          config: Configuration
          checkpoint_dir: Directory to save checkpoint
      """
      # TODO: Implement
      # 1. Create checkpoint dict
      # 2. Save to checkpoint_dir / f"checkpoint_{step}.pt"
      # 3. Keep only top-K checkpoints by loss
      # 4. Save config alongside
      raise NotImplementedError


  def load_checkpoint(
      checkpoint_path: Path,
      model: nn.Module,
      optimizer: Optional[torch.optim.Optimizer] = None,
      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
  ) -> dict:
      """Load checkpoint.

      Args:
          checkpoint_path: Path to checkpoint
          model: Model to load weights into
          optimizer: Optional optimizer to load state
          scheduler: Optional scheduler to load state

      Returns:
          Dict with metadata (step, loss, etc.)
      """
      # TODO: Implement
      # 1. Load checkpoint
      # 2. Load model weights
      # 3. Optionally load optimizer and scheduler
      # 4. Return metadata
      raise NotImplementedError


  def log_metrics(metrics: dict, step: int) -> None:
      """Log training metrics.

      Args:
          metrics: Dict of metrics to log
          step: Training step
      """
      # TODO: Implement
      # Print to console
      # Optionally log to wandb/tensorboard
      raise NotImplementedError


  if __name__ == '__main__':
      """Command-line interface for training."""
      # TODO: Implement argument parsing
      # - --data-dir
      # - --output-dir
      # - --batch-size
      # - --lr
      # - --epochs
      # - --resume-from
      # Parse args, create config, call train()
      raise NotImplementedError

  ---
  future_position/train/validate.py

  """Validation utilities."""

  import torch
  from torch.utils.data import DataLoader
  from typing import Dict
  from tqdm import tqdm

  from ..model import FuturePositionPredictor, mixture_nll_loss, mixture_mode_accuracy
  from ..config import Config


  def validate(
      model: FuturePositionPredictor,
      val_loader: DataLoader,
      device: str = 'cuda',
  ) -> Dict[str, float]:
      """Run validation on validation set.

      Args:
          model: Model to validate
          val_loader: Validation dataloader
          device: Device to run on

      Returns:
          Dict of validation metrics:
              - loss: Average NLL loss
              - mode_accuracy: Fraction within threshold of mode
              - per_horizon_loss: Loss breakdown by horizon
      """
      # TODO: Implement
      # 1. Set model to eval mode
      # 2. Iterate through val_loader
      # 3. Compute loss without gradients
      # 4. Compute additional metrics (mode accuracy, etc.)
      # 5. Aggregate and return
      raise NotImplementedError


  @torch.inference_mode()
  def validate_step(
      model: FuturePositionPredictor,
      batch: tuple,
      device: str,
  ) -> dict:
      """Single validation step.

      Args:
          model: Model
          batch: (context, targets, valid_mask)
          device: Device

      Returns:
          Dict with metrics for this batch
      """
      # TODO: Implement
      # 1. Unpack batch, move to device, convert to bf16
      # 2. Forward pass
      # 3. Compute losses and metrics
      # 4. Return dict
      raise NotImplementedError


  def compute_per_horizon_metrics(
      p1_trajectories: list,
      p2_trajectories: list,
      targets: torch.Tensor,
      valid_mask: torch.Tensor,
  ) -> Dict[int, Dict[str, float]]:
      """Compute metrics broken down by horizon.

      Args:
          p1_trajectories: P1 predictions
          p2_trajectories: P2 predictions
          targets: Ground truth
          valid_mask: Validity mask

      Returns:
          Dict mapping horizon -> {loss, accuracy, etc.}
      """
      # TODO: Implement
      # For each horizon, compute:
      # - NLL loss
      # - Mode accuracy
      # - Average predicted sigma (uncertainty)
      raise NotImplementedError


  def compute_calibration_metrics(
      p1_trajectories: list,
      p2_trajectories: list,
      targets: torch.Tensor,
  ) -> dict:
      """Compute calibration metrics.

      Args:
          p1_trajectories: P1 predictions
          p2_trajectories: P2 predictions
          targets: Ground truth

      Returns:
          Dict with calibration error, sharpness, etc.
      """
      # TODO: Implement
      # Calibration: does predicted uncertainty match actual error?
      # Sharpness: how confident are predictions?
      raise NotImplementedError


  if __name__ == '__main__':
      """Command-line interface for validation."""
      # TODO: Implement
      # - Load checkpoint
      # - Load val data
      # - Run validation
      # - Print/save results
      raise NotImplementedError

  ---
  future_position/inference/__init__.py

  """Inference and visualization tools."""

  from .engine import FuturePredictor
  from .visualize import visualize_replay

  __all__ = ['FuturePredictor', 'visualize_replay']

  ---
  future_position/inference/engine.py

  """Inference engine for real-time prediction."""

  import torch
  from pathlib import Path
  from collections import deque
  from typing import List, Dict, Optional
  import numpy as np

  from ..model import FuturePositionPredictor
  from ..config import Config, InferenceConfig
  from ..constants import CONTEXT_LENGTH, HORIZONS


  class FuturePredictor:
      """Inference engine for future position prediction.

      Maintains rolling context window and provides fast prediction.
      """

      def __init__(
          self,
          checkpoint_path: Path,
          config: Optional[InferenceConfig] = None,
          device: str = 'cuda',
      ):
          """Initialize predictor.

          Args:
              checkpoint_path: Path to model checkpoint
              config: Optional inference config
              device: Device to run on
          """
          # TODO: Implement
          # 1. Load checkpoint and model config
          # 2. Initialize model
          # 3. Load weights
          # 4. Set to eval mode
          # 5. Optionally compile
          # 6. Initialize rolling buffer
          raise NotImplementedError

          self.model: FuturePositionPredictor = None
          self.device: str = device
          self.context_length: int = CONTEXT_LENGTH
          self.horizons: List[int] = HORIZONS
          self.buffer: deque = deque(maxlen=CONTEXT_LENGTH)

      @torch.inference_mode()
      def predict(
          self,
          current_frame: np.ndarray,
          horizons: Optional[List[int]] = None,
          return_samples: bool = False,
          n_samples: int = 100,
      ) -> Dict:
          """Predict future positions.

          Args:
              current_frame: [feature_dim] current frame features
              horizons: Optional list of horizons (default: all)
              return_samples: Whether to sample from mixture
              n_samples: Number of samples if sampling

          Returns:
              Dict with predictions:
                  'p1': Dict[horizon -> mixture_params or samples]
                  'p2': Dict[horizon -> mixture_params or samples]
          """
          # TODO: Implement
          # 1. Add current_frame to buffer
          # 2. If buffer not full, return None or zeros
          # 3. Stack buffer to context tensor [1, T, feature_dim]
          # 4. Convert to torch, move to device, bf16
          # 5. Forward pass
          # 6. Optionally sample from mixtures
          # 7. Return structured predictions
          raise NotImplementedError

      def reset(self) -> None:
          """Reset context buffer."""
          # TODO: Implement
          # self.buffer.clear()
          raise NotImplementedError

      def warmup(self, initial_frames: np.ndarray) -> None:
          """Warmup buffer with initial frames.

          Args:
              initial_frames: [n_frames, feature_dim] initial context

          Notes:
              - Should be called before first prediction
              - Fills buffer with initial frames
          """
          # TODO: Implement
          # For each frame in initial_frames:
          #   self.buffer.append(frame)
          raise NotImplementedError

      @staticmethod
      def load_checkpoint(checkpoint_path: Path) -> tuple:
          """Load checkpoint and return model + config.

          Args:
              checkpoint_path: Path to checkpoint

          Returns:
              (model, config)
          """
          # TODO: Implement
          raise NotImplementedError

  ---
  future_position/inference/visualize.py

  """Visualization tools for future position predictions."""

  import numpy as np
  import cv2
  import matplotlib.pyplot as plt
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from pathlib import Path
  from typing import List, Dict, Optional
  from tqdm import tqdm

  from .engine import FuturePredictor
  from ..data.parse_slp import parse_slp_to_features
  from ..config import VisualizationConfig


  def visualize_replay(
      replay_path: Path,
      checkpoint_path: Path,
      output_path: Path,
      config: Optional[VisualizationConfig] = None,
  ) -> None:
      """Visualize predictions on a replay.

      Args:
          replay_path: Path to .slp replay file
          checkpoint_path: Path to model checkpoint
          output_path: Path to save output video
          config: Visualization configuration

      Notes:
          - Parses replay
          - Runs inference per frame
          - Renders predictions as heatmaps overlaid on stage
          - Outputs video
      """
      # TODO: Implement
      # 1. Parse replay to features
      # 2. Initialize predictor
      # 3. Initialize video writer
      # 4. For each frame:
      #    - Run prediction
      #    - Render frame with predictions
      #    - Write to video
      # 5. Close video writer
      raise NotImplementedError


  def render_frame(
      frame_idx: int,
      positions: np.ndarray,
      predictions: Dict,
      stage_id: int,
      config: VisualizationConfig,
  ) -> np.ndarray:
      """Render a single frame with predictions.

      Args:
          frame_idx: Current frame index
          positions: [4] current positions (p1_x, p1_y, p2_x, p2_y)
          predictions: Prediction dict from FuturePredictor
          stage_id: Stage ID for rendering
          config: Visualization config

      Returns:
          [H, W, 3] RGB image (uint8)
      """
      # TODO: Implement
      # 1. Create matplotlib figure
      # 2. Draw stage background
      # 3. Draw current player positions
      # 4. For each horizon in config.display_horizons:
      #    - Render heatmap from mixture
      #    - Optionally draw ground truth
      # 5. Render figure to numpy array
      # 6. Return image
      raise NotImplementedError


  def mixture_to_heatmap(
      mixture_params: Dict,
      x_range: tuple,
      y_range: tuple,
      resolution: int = 64,
  ) -> np.ndarray:
      """Convert mixture of Gaussians to heatmap.

      Args:
          mixture_params: Dict with mixture parameters
          x_range: (x_min, x_max) for grid
          y_range: (y_min, y_max) for grid
          resolution: Grid resolution

      Returns:
          [resolution, resolution] heatmap (probability density)

      Notes:
          - Evaluates mixture on grid
          - Returns normalized density
      """
      # TODO: Implement
      # 1. Create grid
      # 2. For each mixture component:
      #    - Evaluate 2D Gaussian on grid
      #    - Weight by mixture weight
      # 3. Sum over components
      # 4. Normalize
      raise NotImplementedError


  def draw_stage(ax: plt.Axes, stage_id: int) -> None:
      """Draw stage background.

      Args:
          ax: Matplotlib axes
          stage_id: Stage ID

      Notes:
          - Draws stage boundaries, platforms
          - Simple schematic representation
      """
      # TODO: Implement
      # Draw rectangles/lines for:
      # - Main platform
      # - Side platforms
      # - Top platforms
      # - Blast zones (dashed)
      raise NotImplementedError


  def draw_player_positions(
      ax: plt.Axes,
      p1_pos: tuple,
      p2_pos: tuple,
  ) -> None:
      """Draw current player positions.

      Args:
          ax: Matplotlib axes
          p1_pos: (x, y) for player 1
          p2_pos: (x, y) for player 2
      """
      # TODO: Implement
      # Draw circles or markers for players
      # Different colors for P1 and P2
      raise NotImplementedError


  def figure_to_array(fig: plt.Figure) -> np.ndarray:
      """Convert matplotlib figure to numpy array.

      Args:
          fig: Matplotlib figure

      Returns:
          [H, W, 3] RGB array (uint8)
      """
      # TODO: Implement
      # 1. Render to canvas
      # 2. Convert to numpy array
      # 3. Reshape to [H, W, 3]
      raise NotImplementedError


  class VideoWriter:
      """Wrapper for cv2.VideoWriter."""

      def __init__(
          self,
          output_path: Path,
          fps: int = 60,
          resolution: tuple = (1200, 1000),
      ):
          """Initialize video writer.

          Args:
              output_path: Path to save video
              fps: Frames per second
              resolution: (width, height)
          """
          # TODO: Implement
          # Initialize cv2.VideoWriter
          raise NotImplementedError

      def write_frame(self, frame: np.ndarray) -> None:
          """Write a frame to video.

          Args:
              frame: [H, W, 3] RGB array
          """
          # TODO: Implement
          raise NotImplementedError

      def close(self) -> None:
          """Close video writer."""
          # TODO: Implement
          raise NotImplementedError


  if __name__ == '__main__':
      """Command-line interface for visualization."""
      # TODO: Implement argument parsing
      # - --replay
      # - --checkpoint
      # - --output
      # - --horizons
      # Parse args and call visualize_replay()
      raise NotImplementedError

  ---
