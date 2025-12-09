"""Shared constants for future position prediction model."""

from typing import List

# Horizons to predict (in frames, 60 FPS)
HORIZONS: List[int] = [5, 10, 20, 30, 40, 50, 60]

# Context window length
CONTEXT_LENGTH: int = 64

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
# Libmelee stage IDs in the dataset reach into the mid-20s; allocate generously to avoid index errors.
N_STAGES: int = 64
N_BUTTONS: int = 5

# Model architecture (shrunk)
D_MODEL: int = 128
N_LAYERS: int = 2
N_HEADS: int = 4
MLP_RATIO: int = 3
OUTPUT_HIDDEN_DIM: int = 256
N_RESBLOCKS: int = 1
N_MIXTURE_COMPONENTS: int = 6

# Training
MIN_SIGMA: float = 0.05  # normalized units; allow sharp modes without collapse
MAX_SIGMA: float = 0.5   # normalized units; avoid over-dispersion
SIGMA_EPSILON: float = 0.01
MAX_GRAD_NORM: float = 1.0

# Feature Dimensions and Indices
PLAYER_FEATURES_DIM: int = 37 # Number of features extracted per player by extract_player_features
P_ACTION_STATE_IDX: int = 5 # Index of Action State ID within a player's feature vector
P_CHAR_PHYSICS_START_IDX: int = 29 # Start index of character physics fields

# Global Feature Indices within the final concatenated feature vector
# Assuming structure: [P1_features, P2_features, Global_Relational_features, Global_Static_Features]
# Global_Static_Features will include StageID, P1_CharID, P2_CharID
GLOBAL_RELATIONAL_FEATURES_DIM: int = 3 # distance, relative_x, relative_y
GLOBAL_FEATURES_DIM: int = 3 # Stage ID, P1 Char ID, P2 Char ID

# Combined feature vector structure for FeatureEmbedder input (from NPZ)
# [P1 features (37), P2 features (37), relational (3), stage_id (1), p1_char_id (1), p2_char_id (1)]
TOTAL_FEATURE_DIM: int = (2 * PLAYER_FEATURES_DIM) + GLOBAL_RELATIONAL_FEATURES_DIM + GLOBAL_FEATURES_DIM

# Indices within the TOTAL_FEATURE_DIM vector for categorical features that need embedding
# Note: P_ACTION_STATE_IDX is relative to a player's features, need to adjust for global vector
# Action state for P1: PLAYER_FEATURES_DIM * 0 + P_ACTION_STATE_IDX
# Action state for P2: PLAYER_FEATURES_DIM * 1 + P_ACTION_STATE_IDX
P1_ACTION_STATE_GLOBAL_IDX: int = P_ACTION_STATE_IDX
P2_ACTION_STATE_GLOBAL_IDX: int = PLAYER_FEATURES_DIM + P_ACTION_STATE_IDX

# Indices for character IDs and stage ID within the global part of the feature vector
STAGE_ID_GLOBAL_IDX: int = (2 * PLAYER_FEATURES_DIM) + GLOBAL_RELATIONAL_FEATURES_DIM
P1_CHAR_ID_GLOBAL_IDX: int = STAGE_ID_GLOBAL_IDX + 1
P2_CHAR_ID_GLOBAL_IDX: int = STAGE_ID_GLOBAL_IDX + 2

# Number of continuous features (after taking out all categorical features that will be embedded)
# Each player has (PLAYER_FEATURES_DIM - 1) continuous features (action state is categorical)
# Relational features are continuous (3)
# Global features are categorical (3: Stage, P1 Char, P2 Char)
NUM_CONTINUOUS_FEATURES = (2 * (PLAYER_FEATURES_DIM - 1)) + GLOBAL_RELATIONAL_FEATURES_DIM
# After embedding all categoricals, total dimension will be:
# NUM_CONTINUOUS_FEATURES
# + ACTION_STATE_EMBED_DIM * 2 (for P1 and P2)
# + STAGE_EMBED_DIM
# + CHARACTER_EMBED_DIM * 2 (for P1 and P2)
# This combined should be projected to D_MODEL

# NPZ field names
NPZ_FEATURES_KEY: str = "features"
NPZ_FUTURE_DELTAS_KEY: str = "future_deltas"
NPZ_VALID_MASK_KEY: str = "valid_mask"
NPZ_STAGE_KEY: str = "stage"
NPZ_P1_CHAR_KEY: str = "p1_char"
NPZ_P2_CHAR_KEY: str = "p2_char"
