"""Shared feature/target column mapping utilities for train and inference."""

from __future__ import annotations

from typing import List, Sequence

from constants import CONTROLLER_KEY_GROUPS, BUTTON_TARGET_NAMES


class ColumnMap:
    """Resolves feature/target column names to indices for model inputs/outputs."""

    def __init__(
        self, feature_names: Sequence[str], target_names: Sequence[str]
    ) -> None:
        self.feat_names: List[str] = list(feature_names)
        self.targ_names: List[str] = list(target_names)

        name2idx = {n: i for i, n in enumerate(self.feat_names)}

        self.stage_idx = name2idx["stage"]
        self.ego_char_idx = name2idx["p1_character"]
        self.opp_char_idx = name2idx["p2_character"]
        self.ego_action_idx = name2idx["p1_action"]
        self.opp_action_idx = name2idx["p2_action"]
        self.value_idx = name2idx.get("value_target")

        controller_idxs: List[int] = []
        for prefix in ("p1_", "p2_"):
            for group in ("main", "c", "buttons", "shoulder"):
                for key in CONTROLLER_KEY_GROUPS[group]:
                    col_name = f"{prefix}{key}"
                    controller_idxs.append(name2idx[col_name])
        self.controller_idxs = controller_idxs

        excluded = {
            self.stage_idx,
            self.ego_char_idx,
            self.opp_char_idx,
            self.ego_action_idx,
            self.opp_action_idx,
            *self.controller_idxs,
        }
        if self.value_idx is not None:
            excluded.add(self.value_idx)
        self.gamestate_idxs = [
            i for i in range(len(self.feat_names)) if i not in excluded
        ]

        targ2idx = {n: i for i, n in enumerate(self.targ_names)}

        def _tid(name: str) -> int:
            if name not in targ2idx:
                raise KeyError(
                    f"Target '{name}' not found; available={self.targ_names}"
                )
            return targ2idx[name]

        # Only set up target indices if targets are provided
        if self.targ_names:
            self.y_main = (_tid("p1_main_stick_x"), _tid("p1_main_stick_y"))
            self.y_c = (_tid("p1_c_stick_x"), _tid("p1_c_stick_y"))
            self.y_buttons = [_tid(name) for name in BUTTON_TARGET_NAMES]
            self.y_shoulder = targ2idx.get("p1_shoulder_analog")
        else:
            # Set to None when no targets available (e.g., for reward computation)
            self.y_main = None
            self.y_c = None
            self.y_buttons = None
            self.y_shoulder = None

    @classmethod
    def from_dataset(
        cls, dataset
    ) -> "ColumnMap":  # dataset typing kept loose to avoid import cycle
        feature_names = getattr(
            dataset, "_feature_names_sel", dataset.index.feature_names
        )
        target_names = getattr(dataset, "_target_names_sel", dataset.index.target_names)
        return cls(feature_names, target_names)


__all__ = ["ColumnMap"]


"""
ColumnMap acts as a translator or a schema resolver. Its primary purpose is to bridge the gap between a high-dimensional, flat array of numerical data and the structured, semantic understanding of that data required by the model and other parts of the codebase.

The data is stored and loaded as a single large tensor X with many columns (features). The model, however, doesn't treat this as one big vector; it needs to know which specific columns correspond to the ego player's actions, the opponent's state, controller inputs, etc.

ColumnMap serves three main functions:

Name-to-Index Mapping: It converts human-readable string names (e.g., "p1_character") into the specific integer index of that column in the feature tensor. This prevents "magic numbers" (like X[..., 1]) from being scattered throughout the code.

Grouping of Indices: It goes beyond a simple dictionary by grouping related indices. For instance:

self.controller_idxs gathers all columns related to both players' controllers.

self.gamestate_idxs collects all numerical features that aren't categorical identifiers or controller inputs.

self.y_main, self.y_c, self.y_buttons group the target indices for different parts of the controller.

Centralization of Semantic Knowledge: The class hardcodes domain knowledge about the data. It knows that features prefixed with "p1_" belong to the "ego" player and that "stage" is a distinct categorical feature. This knowledge is crucial for slicing the flat tensor into meaningful chunks for the model's different embedding layers and processing steps.

Why Is It Necessary (in the Current Architecture)?

In the current architecture, ColumnMap is absolutely necessary. The entire data pipeline, from the WindowDataset to the model's forward pass (train.batch_utils.build_model_inputs), relies on a flat tensor representation for features.

Without ColumnMap, every part of the code would need to hardcode the integer indices for every feature. For example, in train/batch_utils.py, this line:

stage = batch_X[..., colmap.stage_idx].to(torch.long).unsqueeze(-1)

would have to be something like:

stage = batch_X[..., 0].to(torch.long).unsqueeze(-1) # Assuming stage is the first column

If you ever changed the order of columns in schema.py or zarr_storage.py, you would have to hunt down and update every single one of these hardcoded indices, which is extremely brittle and error-prone. ColumnMap solves this by centralizing the lookup.

However, the necessity of ColumnMap is a symptom of a larger architectural choice: representing structured data as a flat tensor.

Patterns to Render ColumnMap Unnecessary

The key to eliminating ColumnMap is to adopt a data representation that is inherently structured and self-describing, rather than relying on an external class to interpret a flat array.

1. The Structured Data Pattern (using TensorDict)

This is the most robust and modern approach. Instead of the dataset yielding a single X tensor, it would yield a dictionary-like object (such as tensordict.TensorDict) where each key corresponds to a semantic group of features.

How It Would Work:

Preprocessing (zarr_storage.py): Instead of creating a single wide X array in the Zarr store, you would create multiple, named arrays for each semantic group.

# Instead of this:
# epg.create_array("X", data=combined_features_array)

# Do this:
epg.create_array("gamestate", data=gamestate_array)
epg.create_array("controller", data=controller_array)
epg.create_array("categorical", data=categorical_array)

Dataset (window_dataset.py): The __getitem__ method would load these named arrays and bundle them into a TensorDict.

# Instead of returning {'X': tensor, 'Y': tensor}
return TensorDict({
    'gamestate': gamestate_tensor,   # shape [L, NumGameStateFeatures]
    'controller': controller_tensor, # shape [L, NumControllerFeatures]
    'stage': stage_tensor,           # shape [L, 1]
    # ... and so on
}, batch_size=[L])

Model Input (train/batch_utils.py): The build_model_inputs function becomes trivial or disappears entirely, as the data is already in the structured format the model needs. The model's forward pass would directly access features by name.

# Before
inputs_td = build_model_inputs(batch_X, colmap)
pred = model(inputs_td)

# After
# batch is already a TensorDict from the dataloader
pred = model(batch)

Impact:

ColumnMap is eliminated. All index lookups are replaced by direct key-based access (e.g., batch['gamestate']).

Code becomes highly readable and self-documenting.

Robust to change. Adding a new feature is as simple as adding a new key-value pair to the TensorDict. The fragile logic for calculating gamestate_idxs by exclusion is gone.

Decoupling. The model is no longer tightly coupled to the specific order of columns in a flat array.

2. The Self-Describing Schema Pattern

This is an intermediate solution that improves upon ColumnMap without requiring changes to the underlying flat tensor storage. You would replace ColumnMap with a more intelligent DataSchema class that derives its properties from a declarative schema definition rather than hardcoding them.

How It Would Work:

Define a Richer Schema: Instead of just a list of names, the schema would include tags or groups for each column.

# In a new schema.py
SCHEMA_DEFINITION = [
    ('stage', {'group': 'categorical', 'type': 'int'}),
    ('p1_character', {'group': 'categorical', 'player': 1}),
    ('p1_percent', {'group': 'gamestate', 'player': 1, 'type': 'float'}),
    ('p1_main_stick_x', {'group': 'controller', 'player': 1, 'type': 'float'}),
    # etc.
]

Create a Dynamic DataSchema Class: This class would replace ColumnMap. Its __init__ would parse the SCHEMA_DEFINITION to build the name-to-index map. It would provide methods to query indices based on tags instead of having hardcoded attributes.

class DataSchema:
    def __init__(self, definition):
        self.name_to_idx = {name: i for i, (name, _) in enumerate(definition)}
        self.idx_to_meta = {i: meta for i, (_, meta) in enumerate(definition)}

    def get_indices(self, group: str, player: Optional[int] = None) -> List[int]:
        indices = []
        for i, meta in self.idx_to_meta.items():
            if meta.get('group') == group:
                if player is None or meta.get('player') == player:
                    indices.append(i)
        return indices

Impact:

ColumnMap is replaced by a robust, dynamic DataSchema class.

Reduces hardcoding and fragility. Adding a new "gamestate" feature only requires adding it to SCHEMA_DEFINITION with the correct tag; get_indices('gamestate') will automatically find it.

Still requires an index mapping class, but its internal logic is generic and data-driven rather than bespoke.

Summary of Patterns
Pattern	Description	Impact on column_map.py
Current Architecture	A flat tensor for features requires an external class to resolve names and groups to integer indices.	Necessary. Centralizes brittle, hardcoded logic.
1. Structured Data	The dataset yields a dictionary-like object (TensorDict) with named tensors for each feature group.	Eliminated. Data is self-describing; access is by key.
2. Self-Describing Schema	ColumnMap is replaced by a class that dynamically queries a rich, declarative schema definition.	Replaced. Becomes a dynamic query tool, not hardcoded.

For this codebase, migrating to the Structured Data Pattern with TensorDict is the recommended architectural improvement. It would significantly clean up the data pipeline, remove layers of indirection, and make the entire system more robust and easier to understand.
"""
