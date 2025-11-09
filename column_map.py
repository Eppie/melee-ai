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

        self.y_main = (_tid("p1_main_stick_x"), _tid("p1_main_stick_y"))
        self.y_c = (_tid("p1_c_stick_x"), _tid("p1_c_stick_y"))
        self.y_buttons = [_tid(name) for name in BUTTON_TARGET_NAMES]
        self.y_shoulder = targ2idx.get("p1_shoulder_analog")

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
