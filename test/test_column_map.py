import pytest
from column_map import ColumnMap
from constants import CONTROLLER_KEY_GROUPS, BUTTON_TARGET_NAMES

# Minimal set of feature names required to instantiate ColumnMap without KeyErrors
MINIMAL_FEATURES = [
    "stage",
    "p1_character",
    "p2_character",
    "p1_action",
    "p2_action",
]
for prefix in ("p1_", "p2_"):
    for group in ("main", "c", "buttons", "shoulder"):
        for key in CONTROLLER_KEY_GROUPS[group]:
            MINIMAL_FEATURES.append(f"{prefix}{key}")

# Minimal set of target names
MINIMAL_TARGETS = ["p1_main_stick_idx", "p1_c_stick_idx", "p1_shoulder_idx"] + list(
    BUTTON_TARGET_NAMES
)


class MockIndex:
    """A mock for the dataset's index attribute."""

    def __init__(self, feature_names, target_names):
        self.feature_names = feature_names
        self.target_names = target_names


class MockDataset:
    """A mock dataset object for testing ColumnMap.from_dataset."""

    def __init__(self, index, feature_names_sel=None, target_names_sel=None):
        self.index = index
        if feature_names_sel is not None:
            self._feature_names_sel = feature_names_sel
        if target_names_sel is not None:
            self._target_names_sel = target_names_sel


def test_from_dataset_with_sel_attributes():
    """Tests that from_dataset uses _feature_names_sel and _target_names_sel when present."""
    index_features = MINIMAL_FEATURES + ["index_feat"]
    index_targets = MINIMAL_TARGETS + ["index_targ"]
    sel_features = MINIMAL_FEATURES + ["sel_feat"]
    sel_targets = MINIMAL_TARGETS + ["sel_targ"]

    mock_index = MockIndex(
        feature_names=index_features,
        target_names=index_targets,
    )
    mock_dataset = MockDataset(
        index=mock_index, feature_names_sel=sel_features, target_names_sel=sel_targets
    )
    column_map = ColumnMap.from_dataset(mock_dataset)

    assert column_map.feat_names == sel_features
    assert column_map.targ_names == sel_targets
    assert "sel_feat" in column_map.feat_names
    assert "index_feat" not in column_map.feat_names
