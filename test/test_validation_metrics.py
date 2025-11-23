import numpy as np
import pytest
import torch
from typing import Tuple

from column_map import ColumnMap
from config.config import get_config, init_config, reset_config
from constants import CONTROLLER_KEY_GROUPS, BUTTON_TARGET_NAMES
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from libmelee.melee.enums import Action
from train.value_head import build_reward_feature_index, compute_value_targets
from validation import (
    EnhancedMetrics,
    _categorize_action,
    _encode_state_codes,
    _frame_rewards_from_batch,
    _tally_jump_types,
    _update_enhanced_metrics,
)


def _build_feature_names(include_value_target: bool = True) -> list[str]:
    base = [
        "stage",
        "p1_character",
        "p2_character",
        "p1_action",
        "p2_action",
        "p1_stock",
        "p2_stock",
        "p1_percent",
        "p2_percent",
        "p1_is_in_hitlag",
        "p2_is_in_hitlag",
        "p1_is_defender_in_hitlag",
        "p2_is_defender_in_hitlag",
        "p1_shield_strength",
        "p2_shield_strength",
        "p1_off_stage",
        "p2_off_stage",
        "p1_l_cancel_status",
        "p1_on_ground",
        "p2_on_ground",
    ]

    controller_features: list[str] = []
    for prefix in ("p1_", "p2_"):
        for group in ("main", "c", "buttons", "shoulder"):
            for key in CONTROLLER_KEY_GROUPS[group]:
                controller_features.append(f"{prefix}{key}")

    names = base + controller_features
    if include_value_target:
        names.append("value_target")
    return names


TARGET_NAMES = [
    "p1_main_stick_x",
    "p1_main_stick_y",
    "p1_c_stick_x",
    "p1_c_stick_y",
    *BUTTON_TARGET_NAMES,
    "p1_shoulder_analog",
]


@pytest.fixture(autouse=True)
def _init_global_config():
    reset_config()
    cfg = init_config(freeze=False)
    yield cfg
    reset_config()


@pytest.fixture()
def column_map() -> ColumnMap:
    return ColumnMap(_build_feature_names(), TARGET_NAMES)


def _set_feature(X: torch.Tensor, feat_idx: dict[str, int], name: str, values) -> None:
    X[..., feat_idx[name]] = torch.as_tensor(values, dtype=X.dtype)


def test_update_enhanced_metrics_accumulates_multimodal_statistics(
    column_map: ColumnMap,
):
    reward_features = build_reward_feature_index(column_map)
    enhanced = EnhancedMetrics()
    B, L = 2, 3
    feat_idx = {name: idx for idx, name in enumerate(column_map.feat_names)}
    X = torch.zeros((B, L, len(column_map.feat_names)), dtype=torch.float32)

    _set_feature(
        X,
        feat_idx,
        "p1_action",
        [
            [
                Action.STANDING.value,
                Action.NEUTRAL_ATTACK_1.value,
                Action.DAMAGE_LOW_1.value,
            ],
            [Action.WALK_SLOW.value, Action.CROUCHING.value, Action.DASH_ATTACK.value],
        ],
    )
    _set_feature(X, feat_idx, "p2_action", [[0, 0, 0], [0, 0, 0]])
    _set_feature(X, feat_idx, "p1_stock", [[4, 4, 3], [4, 4, 4]])
    _set_feature(X, feat_idx, "p2_stock", [[4, 3, 3], [4, 4, 3]])
    _set_feature(X, feat_idx, "p1_percent", [[10, 30, 60], [5, 5, 20]])
    _set_feature(X, feat_idx, "p2_percent", [[0, 10, 25], [0, 5, 5]])
    _set_feature(X, feat_idx, "p1_is_in_hitlag", [[0, 1, 0], [0, 0, 0]])
    _set_feature(X, feat_idx, "p2_is_in_hitlag", [[0, 0, 1], [0, 1, 0]])
    _set_feature(X, feat_idx, "p1_is_defender_in_hitlag", [[0, 0, 0], [0, 0, 0]])
    _set_feature(X, feat_idx, "p2_is_defender_in_hitlag", [[0, 0, 0], [0, 0, 0]])
    _set_feature(X, feat_idx, "p1_shield_strength", [[0.2, 0.4, 0.1], [0.5, 0.5, 0.5]])
    _set_feature(X, feat_idx, "p2_shield_strength", [[0.4, 0.4, 0.4], [0.5, 0.4, 0.4]])
    _set_feature(X, feat_idx, "value_target", [[0.5, 0.7, 1.0], [1.1, 1.2, 1.3]])

    target_main = torch.tensor([[0, 1, 1], [2, 2, 3]], dtype=torch.long)
    target_c = torch.tensor([[0, 1, 2], [3, 3, 4]], dtype=torch.long)
    target_btn = torch.tensor(
        [
            [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 1]],
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0]],
        ],
        dtype=torch.long,
    )
    btn_pred = torch.tensor(
        [
            [[1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 0, 1]],
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        ],
        dtype=torch.long,
    )

    pred_main_idx = torch.tensor([[0, 0, 1], [2, 3, 3]], dtype=torch.long)
    pred_c_idx = torch.tensor([[1, 1, 2], [2, 4, 4]], dtype=torch.long)

    vocab_main = len(CONTROL_STICK_QUANTIZED)
    vocab_c = len(C_STICK_QUANTIZED)
    logits_main = (
        torch.linspace(-1.0, 1.0, vocab_main).repeat(B * L, 1).reshape(B, L, vocab_main)
    )
    logits_main += torch.arange(B * L, dtype=torch.float32).reshape(B, L, 1) * 0.05
    logits_c = (
        torch.linspace(-0.5, 0.5, vocab_c).repeat(B * L, 1).reshape(B, L, vocab_c)
    )
    logits_c += torch.arange(B * L, dtype=torch.float32).reshape(B, L, 1) * 0.1
    shoulder_vocab = len(SHOULDER_QUANTIZED)
    logits_shoulder = (
        torch.linspace(-0.2, 0.2, shoulder_vocab)
        .repeat(B * L, 1)
        .reshape(B, L, shoulder_vocab)
    )
    logits_shoulder += torch.arange(B * L, dtype=torch.float32).reshape(B, L, 1) * 0.02

    value_pred = torch.tensor(
        [[[0.6], [0.8], [1.05]], [[0.95], [1.3], [1.4]]], dtype=torch.float32
    )

    prev_pred_main = torch.tensor([[0.0, 0.0], [0.5, -0.5]], dtype=torch.float32)
    prev_true_main = torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float32)
    prev_pred_c = torch.tensor([[0.0, 0.0], [0.7, -0.7]], dtype=torch.float32)
    prev_true_c = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    target_info = {
        "main_idx": target_main.clone(),
        "c_idx": target_c.clone(),
        "buttons": target_btn.clone(),
        "shoulder_idx": torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long),
    }

    results = _update_enhanced_metrics(
        enhanced,
        X,
        target_info,
        pred_main_idx,
        pred_c_idx,
        btn_pred,
        logits_main,
        logits_c,
        logits_shoulder,
        column_map,
        column_map.value_idx,
        reward_features,
        prev_pred_main,
        prev_pred_c,
        prev_true_main,
        prev_true_c,
        value_pred,
    )

    main_palette = torch.tensor(CONTROL_STICK_QUANTIZED, dtype=torch.float32)
    c_palette = torch.tensor(C_STICK_QUANTIZED, dtype=torch.float32)
    main_pred_coords = main_palette[pred_main_idx.reshape(-1)].reshape(B, L, 2)
    main_true_coords = main_palette[target_main.reshape(-1)].reshape(B, L, 2)
    c_pred_coords = c_palette[pred_c_idx.reshape(-1)].reshape(B, L, 2)
    c_true_coords = c_palette[target_c.reshape(-1)].reshape(B, L, 2)

    main_errors = torch.linalg.norm(main_pred_coords - main_true_coords, dim=-1)
    c_errors = torch.linalg.norm(c_pred_coords - c_true_coords, dim=-1)
    assert enhanced.total_main_stick_error == pytest.approx(main_errors.sum().item())
    assert enhanced.total_c_stick_error == pytest.approx(c_errors.sum().item())

    main_change_mask = torch.zeros_like(target_main, dtype=torch.bool)
    main_change_mask[:, 1:] = target_main[:, 1:] != target_main[:, :-1]
    main_hold_mask = ~main_change_mask
    c_change_mask = torch.zeros_like(target_c, dtype=torch.bool)
    c_change_mask[:, 1:] = target_c[:, 1:] != target_c[:, :-1]
    c_hold_mask = ~c_change_mask
    assert enhanced.total_main_stick_error_change == pytest.approx(
        main_errors[main_change_mask].sum().item()
    )
    assert enhanced.total_main_stick_error_hold == pytest.approx(
        main_errors[main_hold_mask].sum().item()
    )
    assert enhanced.total_c_stick_error_change == pytest.approx(
        c_errors[c_change_mask].sum().item()
    )
    assert enhanced.total_c_stick_error_hold == pytest.approx(
        c_errors[c_hold_mask].sum().item()
    )

    cross_pred_main = (
        torch.linalg.norm(main_pred_coords[:, 0] - prev_pred_main, dim=-1).sum().item()
    )
    cross_true_main = (
        torch.linalg.norm(main_true_coords[:, 0] - prev_true_main, dim=-1).sum().item()
    )
    cross_pred_c = (
        torch.linalg.norm(c_pred_coords[:, 0] - prev_pred_c, dim=-1).sum().item()
    )
    cross_true_c = (
        torch.linalg.norm(c_true_coords[:, 0] - prev_true_c, dim=-1).sum().item()
    )
    within_pred_main = (
        torch.linalg.norm(main_pred_coords[:, 1:] - main_pred_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    within_true_main = (
        torch.linalg.norm(main_true_coords[:, 1:] - main_true_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    within_pred_c = (
        torch.linalg.norm(c_pred_coords[:, 1:] - c_pred_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    within_true_c = (
        torch.linalg.norm(c_true_coords[:, 1:] - c_true_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )

    assert enhanced.total_pred_main_jitter == pytest.approx(
        cross_pred_main + within_pred_main
    )
    assert enhanced.total_true_main_jitter == pytest.approx(
        cross_true_main + within_true_main
    )
    assert enhanced.total_pred_c_jitter == pytest.approx(cross_pred_c + within_pred_c)
    assert enhanced.total_true_c_jitter == pytest.approx(cross_true_c + within_true_c)
    assert enhanced.jitter_frames == B + B * (L - 1)

    probs_main = torch.softmax(logits_main, dim=-1)
    probs_c = torch.softmax(logits_c, dim=-1)
    probs_sh = torch.softmax(logits_shoulder, dim=-1)
    entropy_main = -(probs_main * torch.log(probs_main + 1e-9)).sum().item()
    entropy_c = -(probs_c * torch.log(probs_c + 1e-9)).sum().item()
    entropy_sh = -(probs_sh * torch.log(probs_sh + 1e-9)).sum().item()
    assert enhanced.total_main_entropy == pytest.approx(entropy_main)
    assert enhanced.total_c_entropy == pytest.approx(entropy_c)
    assert enhanced.total_shoulder_entropy == pytest.approx(entropy_sh)
    assert enhanced.entropy_frames == B * L

    actions = X[..., column_map.ego_action_idx].flatten().tolist()
    correct_mask = (pred_main_idx == target_main).flatten().tolist()
    expected_total = {}
    expected_correct = {}
    for action, correct in zip(actions, correct_mask):
        category = _categorize_action(int(action))
        expected_total[category] = expected_total.get(category, 0) + 1
        if correct:
            expected_correct[category] = expected_correct.get(category, 0) + 1
    assert dict(enhanced.state_total) == expected_total
    assert dict(enhanced.state_correct) == expected_correct

    pred_codes = _encode_state_codes(pred_main_idx, pred_c_idx, btn_pred)
    true_codes = _encode_state_codes(target_main, target_c, target_btn)
    for stats, codes in (
        (enhanced.pred_run_stats, pred_codes),
        (enhanced.true_run_stats, true_codes),
    ):
        codes_np = codes.reshape(-1).cpu().numpy()
        if codes_np.size == 0:
            continue
        diffs = np.diff(codes_np)
        change_points = np.nonzero(diffs)[0] + 1
        starts = np.concatenate(([0], change_points))
        lengths = np.diff(np.concatenate((starts, [codes_np.size])))
        if lengths.size > 0:
            assert stats.pending_length == int(lengths[-1])
            assert stats.pending_value == int(codes_np[starts[-1]])
            recorded = lengths[:-1]
            assert stats.total_length == int(recorded.sum())
            assert stats.run_count == int(recorded.size)

    preds_vec = np.concatenate(
        [
            main_pred_coords.reshape(-1, 2).numpy(),
            c_pred_coords.reshape(-1, 2).numpy(),
            btn_pred.reshape(-1, btn_pred.shape[-1]).numpy().astype(np.float32),
        ],
        axis=1,
    )
    labels_vec = np.concatenate(
        [
            main_true_coords.reshape(-1, 2).numpy(),
            c_true_coords.reshape(-1, 2).numpy(),
            target_btn.reshape(-1, target_btn.shape[-1]).numpy(),
        ],
        axis=1,
    )
    assert len(enhanced.all_preds_list) == 1
    assert len(enhanced.all_labels_list) == 1
    assert np.allclose(enhanced.all_preds_list[0], preds_vec)
    assert np.allclose(enhanced.all_labels_list[0], labels_vec)

    assert enhanced.total_frames == B * L

    config = get_config()
    value_target = X[..., column_map.value_idx].unsqueeze(-1)
    mse = ((value_pred - value_target) ** 2).mean().item()
    mae = (value_pred - value_target).abs().mean().item()
    assert enhanced.total_value_mse == pytest.approx(mse * (B * L))
    assert enhanced.total_value_mae == pytest.approx(mae * (B * L))
    assert enhanced.total_value_pred == pytest.approx(value_pred.sum().item())
    assert enhanced.total_value_target == pytest.approx(value_target.sum().item())

    flat_pred = value_pred.cpu().numpy().flatten().tolist()
    flat_target = value_target.cpu().numpy().flatten().tolist()
    assert enhanced.value_pred_list == pytest.approx(flat_pred)
    assert enhanced.value_target_list == pytest.approx(flat_target)
    assert enhanced.value_frames == B * L

    frame_rewards = _frame_rewards_from_batch(X, column_map, reward_features)
    assert len(enhanced.value_frame_data) == B * L
    first_entry = enhanced.value_frame_data[0]
    assert first_entry[0] == pytest.approx(value_pred[0, 0, 0].item())
    assert first_entry[1] == pytest.approx(value_target[0, 0, 0].item())
    assert first_entry[2] == pytest.approx(frame_rewards[0, 0].item())
    assert isinstance(first_entry[3], np.ndarray)
    assert first_entry[3].shape[0] == len(column_map.feat_names)

    target_lr = target_btn[:, :, 4]
    pred_lr = btn_pred[:, :, 4]
    true_changes = ((target_lr[:, :-1] == 0) & (target_lr[:, 1:] == 1)).cpu()
    pred_changes = ((pred_lr[:, :-1] == 0) & (pred_lr[:, 1:] == 1)).cpu()
    batch_offsets = torch.arange(B).unsqueeze(1) * L
    frame_offsets = torch.arange(1, L).unsqueeze(0)
    frame_indices = (batch_offsets + frame_offsets).tolist()
    expected_true_frames = [
        frame_indices[b][i]
        for b in range(B)
        for i, changed in enumerate(true_changes[b].tolist())
        if changed
    ]
    expected_pred_frames = [
        frame_indices[b][i]
        for b in range(B)
        for i, changed in enumerate(pred_changes[b].tolist())
        if changed
    ]
    assert enhanced.lr_button_changes_true == expected_true_frames
    assert enhanced.lr_button_changes_pred == expected_pred_frames

    last_pred_main, last_pred_c, last_true_main, last_true_c = results
    assert torch.allclose(last_pred_main, main_pred_coords[:, -1])
    assert torch.allclose(last_pred_c, c_pred_coords[:, -1])
    assert torch.allclose(last_true_main, main_true_coords[:, -1])
    assert torch.allclose(last_true_c, c_true_coords[:, -1])


def test_update_enhanced_metrics_handles_single_frame_batches_and_copies_frame_features(
    column_map: ColumnMap,
):
    reward_features = build_reward_feature_index(column_map)
    enhanced = EnhancedMetrics()
    B = L = 1
    feat_idx = {name: idx for idx, name in enumerate(column_map.feat_names)}
    X = torch.zeros((B, L, len(column_map.feat_names)), dtype=torch.float32)

    _set_feature(X, feat_idx, "p1_action", [[Action.DEAD_DOWN.value]])
    _set_feature(X, feat_idx, "p2_action", [[0]])
    _set_feature(X, feat_idx, "p1_stock", [[4]])
    _set_feature(X, feat_idx, "p2_stock", [[4]])
    _set_feature(X, feat_idx, "p1_percent", [[10]])
    _set_feature(X, feat_idx, "p2_percent", [[5]])
    _set_feature(X, feat_idx, "p1_is_in_hitlag", [[0]])
    _set_feature(X, feat_idx, "p2_is_in_hitlag", [[0]])
    _set_feature(X, feat_idx, "p1_is_defender_in_hitlag", [[0]])
    _set_feature(X, feat_idx, "p2_is_defender_in_hitlag", [[0]])
    _set_feature(X, feat_idx, "p1_shield_strength", [[0.3]])
    _set_feature(X, feat_idx, "p2_shield_strength", [[0.4]])

    target_main = torch.tensor([[5]], dtype=torch.long)
    target_c = torch.tensor([[2]], dtype=torch.long)
    target_btn = torch.tensor([[[1, 0, 0, 0, 0]]], dtype=torch.long)
    btn_pred = torch.tensor([[[0, 0, 0, 0, 0]]], dtype=torch.long)

    pred_main_idx = torch.tensor([[6]], dtype=torch.long)
    pred_c_idx = torch.tensor([[3]], dtype=torch.long)

    vocab_main = len(CONTROL_STICK_QUANTIZED)
    vocab_c = len(C_STICK_QUANTIZED)
    logits_main = torch.linspace(-1.0, 1.0, vocab_main).reshape(1, 1, vocab_main)
    logits_c = torch.linspace(-0.5, 0.5, vocab_c).reshape(1, 1, vocab_c)
    shoulder_vocab = len(SHOULDER_QUANTIZED)
    logits_shoulder = torch.zeros((1, 1, shoulder_vocab))

    value_pred = torch.tensor([[[0.25]]], dtype=torch.float32)
    target_info = {
        "main_idx": target_main.clone(),
        "c_idx": target_c.clone(),
        "buttons": target_btn.clone(),
    }

    X_before = X.clone()

    results = _update_enhanced_metrics(
        enhanced,
        X,
        target_info,
        pred_main_idx,
        pred_c_idx,
        btn_pred,
        logits_main,
        logits_c,
        logits_shoulder,
        column_map,
        value_idx=None,
        reward_features=reward_features,
        prev_pred_main_coords=None,
        prev_pred_c_coords=None,
        prev_true_main_coords=None,
        prev_true_c_coords=None,
        value_pred=value_pred,
    )

    main_palette = torch.tensor(CONTROL_STICK_QUANTIZED, dtype=torch.float32)
    c_palette = torch.tensor(C_STICK_QUANTIZED, dtype=torch.float32)
    main_errors = torch.linalg.norm(
        main_palette[pred_main_idx] - main_palette[target_main], dim=-1
    )
    c_errors = torch.linalg.norm(c_palette[pred_c_idx] - c_palette[target_c], dim=-1)
    assert enhanced.total_main_stick_error_hold == pytest.approx(
        main_errors.sum().item()
    )
    assert enhanced.total_main_stick_error_change == 0.0
    assert enhanced.total_c_stick_error_hold == pytest.approx(c_errors.sum().item())
    assert enhanced.total_c_stick_error_change == 0.0
    assert enhanced.jitter_frames == 0

    assert enhanced.lr_button_changes_true == []
    assert enhanced.lr_button_changes_pred == []

    assert enhanced.state_total["other"] == 1
    assert enhanced.state_correct.get("other", 0) == 0

    assert enhanced.total_frames == B * L
    assert enhanced.entropy_frames == B * L
    assert enhanced.value_frames == B * L

    cfg = get_config()
    expected_value_target = compute_value_targets(
        X_before,
        column_map,
        gamma=cfg.rl.gamma,
        reward_idx=None,
        reward_features=reward_features,
    )
    expected_frame_rewards = _frame_rewards_from_batch(
        X_before, column_map, reward_features
    )
    mse = ((value_pred - expected_value_target) ** 2).mean().item()
    mae = (value_pred - expected_value_target).abs().mean().item()
    assert enhanced.total_value_mse == pytest.approx(mse * (B * L))
    assert enhanced.total_value_mae == pytest.approx(mae * (B * L))
    assert enhanced.total_value_target == pytest.approx(
        expected_value_target.sum().item()
    )

    assert enhanced.value_pred_list == pytest.approx([value_pred.item()])
    assert enhanced.value_target_list == pytest.approx(
        expected_value_target.flatten().tolist()
    )

    assert len(enhanced.value_frame_data) == 1
    frame_entry = enhanced.value_frame_data[0]
    assert frame_entry[0] == pytest.approx(value_pred.item())
    assert frame_entry[1] == pytest.approx(expected_value_target.item())
    assert frame_entry[2] == pytest.approx(expected_frame_rewards.item())
    assert isinstance(frame_entry[3], np.ndarray)
    X[0, 0, feat_idx["stage"]] = 99.0
    assert frame_entry[3][feat_idx["stage"]] == pytest.approx(
        X_before[0, 0, feat_idx["stage"]]
    )

    pred_codes = _encode_state_codes(pred_main_idx, pred_c_idx, btn_pred)
    true_codes = _encode_state_codes(target_main, target_c, target_btn)
    assert enhanced.pred_run_stats.pending_length == 1
    assert enhanced.pred_run_stats.pending_value == int(pred_codes.item())
    assert enhanced.pred_run_stats.total_length == 0
    assert enhanced.pred_run_stats.run_count == 0
    assert enhanced.true_run_stats.pending_length == 1
    assert enhanced.true_run_stats.pending_value == int(true_codes.item())

    preds_vec = np.concatenate(
        [
            main_palette[pred_main_idx].reshape(-1, 2).numpy(),
            c_palette[pred_c_idx].reshape(-1, 2).numpy(),
            btn_pred.reshape(-1, btn_pred.shape[-1]).numpy().astype(np.float32),
        ],
        axis=1,
    )
    labels_vec = np.concatenate(
        [
            main_palette[target_main].reshape(-1, 2).numpy(),
            c_palette[target_c].reshape(-1, 2).numpy(),
            target_btn.reshape(-1, target_btn.shape[-1]).numpy(),
        ],
        axis=1,
    )
    assert np.allclose(enhanced.all_preds_list[0], preds_vec)
    assert np.allclose(enhanced.all_labels_list[0], labels_vec)

    last_pred_main, last_pred_c, last_true_main, last_true_c = results
    assert torch.allclose(last_pred_main, main_palette[pred_main_idx].reshape(B, 2))
    assert torch.allclose(last_pred_c, c_palette[pred_c_idx].reshape(B, 2))
    assert torch.allclose(last_true_main, main_palette[target_main].reshape(B, 2))
    assert torch.allclose(last_true_c, c_palette[target_c].reshape(B, 2))


def test_tally_jump_types_tracks_confusions_and_misses():
    enhanced = EnhancedMetrics()
    num_buttons = len(BUTTON_TARGET_NAMES)
    B, L = 1, 20
    target_btn = torch.zeros((B, L, num_buttons), dtype=torch.long)
    btn_pred = torch.zeros_like(target_btn)
    xy_idx = BUTTON_TARGET_NAMES.index("p1_button_xy")

    def _mark(window: Tuple[int, int], tensor: torch.Tensor) -> None:
        start, end = window
        tensor[:, start:end, xy_idx] = 1

    # Short-hop runs: [0-2), [3-5), [6-8)
    _mark((0, 2), target_btn)
    _mark((3, 5), target_btn)
    _mark((6, 8), target_btn)
    # Full-hop runs: [9-12), [13-16), [17-20)
    _mark((9, 12), target_btn)
    _mark((13, 16), target_btn)
    _mark((17, 20), target_btn)

    # Model behavior
    _mark((0, 3), btn_pred)  # Full hop when short desired
    # Missed short hop at [3,5)
    _mark((6, 8), btn_pred)  # Correct short hop
    _mark((9, 11), btn_pred)  # Short hop when full desired
    # Missed full hop at [13,16)
    _mark((17, 20), btn_pred)  # Correct full hop

    grounded_mask = torch.ones((B, L), dtype=torch.bool)
    _tally_jump_types(enhanced, target_btn, btn_pred, grounded_mask)

    assert enhanced.jumps_true_short == 3
    assert enhanced.jumps_true_full == 3
    assert enhanced.jumps_pred_short_correct == 1
    assert enhanced.jumps_pred_full_correct == 1
    assert enhanced.jumps_pred_full_when_short == 1
    assert enhanced.jumps_missed_short == 1
    assert enhanced.jumps_pred_short_when_full == 1
    assert enhanced.jumps_missed_full == 1
