"""
Tests for KV Cache implementation.

This test suite:
1. Verifies exact model outputs without cache (baseline)
2. Verifies KV cache produces identical outputs for ALiBi models
3. Verifies cache is disabled for RoPE models
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from config import get_config, init_config, reset_config
from model.nano_gpt import GPT
from schema import get_feature_names, get_raw_target_names, get_target_names
from train.batch_utils import build_model_inputs
from column_map import ColumnMap
from utils import _resolve_device, match_state_dict_keys
from zarr_storage import Schema, _process_episode_task


# Reference outputs from checkpoint model_ep013_000001.pt
# These are exact outputs that the KV cache must match

# Frame 0 (seq_len=64)
EXPECTED_MAIN_STICK_FRAME_0 = np.array([-0.8306569457054138, 0.5121105313301086, -1.3585995435714722, 4.921778202056885, 4.165994644165039, 0.20629391074180603, 0.1806170493364334, 0.155708909034729, -0.3208843469619751, -1.4553029537200928, -1.6985514163970947, -0.958165168762207, -0.7366436123847961, -2.5033929347991943, -3.335980176925659, -2.8867380619049072, -2.6521217823028564, -2.867656946182251, -2.809202194213867, -2.3838374614715576, -1.3234052658081055, -1.3121212720870972, -2.397413730621338, -2.0701966285705566, -0.9381019473075867, -2.618952751159668, -2.731304883956909, -2.8022170066833496, -2.5665721893310547, -2.442664861679077, -3.2076165676116943, -1.9666818380355835, -1.6991633176803589, -2.6334753036499023, -2.219123363494873, -2.5074470043182373, -3.1011478900909424, -2.6821303367614746, -3.2482895851135254, -3.0796990394592285, -2.9267539978027344, -2.6442277431488037, -3.0754268169403076, -1.5733695030212402, -3.0760717391967773, -3.0766336917877197, -2.9021832942962646, -3.346863031387329, -3.7925539016723633, -2.239638328552246, -0.015213601291179657, -2.699826955795288, -2.5983760356903076, -1.6858627796173096, -2.5148067474365234, -2.1098685264587402, -1.6693403720855713, -0.030449986457824707, -2.601022958755493, -2.632816791534424, 4.338954925537109, -0.20636850595474243, 2.9845170974731445, -0.2177901715040207])
EXPECTED_C_STICK_FRAME_0 = np.array([1.447733759880066, -5.1968817710876465, -4.980901718139648, -4.833549499511719, -4.754668235778809, -4.67753791809082, -5.155945777893066, -5.219568729400635, -5.267504692077637])
EXPECTED_BUTTONS_FRAME_0 = np.array([-5.6360063552856445, -6.6892409324646, -3.403089761734009, -8.447632789611816, -4.897694110870361])
EXPECTED_SHOULDER_FRAME_0 = np.array([0.9925485253334045, -4.895257472991943, -4.880113124847412, -4.737253189086914, -4.176901340484619])
EXPECTED_VALUE_FRAME_0 = np.array([-0.03821321576833725])

# Frame 31 (seq_len=64)
EXPECTED_MAIN_STICK_FRAME_31 = np.array([-1.8633660078048706, -0.8092457056045532, 1.1863641738891602, 5.550095558166504, -1.5781193971633911, -0.9997081756591797, -0.7146463394165039, 0.7905187010765076, 0.5197795629501343, 0.1655983179807663, -0.1527564525604248, -1.4994813203811646, -0.6552495360374451, -0.8481056690216064, -1.643477201461792, -2.035045623779297, -0.3237394690513611, -1.5378128290176392, -2.4079885482788086, -1.205424427986145, -0.38913917541503906, 4.801713943481445, 6.607621669769287, 5.7744550704956055, 3.8767383098602295, -0.329061359167099, -0.7309067845344543, -0.032426610589027405, -0.7342703938484192, -0.49471262097358704, -0.6317626237869263, -1.3528908491134644, -1.6200402975082397, -1.3201627731323242, -1.7374019622802734, -0.4630904793739319, -0.7167322635650635, -0.2899162471294403, -0.10289464145898819, -2.787868022918701, -0.3804457187652588, -1.4515674114227295, -0.39513587951660156, -1.0907692909240723, -0.39546167850494385, -0.39564546942710876, 1.1622488498687744, 0.2870149314403534, -0.45047158002853394, -0.45912933349609375, -0.04414834827184677, 0.4355621337890625, -1.6901582479476929, -0.9991293549537659, -0.7662875056266785, -1.271674394607544, 0.32665324211120605, -1.0059294700622559, -0.7136099338531494, -0.5968688130378723, -0.09244085848331451, -2.333815336227417, 1.0932849645614624, -1.7850878238677979])
EXPECTED_C_STICK_FRAME_31 = np.array([1.6415762901306152, -4.862422466278076, -4.774796962738037, -4.8123345375061035, -4.760606288909912, -4.722883701324463, -4.800519943237305, -4.8104681968688965, -4.7742767333984375])
EXPECTED_BUTTONS_FRAME_31 = np.array([-16.276338577270508, -14.641918182373047, -13.244035720825195, -17.110332489013672, -13.950730323791504])
EXPECTED_SHOULDER_FRAME_31 = np.array([1.0555245876312256, -4.811429500579834, -4.7046403884887695, -4.797066688537598, -4.860406398773193])
EXPECTED_VALUE_FRAME_31 = np.array([-0.33770430088043213])

# Frame 63 (seq_len=64, last frame)
EXPECTED_MAIN_STICK_FRAME_63 = np.array([-1.4332443475723267, -1.06789231300354, 2.151458501815796, 3.49544095993042, -2.1711575984954834, -1.1157863140106201, 6.3372039794921875, 3.8057024478912354, 5.6975016593933105, 0.5047382116317749, -3.4486727714538574, 1.8329534530639648, -0.966900110244751, -0.8153102993965149, -1.2559688091278076, -2.8009653091430664, -1.1393303871154785, 0.28158503770828247, -1.0817917585372925, -0.18988433480262756, -0.5946659445762634, -2.3837802410125732, -1.725681185722351, -0.0259877759963274, -0.4728770852088928, -2.187732458114624, 1.5808442831039429, 1.4479858875274658, 0.3329557478427887, -0.45079296827316284, -0.6738471984863281, 1.8762842416763306, -0.2560299038887024, -0.4448162317276001, -2.2818901538848877, -0.8596280813217163, -0.6018810272216797, 1.7915362119674683, -2.350543260574341, 0.5811882019042969, -0.5349310636520386, 0.6646634340286255, 0.36553335189819336, 0.4685525894165039, 0.36468446254730225, 0.36479485034942627, -0.21947439014911652, -1.0948418378829956, 0.0786265954375267, -0.8429306149482727, -1.8403873443603516, -0.6783123016357422, 0.957594633102417, 0.5910704731941223, -0.6963503956794739, -0.4020524024963379, 1.1116106510162354, 8.238298416137695, -0.44870850443840027, -0.7629361152648926, 1.3128539323806763, -1.9372761249542236, 0.11934418231248856, 0.9628885984420776])
EXPECTED_C_STICK_FRAME_63 = np.array([1.503814935684204, -5.559733867645264, -5.386806011199951, -3.4779131412506104, -5.342958450317383, -5.299296855926514, -5.26181697845459, -5.3330302238464355, -5.561084270477295])
EXPECTED_BUTTONS_FRAME_63 = np.array([-14.302864074707031, -15.966242790222168, -17.8900203704834, -19.910871505737305, 3.8292136192321777])
EXPECTED_SHOULDER_FRAME_63 = np.array([-0.2416037619113922, -3.389803647994995, -2.9730098247528076, -3.4343841075897217, 2.701099157333374])
EXPECTED_VALUE_FRAME_63 = np.array([0.12206155061721802])


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure each test sees a fresh config singleton."""
    reset_config()
    init_config(freeze=False)
    yield
    reset_config()


@pytest.fixture
def test_slp_path():
    """Path to the test .slp file."""
    return Path(__file__).with_name("test.slp")


@pytest.fixture
def checkpoint_path():
    """Path to the test checkpoint."""
    return Path(__file__).parent.parent / "checkpoints" / "model_ep013_000001.pt"


@pytest.fixture
def test_data(test_slp_path):
    """Load and process test data from test.slp."""
    schema = Schema(features=get_feature_names(), targets=get_raw_target_names())
    episodes = _process_episode_task(str(test_slp_path), schema)

    # Use the first episode (original perspective)
    episode = episodes[0]

    # Convert to torch tensors
    features = torch.from_numpy(episode.features).float()
    targets = torch.from_numpy(episode.targets).float()

    return features, targets, episode.feature_names, episode.target_names


@pytest.fixture
def model_and_config(checkpoint_path):
    """Load model from checkpoint."""
    device = _resolve_device()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config from checkpoint
    config = get_config()
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        model_cfg = ckpt_cfg.get("model", {})
        for field, value in model_cfg.items():
            setattr(config.model, field, value)

    # Create model
    model = GPT(config).to(device)

    # Load weights
    model_state = match_state_dict_keys(ckpt["model"], model)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    return model, config, device


def test_checkpoint_loads_successfully(model_and_config):
    """Verify checkpoint loads without errors."""
    model, config, device = model_and_config
    assert model is not None
    assert config is not None
    assert device is not None


def test_model_forward_pass_without_cache(model_and_config, test_data):
    """Test model forward pass produces exact expected outputs without cache.

    This establishes the baseline outputs that the KV cache must match.
    """
    model, config, device = model_and_config
    features, targets, feature_names, target_names = test_data

    # Create column map
    colmap = ColumnMap(feature_names, target_names)

    # Take a sequence of 64 frames
    seq_len = 64
    batch_X = features[:seq_len].unsqueeze(0)  # [1, seq_len, feature_dim]

    # Add horizon feature (normalized by 60.0)
    horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
    batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1)

    # Build model inputs
    batch_X_with_horizon = batch_X_with_horizon.to(device)
    inputs = build_model_inputs(batch_X_with_horizon, colmap)

    # Run forward pass (no gradients needed, no cache)
    with torch.inference_mode():
        outputs, cache = model(inputs, use_cache=False)

    # Verify no cache returned when use_cache=False
    assert cache is None

    # Verify outputs have expected keys
    assert "main_stick" in outputs
    assert "c_stick" in outputs
    assert "buttons" in outputs
    assert "shoulder" in outputs
    assert "value" in outputs

    # Verify output shapes
    batch_size, sequence_length = outputs.batch_size
    assert batch_size == 1
    assert sequence_length == seq_len

    # Extract outputs for specific frames
    main_stick_logits = outputs["main_stick"][0].cpu().numpy()
    c_stick_logits = outputs["c_stick"][0].cpu().numpy()
    button_logits = outputs["buttons"][0].cpu().numpy()
    shoulder_logits = outputs["shoulder"][0].cpu().numpy()
    value = outputs["value"][0].cpu().numpy()

    # Assert exact values for frame 0
    np.testing.assert_allclose(
        main_stick_logits[0], EXPECTED_MAIN_STICK_FRAME_0, rtol=1e-5, atol=1e-6,
        err_msg="Frame 0 main_stick mismatch"
    )
    np.testing.assert_allclose(
        c_stick_logits[0], EXPECTED_C_STICK_FRAME_0, rtol=1e-5, atol=1e-6,
        err_msg="Frame 0 c_stick mismatch"
    )
    np.testing.assert_allclose(
        button_logits[0], EXPECTED_BUTTONS_FRAME_0, rtol=1e-5, atol=1e-6,
        err_msg="Frame 0 buttons mismatch"
    )
    np.testing.assert_allclose(
        shoulder_logits[0], EXPECTED_SHOULDER_FRAME_0, rtol=1e-5, atol=1e-6,
        err_msg="Frame 0 shoulder mismatch"
    )
    np.testing.assert_allclose(
        value[0], EXPECTED_VALUE_FRAME_0, rtol=1e-5, atol=1e-6,
        err_msg="Frame 0 value mismatch"
    )

    # Assert exact values for frame 31 (middle)
    np.testing.assert_allclose(
        main_stick_logits[31], EXPECTED_MAIN_STICK_FRAME_31, rtol=1e-5, atol=1e-6,
        err_msg="Frame 31 main_stick mismatch"
    )
    np.testing.assert_allclose(
        c_stick_logits[31], EXPECTED_C_STICK_FRAME_31, rtol=1e-5, atol=1e-6,
        err_msg="Frame 31 c_stick mismatch"
    )
    np.testing.assert_allclose(
        button_logits[31], EXPECTED_BUTTONS_FRAME_31, rtol=1e-5, atol=1e-6,
        err_msg="Frame 31 buttons mismatch"
    )
    np.testing.assert_allclose(
        shoulder_logits[31], EXPECTED_SHOULDER_FRAME_31, rtol=1e-5, atol=1e-6,
        err_msg="Frame 31 shoulder mismatch"
    )
    np.testing.assert_allclose(
        value[31], EXPECTED_VALUE_FRAME_31, rtol=1e-5, atol=1e-6,
        err_msg="Frame 31 value mismatch"
    )

    # Assert exact values for frame 63 (last)
    np.testing.assert_allclose(
        main_stick_logits[63], EXPECTED_MAIN_STICK_FRAME_63, rtol=1e-5, atol=1e-6,
        err_msg="Frame 63 main_stick mismatch"
    )
    np.testing.assert_allclose(
        c_stick_logits[63], EXPECTED_C_STICK_FRAME_63, rtol=1e-5, atol=1e-6,
        err_msg="Frame 63 c_stick mismatch"
    )
    np.testing.assert_allclose(
        button_logits[63], EXPECTED_BUTTONS_FRAME_63, rtol=1e-5, atol=1e-6,
        err_msg="Frame 63 buttons mismatch"
    )
    np.testing.assert_allclose(
        shoulder_logits[63], EXPECTED_SHOULDER_FRAME_63, rtol=1e-5, atol=1e-6,
        err_msg="Frame 63 shoulder mismatch"
    )
    np.testing.assert_allclose(
        value[63], EXPECTED_VALUE_FRAME_63, rtol=1e-5, atol=1e-6,
        err_msg="Frame 63 value mismatch"
    )


def test_model_autoregressive_without_cache(model_and_config, test_data):
    """Test autoregressive generation without cache runs successfully.

    This simulates the inference scenario where we generate one token at a time,
    which is where KV cache provides speedup.
    """
    model, config, device = model_and_config
    features, targets, feature_names, target_names = test_data

    # Create column map
    colmap = ColumnMap(feature_names, target_names)

    # Autoregressive generation: start with a context, then predict next tokens
    context_len = 32
    num_predictions = 3  # Just a few steps to verify it works

    for step in range(num_predictions):
        # Current sequence includes context + all predictions so far
        seq_len = context_len + step
        batch_X = features[:seq_len].unsqueeze(0)

        # Add horizon feature
        horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
        batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1)

        # Build model inputs and run forward
        batch_X_with_horizon = batch_X_with_horizon.to(device)
        inputs = build_model_inputs(batch_X_with_horizon, colmap)

        with torch.inference_mode():
            outputs, _ = model(inputs, use_cache=False)

        # Verify output shapes for the last position
        assert outputs["main_stick"][0, -1].shape == (config.model.target_shapes_by_head["main_stick"],)
        assert outputs["c_stick"][0, -1].shape == (config.model.target_shapes_by_head["c_stick"],)
        assert outputs["buttons"][0, -1].shape == (config.model.target_shapes_by_head["buttons"],)
        assert outputs["shoulder"][0, -1].shape == (config.model.target_shapes_by_head["shoulder"],)
        assert outputs["value"][0, -1].shape == (1,)


def test_model_deterministic_without_cache(model_and_config, test_data):
    """Verify model outputs are deterministic (same inputs = same outputs)."""
    model, config, device = model_and_config
    features, targets, feature_names, target_names = test_data

    # Create column map
    colmap = ColumnMap(feature_names, target_names)

    # Prepare input
    seq_len = 64
    batch_X = features[:seq_len].unsqueeze(0)
    horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
    batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1).to(device)
    inputs = build_model_inputs(batch_X_with_horizon, colmap)

    # Run forward pass twice
    with torch.inference_mode():
        outputs1, _ = model(inputs, use_cache=False)
        outputs2, _ = model(inputs, use_cache=False)

    # Verify outputs are identical
    for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
        np.testing.assert_array_equal(
            outputs1[key].cpu().numpy(),
            outputs2[key].cpu().numpy(),
            err_msg=f"Output {key} is not deterministic"
        )


@pytest.mark.parametrize("seq_len", [16, 32, 64, 128])
def test_model_different_sequence_lengths(model_and_config, test_data, seq_len):
    """Verify model handles different sequence lengths correctly."""
    model, config, device = model_and_config
    features, targets, feature_names, target_names = test_data

    # Skip if test data doesn't have enough frames
    if features.shape[0] < seq_len:
        pytest.skip(f"Test data has only {features.shape[0]} frames, need {seq_len}")

    # Create column map
    colmap = ColumnMap(feature_names, target_names)

    # Prepare input
    batch_X = features[:seq_len].unsqueeze(0)
    horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
    batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1).to(device)
    inputs = build_model_inputs(batch_X_with_horizon, colmap)

    # Run forward pass
    with torch.inference_mode():
        outputs, _ = model(inputs, use_cache=False)

    # Verify output shapes
    for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
        assert outputs[key].shape[0] == 1  # batch size
        assert outputs[key].shape[1] == seq_len  # sequence length


def test_kv_cache_not_supported_with_rope(model_and_config, test_data):
    """Verify that KV cache raises an error when used with RoPE (not ALiBi)."""
    model, config, device = model_and_config
    features, targets, feature_names, target_names = test_data

    # Skip if model uses ALiBi (not RoPE)
    if config.model.use_alibi:
        pytest.skip("Model uses ALiBi, this test is for RoPE models")

    # Create column map
    colmap = ColumnMap(feature_names, target_names)

    # Prepare input
    seq_len = 32
    batch_X = features[:seq_len].unsqueeze(0)
    horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
    batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1).to(device)
    inputs = build_model_inputs(batch_X_with_horizon, colmap)

    # Trying to use cache with RoPE should raise an error
    with pytest.raises(ValueError, match="KV caching is only supported with ALiBi"):
        with torch.inference_mode():
            outputs, cache = model(inputs, use_cache=True)


def test_kv_cache_produces_identical_outputs_with_alibi():
    """
    Test that KV cache produces identical outputs to non-cached inference with ALiBi.

    This creates a fresh ALiBi model and verifies that:
    1. Running full sequence without cache produces output A
    2. Running autoregressive with cache produces output B
    3. A == B (outputs are identical)
    """
    # Create a small ALiBi model for testing
    reset_config()
    config = init_config(
        overrides={
            "model.use_alibi": "true",  # Enable ALiBi
            "model.block_size": "64",
            "model.n_embd": "128",
            "model.n_layer": "2",
            "model.n_head": "4",
            "model.n_kv_head": "4",
        }
    )

    device = torch.device("cpu")
    model = GPT(config).to(device)
    model.eval()

    # Create sample inputs (use zeros to avoid invalid embedding indices)
    batch_size = 1
    sequence_length = 16
    colmap = ColumnMap(get_feature_names(), get_target_names())
    X = torch.zeros(batch_size, sequence_length, len(colmap.feat_names) + 1, device=device)
    inputs = build_model_inputs(X, colmap)

    # Test 1: Full sequence without cache
    with torch.no_grad():
        outputs_no_cache, _ = model(inputs, use_cache=False)

    # Test 2: Autoregressive with cache
    # First, run full sequence with cache to initialize
    with torch.no_grad():
        outputs_with_cache, cache = model(inputs, use_cache=True)

    # Verify outputs are identical
    for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
        np.testing.assert_allclose(
            outputs_no_cache[key].cpu().numpy(),
            outputs_with_cache[key].cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Outputs for {key} differ between cached and non-cached"
        )

    # Test 3: Verify cache is actually populated
    assert cache is not None, "Cache should not be None when use_cache=True"
    assert len(cache) == config.model.n_layer, f"Cache should have {config.model.n_layer} layers"

    # Test 4: Autoregressive generation (simulate real inference)
    # First pass: context with multiple tokens
    # Subsequent passes: one token at a time, reusing cache
    context_len = 8
    generation_len = sequence_length - context_len

    # 4a. Initialize cache with context
    X_context = X[:, :context_len, :]
    inputs_context = build_model_inputs(X_context, colmap)
    with torch.no_grad():
        context_outputs, cache = model(inputs_context, kv_cache=None, use_cache=True)

    # Verify context outputs match
    for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
        np.testing.assert_allclose(
            outputs_no_cache[key][0, :context_len].cpu().numpy(),
            context_outputs[key][0].cpu().numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Context outputs for {key} differ"
        )

    # 4b. Generate remaining tokens autoregressively, one at a time
    for step in range(generation_len):
        # Only pass the NEW token (not the full sequence)
        current_pos = context_len + step
        X_new_token = X[:, current_pos:current_pos+1, :]
        inputs_new = build_model_inputs(X_new_token, colmap)

        with torch.no_grad():
            new_outputs, cache = model(inputs_new, kv_cache=cache, use_cache=True)

        # Verify this new token's output matches the full-sequence output
        for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
            np.testing.assert_allclose(
                outputs_no_cache[key][0, current_pos].cpu().numpy(),
                new_outputs[key][0, 0].cpu().numpy(),  # First (and only) position in output
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Autoregressive output at pos {current_pos} for {key} differs from full-sequence"
            )


def test_kv_cache_with_different_sequence_lengths_alibi():
    """Test that KV cache works correctly with different sequence lengths in ALiBi models."""
    reset_config()
    config = init_config(
        overrides={
            "model.use_alibi": "true",
            "model.block_size": "64",
            "model.n_embd": "128",
            "model.n_layer": "2",
            "model.n_head": "4",
            "model.n_kv_head": "4",
        }
    )

    device = torch.device("cpu")
    model = GPT(config).to(device)
    model.eval()

    colmap = ColumnMap(get_feature_names(), get_target_names())

    # Test with progressively longer sequences
    for seq_len in [8, 16, 32]:
        X = torch.zeros(1, seq_len, len(colmap.feat_names) + 1, device=device)
        inputs = build_model_inputs(X, colmap)

        # Without cache
        with torch.no_grad():
            outputs_no_cache, _ = model(inputs, use_cache=False)

        # With cache
        with torch.no_grad():
            outputs_with_cache, cache = model(inputs, use_cache=True)

        # Verify identical outputs
        for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
            np.testing.assert_allclose(
                outputs_no_cache[key].cpu().numpy(),
                outputs_with_cache[key].cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Mismatch at seq_len={seq_len} for {key}"
            )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
