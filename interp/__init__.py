"""
Interpretability toolkit for Nano-Melee GPT model.

This module provides tools for understanding model decisions:

Phase 1 (Foundation):
- Activation hooking and caching
- TopK Sparse Autoencoders for feature discovery
- Decision explanation for individual frames

Phase 2 (Causal Analysis):
- Activation patching for causal tracing
- Attention pattern extraction and visualization
- Logit lens for prediction evolution analysis

Phase 3 (Probes & Labeling):
- Linear probes for detecting game concepts
- Smart frame sampling (diversity, uncertainty)
- Terminal labeling interface

Phase 4 (Disentanglement):
- P1/P2 feature separation analysis
- Player swap effect measurement
- Opponent modeling feature discovery

Phase 5 (Circuit Analysis):
- Behavior circuit definitions (wavedash, recovery, etc.)
- Layer importance for specific behaviors
- SAE feature circuit analysis

Quick Start:
    from interp import DecisionExplainer, TopKSparseAutoencoder, train_sae

    # Train an SAE on model activations
    sae, history = train_sae(model, dataloader, colmap, hook_point, config, device)

    # Explain model decisions
    explainer = DecisionExplainer(model, colmap, device, sae, hook_point)
    explanation = explainer.explain_frame(inputs)
    print(explanation.summary())

    # Causal tracing - which layers matter?
    from interp import ActivationPatcher
    patcher = ActivationPatcher(model, colmap, device)
    trace = patcher.trace_all_layers(clean_inputs, corrupted_inputs)
    print(f"Critical layer: {trace.get_critical_layer()}")

    # When did the model decide?
    from interp import LogitLens
    lens = LogitLens(model, colmap, device)
    evolution = lens.analyze(inputs)
    print(f"Crystallized at layer: {evolution.get_crystallization_layer()}")

    # Train probes for game concepts
    from interp import LinearProbe, ProbeTrainer, MELEE_PROBES
    config = MELEE_PROBES["game_phase"]
    probe = LinearProbe(input_dim=512, num_classes=config.num_classes)
    trainer = ProbeTrainer(probe, config, device)
    history = trainer.train(activations, labels)

    # Label frames interactively
    from interp.labeling import FrameSampler, LabelStorage, TerminalLabeler
    sampler = FrameSampler(model, dataset, colmap, device)
    frames = sampler.sample(100, strategy="diversity")
    storage = LabelStorage("labels/game_phase.json", probe_name="game_phase")
    labeler = TerminalLabeler(config, storage, frames.frames)
    labeler.run()

    # Analyze P1/P2 disentanglement
    from interp import EntityDisentangler
    disentangler = EntityDisentangler(model, colmap, device)
    swap_effect = disentangler.analyze_swap_effect(inputs)
    print(f"Prediction changed on swap: {swap_effect.any_prediction_changed}")
    summary = disentangler.full_analysis(inputs)
    print(summary.summary())

    # Analyze behavior circuits
    from interp import CircuitAnalyzer, MELEE_CIRCUITS
    analyzer = CircuitAnalyzer(model, colmap, device)
    wavedash = MELEE_CIRCUITS["wavedash"]
    analysis = analyzer.analyze_circuit(wavedash, dataloader)
    print(analysis.summary())
"""

from interp.config import InterpConfig, SAEConfig, SamplingConfig
from interp.hooks import (
    HookPoint,
    HookPointType,
    HookManager,
    all_block_outputs,
    residual_stream_hooks,
    mlp_activation_hooks,
)
from interp.cache import ActivationCache, CachedActivations
from interp.sae.topk import TopKSparseAutoencoder, SAEOutput
from interp.sae.trainer import SAETrainer, TrainingHistory, train_sae
from interp.sae.steering import (
    SAESteering,
    SteeringEffect,
    SweetSpotResult,
    compute_persona_vector,
    decompose_vector_into_sae_features,
)
from interp.explainer import (
    DecisionExplainer,
    DecisionExplanation,
    PredictedOutput,
    InputImportance,
    SAEFeatureActivation,
    DeathAnalysis,
    analyze_death_log,
)

# Phase 2: Causal Analysis
from interp.patching import (
    ActivationPatcher,
    PatchingResult,
    CausalTrace,
    create_corrupted_input,
)
from interp.attention import (
    AttentionExtractor,
    AttentionPattern,
    AttentionSummary,
    CrossPlayerAttention,
    visualize_attention_pattern,
)
from interp.logit_lens import (
    LogitLens,
    LayerPrediction,
    PredictionEvolution,
    visualize_evolution,
)

# Phase 3: Probes & Labeling
from interp.probes import (
    ProbeConfig,
    MELEE_PROBES,
    LinearProbe,
    MLPProbe,
    PooledProbe,
    ProbeResults,
    ProbeTrainer,
    compute_class_weights,
    train_probe_on_cached_activations,
)
# Note: labeling submodule imported separately to avoid circular imports
# Use: from interp.labeling import FrameSampler, LabelStorage, TerminalLabeler

# Phase 4: Disentanglement
from interp.disentangle import (
    EntityDisentangler,
    PlayerAttentionRatio,
    SwapEffect,
    PlayerSensitivity,
    OpponentModelingFeatures,
    DisentanglementSummary,
    analyze_opponent_modeling,
)

# Phase 5: Circuit Analysis
from interp.circuits import (
    CircuitSpec,
    CircuitAnalysis,
    LayerImportance,
    FeatureImportance,
    CircuitAnalyzer,
    MELEE_CIRCUITS,
    get_melee_circuits,
    create_custom_circuit,
)

# Feature Importance Analysis
from interp.feature_importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceResult,
    categorize_feature,
)

__all__ = [
    # Config
    "InterpConfig",
    "SAEConfig",
    "SamplingConfig",
    # Hooks
    "HookPoint",
    "HookPointType",
    "HookManager",
    "all_block_outputs",
    "residual_stream_hooks",
    "mlp_activation_hooks",
    # Cache
    "ActivationCache",
    "CachedActivations",
    # SAE
    "TopKSparseAutoencoder",
    "SAEOutput",
    "SAETrainer",
    "TrainingHistory",
    "train_sae",
    # SAE Steering
    "SAESteering",
    "SteeringEffect",
    "SweetSpotResult",
    "compute_persona_vector",
    "decompose_vector_into_sae_features",
    # Explainer (Phase 1)
    "DecisionExplainer",
    "DecisionExplanation",
    "PredictedOutput",
    "InputImportance",
    "SAEFeatureActivation",
    "DeathAnalysis",
    "analyze_death_log",
    # Patching (Phase 2)
    "ActivationPatcher",
    "PatchingResult",
    "CausalTrace",
    "create_corrupted_input",
    # Attention (Phase 2)
    "AttentionExtractor",
    "AttentionPattern",
    "AttentionSummary",
    "CrossPlayerAttention",
    "visualize_attention_pattern",
    # Logit Lens (Phase 2)
    "LogitLens",
    "LayerPrediction",
    "PredictionEvolution",
    "visualize_evolution",
    # Probes (Phase 3)
    "ProbeConfig",
    "MELEE_PROBES",
    "LinearProbe",
    "MLPProbe",
    "PooledProbe",
    "ProbeResults",
    "ProbeTrainer",
    "compute_class_weights",
    "train_probe_on_cached_activations",
    # Disentanglement (Phase 4)
    "EntityDisentangler",
    "PlayerAttentionRatio",
    "SwapEffect",
    "PlayerSensitivity",
    "OpponentModelingFeatures",
    "DisentanglementSummary",
    "analyze_opponent_modeling",
    # Circuits (Phase 5)
    "CircuitSpec",
    "CircuitAnalysis",
    "LayerImportance",
    "FeatureImportance",
    "CircuitAnalyzer",
    "MELEE_CIRCUITS",
    "get_melee_circuits",
    "create_custom_circuit",
    # Feature Importance
    "FeatureImportanceAnalyzer",
    "FeatureImportanceResult",
    "categorize_feature",
]
