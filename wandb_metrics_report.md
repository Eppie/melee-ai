# Comprehensive Wandb Metrics Report

This document details all metrics logged to wandb during `train.py` execution. Updated post-implementation of comprehensive metrics suite.

---

## 1. TRAINING METADATA & SCHEDULING

### `epoch`
- **Description**: Current training epoch number (1-indexed)
- **Range**: 1 to `config.train.epochs`
- **Related metrics**: `iter`, `global_step`
- **Interpretation**:
  - Tracks which pass through the dataset is currently running
  - Used to coordinate learning rate schedules and checkpoint saving
- **Utility**: Essential for tracking training progress

### `iter`
- **Description**: Completed batches within current epoch
- **Range**: 0 to number of batches per epoch
- **Related metrics**: `epoch`, `global_step`
- **Interpretation**:
  - Resets each epoch
  - Combined with epoch gives full training position
- **Utility**: Useful for debugging within-epoch issues

### `global_step`
- **Description**: Total optimizer steps across all epochs
- **Range**: 0 to total training steps
- **Related metrics**: `epoch`, `iter`
- **Interpretation**:
  - Never resets, monotonically increasing
  - Primary x-axis for wandb plots
- **Utility**: Critical for tracking overall training progress

### `lr`
- **Description**: Current learning rate
- **Range**: Typically 1e-5 to 1e-3 (depends on schedule)
- **Related metrics**: `global_step`, `schedule/*`
- **Interpretation**:
  - **Too high**: Loss may oscillate or diverge, gradients explode
  - **Too low**: Training too slow, may get stuck in local minima
  - Should follow expected schedule (warmup → plateau → decay)
- **Utility**: Essential for diagnosing optimization issues

### `schedule/label_smoothing`
- **Description**: Current label smoothing value (prevents overconfident predictions)
- **Range**: Starts at `config.train.label_smoothing`, decays to 0.5× that value
- **Related metrics**: `loss/*` (affects all CE losses)
- **Interpretation**:
  - Starts high during warmup, gradually decreases
  - Higher values = softer targets, lower overconfidence
  - **Too high**: Model may underfit, predictions too uncertain
  - **Too low**: Model may overfit to noise in data
- **Utility**: Useful for tuning regularization

---

## 2. LOSS METRICS

### `loss/total`
- **Description**: Total weighted loss across all output heads
- **Range**: Typically 0.5-3.0 at start, converges to 0.1-0.5
- **Related metrics**: `loss/main`, `loss/c`, `loss/buttons`, `loss/shoulder`, `loss/value`, `loss_fraction/*`
- **Interpretation**:
  - Sum of all component losses weighted by their coefficients
  - **Increasing**: Model diverging, check gradients/lr
  - **Plateauing early**: Model may need more capacity or lr adjustment
  - **Noisy**: Check `loss/total_std` and batch size
- **Utility**: PRIMARY training metric; absolutely essential

### `loss/total_std`
- **Description**: Standard deviation of total loss across last 100 batches
- **Range**: 0.0-2.0, typically 0.05-0.3
- **Related metrics**: `loss/total`, `loss/total_cv`
- **Interpretation**:
  - Measures batch-to-batch loss variance
  - **High (>0.5)**: Very noisy training, inconsistent batches
  - **Increasing over time**: Training becoming unstable
  - **Very low (<0.02)**: Batches very similar (may indicate limited data diversity)
- **Utility**: Essential for detecting training instability

### `loss/total_cv`
- **Description**: Coefficient of variation for loss (std/mean)
- **Range**: 0.0-2.0, typically 0.1-0.5
- **Related metrics**: `loss/total_std`, `loss/total`
- **Interpretation**:
  - Normalized measure of loss variability
  - **>0.5**: Very unstable training → consider gradient accumulation or larger batch size
  - **0.2-0.4**: Moderate variance (normal)
  - **<0.1**: Very stable training
- **Utility**: Better than std for comparing across different loss scales

### `loss/main`
- **Description**: Cross-entropy loss for main stick (64 discrete positions)
- **Range**: Typically 1.5-4.0 at start, converges to 0.3-1.0
- **Related metrics**: `metrics/acc_main_batch`, `logits/main_*`, `loss_fraction/main`
- **Interpretation**:
  - Higher than other stick losses because 64 classes vs 9 for c-stick
  - Random baseline ≈ ln(64) ≈ 4.16
  - **> 3.0**: Model not learning stick control well
  - **< 0.5**: Very good, approaching human-level precision
- **Utility**: Essential for diagnosing stick control

### `loss/c`
- **Description**: Cross-entropy loss for C-stick (9 discrete positions)
- **Range**: Typically 0.8-2.2 at start, converges to 0.1-0.5
- **Related metrics**: `metrics/acc_c_batch`, `logits/c_*`, `loss_fraction/c`
- **Interpretation**:
  - Lower than main stick loss (fewer classes)
  - Random baseline ≈ ln(9) ≈ 2.20
  - C-stick mostly neutral, so should converge faster than main
  - **> 1.5**: Model struggling with c-stick directional inputs
  - **< 0.3**: Excellent c-stick control
- **Utility**: Essential for diagnosing c-stick control

### `loss/buttons`
- **Description**: Binary cross-entropy loss for button outputs (5 buttons: A, B, X/Y, Z, L/R)
- **Range**: Typically 0.2-0.7 at start, converges to 0.05-0.2
- **Related metrics**: `metrics/buttons_em_batch`, `buttons/*_f1`, `loss_fraction/buttons`
- **Interpretation**:
  - Multi-label BCE summed across 5 buttons
  - Most frames have no buttons pressed (imbalanced)
  - **> 0.4**: Poor button prediction
  - **< 0.1**: Good button control
  - Compare to majority baseline (all zeros)
- **Utility**: Essential for diagnosing button prediction

### `loss/shoulder`
- **Description**: Cross-entropy loss for shoulder triggers (5 discrete levels)
- **Range**: Typically 0.5-1.6 at start, converges to 0.1-0.4
- **Related metrics**: Shoulder accuracy metrics, `loss_fraction/shoulder`
- **Interpretation**:
  - 5 classes: [0.0, 0.31, 0.42, 0.55, 1.0]
  - Heavily imbalanced toward 0.0 (not pressing)
  - Random baseline ≈ ln(5) ≈ 1.61
  - **> 1.0**: Model not learning trigger control
  - **< 0.2**: Good trigger precision
- **Utility**: Essential for shoulder trigger control

### `loss/value`
- **Description**: MSE loss for value head (predicts future reward/advantage)
- **Range**: Typically 0.01-0.5, weighted by `config.rl.value_loss_coef`
- **Related metrics**: `value/mse`, `value/pred_mean`, `value/target_mean`, `loss_fraction/value`
- **Interpretation**:
  - Trained on full unfiltered distribution (not value-weighted)
  - Used for value-based sample weighting in imitation
  - **High MSE + low correlation**: Value head not learning
- **Utility**: Important for value-weighted imitation

### `loss_fraction/{component}`
- **Description**: What percentage of total loss comes from each component (main, c, buttons, shoulder, value)
- **Range**: 0.0-1.0, sum = 1.0
- **Related metrics**: `loss/*` components
- **Interpretation**:
  - Shows which head contributes most to training signal
  - **One head >> others**: That head dominating learning
  - **Changing over time**: Learning dynamics shifting
  - Expected: main stick often largest due to 64 classes
- **Utility**: VERY USEFUL for understanding learning dynamics and head balance

---

## 3. ACCURACY METRICS (MAIN STICK)

### `metrics/acc_main_batch`
- **Description**: Overall main stick prediction accuracy across full batch
- **Range**: 0.0-1.0, typically 0.05-0.70
- **Related metrics**: `metrics/acc_main_change`, `metrics/acc_main_hold`, `accuracy/main_stick/top3`
- **Interpretation**:
  - % of frames where predicted stick position exactly matches target
  - Random baseline ≈ 1/64 ≈ 0.016
  - **< 0.2**: Poor, barely better than random
  - **0.3-0.5**: Moderate performance
  - **> 0.6**: Good stick control
  - Compare to change/hold split to diagnose bias
- **Utility**: PRIMARY accuracy metric for main stick

### `metrics/acc_main_change`
- **Description**: Accuracy on frames where stick position changed from previous frame
- **Range**: 0.0-1.0, typically lower than batch accuracy
- **Related metrics**: `metrics/acc_main_batch`, `schedule/change_weight_scale`, `consistency/main_stick/target_change_rate`
- **Interpretation**:
  - Harder than hold accuracy (predicting new movements)
  - If much lower than hold: model struggling with stick movements
  - Should improve as `change_weight_scale` decreases during training
  - **< 0.2**: Not learning movement patterns
  - **0.3-0.5**: Learning basic movements
  - **> 0.5**: Good movement prediction
- **Utility**: Critical for diagnosing movement vs holding

### `metrics/acc_main_hold`
- **Description**: Accuracy on frames where stick position unchanged from previous
- **Range**: 0.0-1.0, typically higher than change accuracy
- **Related metrics**: `metrics/acc_main_batch`, `metrics/acc_main_change`
- **Interpretation**:
  - Easier task (repeat last position)
  - If much higher than change: model biased to repeating
  - **Ideal**: Should be moderately higher than change (0.1-0.2 gap)
  - **> 0.8 with low change acc**: Repetition bias problem
- **Utility**: Useful for diagnosing repetition bias

---

## 4. ACCURACY METRICS (C-STICK & SHOULDER)

### `metrics/acc_c_batch`, `metrics/acc_c_change`, `metrics/acc_c_hold`
- Similar to main stick metrics but for C-stick (9 classes)
- Expected accuracy typically higher due to fewer classes and C-stick being mostly neutral

---

## 5. BUTTON METRICS

### `metrics/buttons_em_batch`, `metrics/buttons_em_change`, `metrics/buttons_em_hold`
- **Description**: Exact match accuracy (all 5 buttons correct simultaneously)
- Critical metric: `buttons_em_change` shows button timing quality

### `metrics/buttons_f1_micro_batch`, `metrics/buttons_f1_micro_maj`
- Micro-averaged F1 vs majority baseline

### Per-button metrics: `buttons/{button}_f1`, `buttons/{button}_precision`, `buttons/{button}_recall`, `buttons/{button}_rate`
- For each button: A, B, X/Y, Z, L/R
- F1 is primary per-button metric
- Rate provides data distribution context

---

## 6. VALUE HEAD METRICS

### `value/pred_mean`, `value/target_mean`, `value/mse`, `value/mae`, `value/corr`
- Standard regression metrics for value prediction
- `value_pred_bias` = `pred_mean - target_mean` (should be ≈0)

---

## 7. CONFIDENCE & ENTROPY METRICS

Per head: `main_stick`, `c_stick`, `shoulder`

### `confidence/{head}/avg_maxprob`
- **Description**: Average maximum probability across all predictions
- **Range**: 0.0-1.0, typically 0.3-0.9
- **Interpretation**: How confident is the model?
- **Should increase during training**

### `confidence/{head}/avg_maxprob_correct`
- **Description**: Average max probability only when prediction is correct
- **Range**: Should be higher than avg_maxprob
- **Interpretation**: Model should be more confident when correct (calibration)
- **Ideal gap**: 0.1-0.3 higher than avg_maxprob

### `entropy/{head}/mean`
- **Description**: Average Shannon entropy of predictions (in nats)
- **Range**: 0.0 to ln(num_classes)
- **Interpretation**: Prediction uncertainty
- **Should decrease during training**
- **Too low too quickly**: Possible mode collapse

**Calibration check:**
- **Low entropy + high accuracy**: Confident and correct ✓
- **Low entropy + low accuracy**: "Confidently wrong" (poor calibration!)

---

## 8. TOP-K ACCURACY

Per head: `main_stick`, `c_stick`, `shoulder`

### `accuracy/{head}/top3`, `accuracy/{head}/top5`
- **Description**: Fraction where target is in top-K predictions
- **Interpretation**:
  - **top5 >> top1**: Model has partial understanding but not precise
  - **top5 ≈ top1**: Model very confident or very wrong
- **Utility**: Shows if model is "close" even when not exactly right

---

## 9. FREQUENCY STATISTICS (MODE COLLAPSE DETECTION)

Per head for both predictions (`freq`) and targets (`tgt_freq`)

### `freq/{head}/top1_class`, `freq/{head}/top1_prop`
- **Description**: Most frequent class and its proportion
- **CRITICAL**: If `top1_prop → 1.0`, model has MODE COLLAPSED!
- **Compare pred vs target**: Should match; if not, model is biased

### `freq/{head}/diversity`
- **Description**: Gini-Simpson diversity index (1 - Σp²)
- **Range**: 0.0-1.0 (higher = more diverse)
- **CRITICAL**:
  - **→ 0**: Mode collapse!
  - **< 0.3**: Very low diversity
  - **> 0.7**: Good diversity
- **Compare pred vs target diversity**

**Example red flag:**

```
freq/main_stick/top1_prop = 0.85  # Bad: 85% neutral
tgt_freq/main_stick/top1_prop = 0.25  # Data only 25% neutral
freq/main_stick/diversity = 0.25  # Very low
# → Model defaulting to neutral (lazy learning!)
```

---

## 10. TEMPORAL CONSISTENCY

Per head: `main_stick`, `c_stick`, `shoulder`, `buttons`

### `consistency/{head}/pred_change_rate`, `consistency/{head}/target_change_rate`
- **Description**: How often predictions/targets change

### `consistency/{head}/change_rate_ratio`
- **Description**: pred_change_rate / target_change_rate
- **Interpretation**:
  - **>> 1 (e.g., 2.0)**: Model "jittery" (predictions flip-flopping)
  - **≈ 1.0**: Good temporal consistency ✓
  - **<< 1 (e.g., 0.5)**: Model too "sticky"
- **CRITICAL metric for prediction stability**

---

## 11. IMITATION WEIGHT STATISTICS

### `imitation/weight_mean`, `imitation/weight_std`, `imitation/weight_max/min`, `imitation/weight_p95/p05`
- Value-based sample weighting statistics

### `imitation/effective_batch_fraction`
- **Description**: Effective batch size / actual batch size
- **Range**: 0.0-1.0
- **CRITICAL**:
  - **→ 0 (e.g., 0.05)**: Only 5% of samples contributing (VERY aggressive filtering!)
  - **0.1-0.3**: Moderate filtering
  - **> 0.7**: Light filtering
  - **≈ 1.0**: No filtering
- **If too low, may be dropping too much training signal**

---

## 12. GRADIENT VARIANCE (STABILITY)

### `gradients/total_norm_variance`, `gradients/total_norm_std`, `gradients/total_norm_cv`
- **Description**: Variance/std/CV of gradient norms across last 100 batches
- **Interpretation**:
  - **CV > 0.7**: Unstable gradients → reduce LR or increase clipping
  - **CV 0.3-0.6**: Normal
  - **CV < 0.2**: Very stable ✓
- **VERY USEFUL for detecting gradient instability**

### `gradients/total_norm`, `gradients/total_norm_pre_clip`, `gradients/total_norm_post_clip`
- Standard gradient norm metrics
- Pre/post clip shows if clipping is active

### `gradients/nan_count`, `gradients/inf_count`
- **Should ALWAYS be 0**
- **If > 0**: CRITICAL BUG, stop training!

---

## 13. WEIGHT DRIFT

### `params/total_norm`
- Current parameter L2 norm

### `params/total_norm_velocity`
- **Description**: Change in parameter norm per step
- **Range**: Typically -0.01 to +0.01
- **Interpretation**:
  - **Positive**: Weights growing (normal early training)
  - **Negative**: Weights shrinking (weight decay or collapse)
  - **|velocity| >> 0.1**: Rapid changes (potential instability)
  - **≈ 0**: Parameters stabilized
- **VERY USEFUL for detecting instability early**

---

## 14. LOGIT & BIAS DIAGNOSTICS

### `logits/{head}_mean/std/max/min`
- Statistics of raw logits (pre-softmax/sigmoid)
- For CE heads: mean ≈ 0 is healthy
- std should be 1-3 (not too low or too high)

### `head_logits/{head}/max_abs`
- Maximum absolute logit value
- **> 15**: Potential instability

### `bias/*_out_mean/std/max`
- Output layer bias statistics
- `head_bias/{head}/max_abs` detects extreme biases

---

## 15. PERFORMANCE

### `throughput/frames_per_s`
- Training throughput
- **Decreasing over time**: Performance issue

### `optimizer/loss_scale`
- Dynamic loss scaling for FP16
- Stable at high value = good FP16 utilization

---

## CRITICAL MONITORING CHECKLIST

**Mode Collapse Detection:**

```python
freq/{head}/diversity < 0.5
freq/{head}/top1_prop > 0.8
```

**Calibration Check:**

```python
confidence/{head}/avg_maxprob_correct - confidence/{head}/avg_maxprob < 0.05
```

**Instability Detection:**

```python
gradients/total_norm_cv > 0.5
loss/total_cv > 0.4
params/total_norm_velocity > 0.1
gradients/nan_count > 0  # CRITICAL!
```

**Value Weighting:**

```python
imitation/effective_batch_fraction < 0.1  # Too aggressive
```

**Temporal Stability:**

```python
consistency/{head}/change_rate_ratio far from 1.0
```

---

## SUMMARY

**Total Metrics: ~200+**

**Most Critical:**

1. `loss/total` and `loss/total_std/cv`
2. `freq/{head}/diversity` and `freq/{head}/top1_prop` (mode collapse!)
3. `confidence/{head}/avg_maxprob_correct` (calibration)
4. `consistency/{head}/change_rate_ratio` (temporal stability)
5. `gradients/total_norm_cv` and `gradients/nan_count` (optimization health)
6. `imitation/effective_batch_fraction` (data usage)
7. `params/total_norm_velocity` (parameter stability)
8. `loss_fraction/{component}` (head balance)

**Primary Accuracy Metrics:**
- `metrics/acc_main_change` (harder than hold)
- `metrics/buttons_em_change` (button timing quality)
- `accuracy/{head}/top5` (partial credit)

*Last updated: Post-implementation of comprehensive metrics suite*
