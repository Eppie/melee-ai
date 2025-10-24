# Weights & Biases Metrics Reference

This report covers every metric emitted by `train.py` through the helpers in `train/wandb_utils.py`. For each metric you will find what it measures, why the signal is useful, and how it should behave over the course of training the controller model.

## Progress & Scheduling

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `epoch` | Outer training epoch (1-indexed) so you can line runs up with checkpoint cadence. | Monotonic increments up to `config.train.epochs`; stalls or skipped numbers hint at resume logic bugs. |
| `iter` | Batch index inside the current epoch (includes any resume offset). | Should sweep from 0 toward `len(loader)` each epoch; resets every epoch are expected. |
| `global_step` | Global batch counter shared with LR schedule and checkpoint metadata. | Must increase by 1 per optimizer step; plateaus mean the loop stopped, regressions indicate accidental rewinds. |
| `lr` | Current learning rate after cosine warmup/decay. | Starts near `config.train.lr` (≈1.3e-4) then follows the cosine schedule toward ~0; spikes or negatives mean scheduler misuse. |

## Loss Terms

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `loss/total` | Sum of all active loss components (stick + buttons + shoulder + optional value). | Should trend downward from ~3–5 in early training toward <1; sudden jumps imply dataloader or optimizer instability. |
| `loss/main` | Cross-entropy over the quantised main stick bins (64-way). | Expect a steady decrease toward ~0.6–0.8; sustained values >1.5 after warmup indicate the model is missing core stick behaviour. |
| `loss/c` | Cross-entropy for the C-stick logits (≈9 classes). | Begins near log(9)≈2.2 and should settle below ~0.5; plateaus >1 show the model is not learning smash DI targets. |
| `loss/buttons` | Weighted BCE for the five button channels. | Should slide toward 0.05–0.15; rising values usually mean the sigmoid head is saturating or class balancing broke. |
| `loss/shoulder` | Cross-entropy over analog shoulder states if present. | Target <0.4 once stable; if it hovers near 1 the shoulder head likely needs more capacity or weighting. |
| `loss/value` | Discounted-return MSE (scaled by `config.rl.value_loss_coef`) when the value head is enabled. | Falls toward the reward variance scale (typically <0.5); if it sticks near zero when the head is enabled, gradients may be zeroed. |

## Main Stick Metrics

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `metrics/acc_main_batch` | Overall main-stick top-1 accuracy versus quantised targets. | Should climb into the 0.7–0.9 band; drops below the repeat baseline mean regressions. |
| `metrics/acc_main_change` | Accuracy on frames where the main stick moves (harder cases). | Expect lower than hold accuracy (0.4–0.6); improvements here signal better reaction modelling. |
| `metrics/acc_main_hold` | Accuracy on frames where the main stick is held. | Typically >0.9; if it lags the repeat baseline the model is forgetting to persist commands. |
| `metrics/acc_main_rep` | Accuracy of the "repeat last frame" baseline for main stick. | Dataset-dependent (often 0.75–0.85); serves as a floor you must beat—large shifts mean the dataset distribution changed. |

## C-Stick Metrics

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `metrics/acc_c_batch` | Overall C-stick accuracy. | Aim for >0.65; much lower implies the head is underfitting high-variance smash inputs. |
| `metrics/acc_c_change` | C-stick accuracy on change frames. | Usually 0.35–0.55; flat lines near zero hint at the model ignoring new smash inputs. |
| `metrics/acc_c_hold` | C-stick accuracy on hold frames. | Should exceed 0.85; big dips mean the model releases the stick too often. |
| `metrics/acc_c_rep` | Repeat-baseline C-stick accuracy. | Reference baseline (~0.7); if it shifts, re-check preprocessing. |

## Button Aggregate Metrics

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `metrics/buttons_em_batch` | Exact-match ratio across the five-button vector. | Should rise toward 0.35–0.55; values below the repeat baseline flag poor button coordination. |
| `metrics/buttons_em_change` | Exact-match on frames where any button toggles. | Expect 0.2–0.4; if it stagnates low, the model is missing tech inputs. |
| `metrics/buttons_em_hold` | Exact-match on hold frames. | Should exceed 0.8; lower numbers mean the model is spuriously tapping buttons. |
| `metrics/buttons_f1_micro_batch` | Micro-averaged F1 across all buttons (model). | Target 0.45–0.65; declining scores often follow class-imbalance issues. |
| `metrics/buttons_f1_micro_maj` | Micro F1 of the majority-class baseline (mostly "no press"). | Should remain near a constant (~0.2–0.3); if the model falls below this, predictions regressed badly. |
| `metrics/buttons_f1_micro_rep` | Micro F1 of the repeat-last-frame baseline. | Baseline typically ~0.35–0.45; use it to gauge real gains on fast sequences. |
| `metrics/buttons_em_rep` | Exact-match for the repeat baseline. | Usually ~0.3; leverage as a sanity check for `buttons_em_*` metrics. |

## Per-Button Metrics

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `buttons/A_acc` | Per-frame accuracy for the A button. | Should clear 0.9 on holds; large drops suggest the head is missing jab/tilt timing. |
| `buttons/A_f1` | F1 score for the A button. | Aim for 0.4–0.6; falling below 0.3 means fails on rare press frames. |
| `buttons/A_rate` | Observed activation rate for A. | Should match dataset frequency (~4–5%); drift hints at label leakage or sampling skew. |
| `buttons/B_acc` | Per-frame accuracy for B. | Same expectations as A; holds above 0.9 indicate stable charge behaviour. |
| `buttons/B_f1` | F1 score for B. | Target 0.35–0.55; low values mean poor special-move precision. |
| `buttons/B_rate` | Activation rate for B. | Should stay near 4%; spikes imply overly aggressive specials. |
| `buttons/X/Y_acc` | Accuracy for X/Y jump buttons. | Aim for >0.9; lower scores imply missed jump buffering. |
| `buttons/X/Y_f1` | F1 for X/Y. | Target 0.5–0.7; use to monitor short-hop timing quality. |
| `buttons/X/Y_rate` | Activation rate for X/Y. | Should hover around 8–9%; trends upward mean jump spam. |
| `buttons/Z_acc` | Accuracy for Z (grab). | Expect >0.95 because grabs are sparse; big dips mean false positives. |
| `buttons/Z_f1` | F1 for Z. | Target 0.25–0.4 given ~1% positives; any drop below 0.2 signals missed tech chases. |
| `buttons/Z_rate` | Activation rate for Z. | Should stay near 1%; increases imply runaway grab spam. |
| `buttons/L/R_acc` | Accuracy for L/R shield triggers. | Should exceed 0.85 due to longer holds. |
| `buttons/L/R_f1` | F1 for L/R. | Target 0.45–0.6; low values mean missed shield drops or wavedashes. |
| `buttons/L/R_rate` | Activation rate for L/R. | Should remain around 11–12%; large swings point to weighting bugs. |

## Value Head Metrics *(only when `config.model.use_value_head` is true)*

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `value/pred_mean` | Mean predicted discounted return. | Should align with `value/target_mean`; divergence shows bias. |
| `value/target_mean` | Mean of the discounted-return targets. | Should stay close to the game reward scale (~0–3); drastic shifts suggest reward preprocessing changes. |
| `value/mse` | Mean-squared error of value predictions. | Should decline below 0.5; spikes usually mean bootstrapping instability. |
| `value/mae` | Mean absolute error for the value head. | Tracks interpretable error; aim for <0.5. |
| `value/corr` | Pearson correlation between prediction and target. | Should rise toward 0.4–0.7; negative values indicate the head is anti-correlated and likely diverging. |

## Throughput & AMP

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `throughput/frames_per_s` | Effective training throughput (batch_size × sequence length ÷ wall-clock). | Target a stable plateau (e.g., 2k–5k frames/s on modern GPUs); sustained drops point to dataloader or GPU throttling. |
| `optimizer/loss_scale` | Automatic mixed-precision loss scale from `GradScaler`. | Starts high (e.g., 2¹⁶) and adapts; rapid collapse toward 1 means underflow issues, while constant growth suggests AMP is healthy. |

## Gradient Diagnostics

| Metric | Tracks / why it matters | Watch for |
| --- | --- | --- |
| `gradients/total_norm` | L2 norm of all gradients before clipping. | Should stay below the clip threshold (~5); spikes imply exploding gradients. |
| `gradients/mean_abs` | Mean absolute gradient magnitude. | Typically 1e-5–1e-3; growth toward 1 hints at instability. |
| `gradients/mean` | Signed average gradient value. | Should sit near 0; persistent bias indicates asymmetrical updates. |
| `gradients/std` | Standard deviation of gradient values. | Expect 1e-5–1e-3; surges correlate with noisy batches. |
| `gradients/max_abs` | Largest absolute gradient entry. | Staying <1 is ideal; anything >>1 precedes NaNs. |
| `gradients/zero_fraction` | Share of gradient elements exactly zero. | High (0.4–0.7) is normal with sparse activations; creeping toward 1 suggests dead neurons. |
| `gradients/num_elements` | Count of gradient elements inspected. | Should be constant (≈parameter count); drops indicate frozen modules. |
| `gradients/zero_count` | Number of zero gradients. | Mirrors `zero_fraction`; sudden jumps imply saturation. |
| `gradients/nan_count` | NaN gradients encountered. | Must remain 0; any positive value requires immediate investigation. |
| `gradients/inf_count` | Inf gradients encountered. | Should stay 0; positives suggest overflow. |
| `gradients/nonfinite_count` | Total NaN+Inf gradients. | Must be 0; non-zero values usually coincide with loss spikes. |
| `gradients/params_with_grad` | Parameter tensors that produced gradients. | Should equal the model parameter count; decreases mean some layers are disconnected. |
| `gradients/param_total_norm` | L2 norm of model parameters. | Slow drift upward is expected; sudden explosions align with weight blow-up. |
| `gradients/param_max_abs` | Maximum absolute parameter value. | Should remain within a few units; huge jumps (>10) imply weight explosion. |
| `gradients/grad_param_ratio_mean` | Mean |grad| / |param| ratio per tensor. | Usually 1e-3–1e-1; values >1 indicate overly aggressive updates. |
| `gradients/grad_param_ratio_max` | Maximum |grad| / |param| ratio. | Should stay <1; spikes above 5 are red flags for layer instability. |
| `gradients/grad_param_ratio_min` | Minimum |grad| / |param| ratio. | Near-zero values that persist point to layers receiving almost no gradient. |
| `gradients/grad_to_param_norm_ratio` | Global gradient norm versus parameter norm. | Should settle well below 1; sustained growth implies step sizes that will overshoot. |
| `gradients/total_norm_pre_clip` | Gradient norm before clipping (duplicate of `total_norm`). | Use to verify clipping behaviour; repeated values ≥5 mean the clip is constantly engaged. |
| `gradients/total_norm_post_clip` | Gradient norm after clipping. | Should be ≤`grad_clip` (5); higher numbers indicate clip misconfiguration. |
| `gradients/was_clipped` | Indicator (0/1) for whether clipping activated. | Mostly 0 with occasional 1s; a stream of 1s means the base LR is too high. |
| `gradients/clip_coef` | Scaling factor applied by clipping. | Equals 1 when unclipped; values <0.5 highlight severe clipping. |
| `gradients/nonfinite_fraction` | Fraction of gradients that were NaN/Inf. | Must stay at 0.0; any rise requires halting the run. |
