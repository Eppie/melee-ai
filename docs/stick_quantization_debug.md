# Diagnosing the "Up-Right Only" Stick Outputs

## What We Saw
- During inference the model always picked steep up-right directions for both sticks, even in neutral scenarios.
- Validation runs showed the main-stick head assigning almost all probability mass to palette indices `24`, `43`, `60`, etc. (all up-right angles).
- Raw labels in the dataset were mostly neutral or horizontal, so the behaviour was clearly inconsistent with the data.

## Investigation Timeline

1. **Check inference decoding**
   - Confirmed `model_interface.GPTInferenceEngine._decode_stick` simply performs `argmax` over logits and maps through the discrete palette.
   - No stochastic sampling or weird post-processing: if logits are skewed, it must come from the model or its targets.

2. **Inspect model biases**
   - Read `main_stick_head.net.2.bias` from the latest checkpoint.
   - Found large positive biases on the up-right palette entries and large negative biases on others.
   - Suggested that training data (or loss weighting) strongly encouraged those angles.

3. **Compare predictions vs. ground truth on validation windows**
   - Ran the model on a few validation windows and tallied the argmax palette indices → overwhelmingly up-right.
   - Quantized the same frames’ targets via `quantize_targets(..., input_domain="unit01")` → distribution was neutral/down/left-heavy, as expected.
   - Using `input_domain="unit11"` on the exact same targets reproduced the model’s diagonal-heavy distribution.
   - Conclusion: the labels fed to training must have been mis-quantized with the wrong domain.

4. **Validate dataset statistics**
   - Cross-referenced `validation_statistics.json` → neutral/horizontal entries dominate; no latent diagonal bias in the raw data.

5. **Trace git history for the regression**
   - `git log -S 'input_domain="unit11"' -- train.py validation.py`.
   - Found commit `2cddf1a48aab34a7dd5f11828067b02e62f88b07` (“lots more”, Oct 14 2025) which explicitly switched the training/validation calls to `quantize_targets(..., input_domain="unit11")`.
   - Earlier revisions left `input_domain` at its default (`auto`) or explicitly used the `[0,1]` path; the stick values have always been stored in `[0,1]`.

6. **Why the fix seemed ineffective**
   - After retraining with `input_domain="unit01"`, offline metrics looked healthy, but the live bot still *never* went left.
   - Instrumenting `GPTInferenceEngine.predict_from_raw` revealed that `_override_controller_features` was **overwriting the current game-state stick values with the previous model output**.  
     - The model mostly learns an “identity” prior on stick inputs, so this feedback loop meant it perpetually believed it was already holding right and never saw a left-facing input.
   - Patch: only fall back to the cached value when the live reading is missing/NaN (see `model_interface.py`).
   - After the change, the same checkpoint produced a balanced left/right distribution both offline and in-game.

## Root Cause
Targets were quantized under the assumption that they were already in `[-1, 1]`. In reality the dataset stores sticks in `[0, 1]`. Mapping `0.5` (neutral) directly into the palette with `input_domain="unit11"` makes it look like `+0.5`, which is closest to the up-right entries. Because training labels were wrong, the model learned exactly that behaviour.

## Fix
- Call `quantize_targets` with `input_domain="unit01"` (or let it auto-detect) everywhere the labels are prepared: `train.py`, `validation.py`, and any downstream scripts.
- Re-train (or at least regenerate labels/metrics) so the model learns against the corrected targets.
- Ensure inference keeps the live stick readings intact—only use the cached values as a fallback.

## Techniques That Helped
- **Direct distribution comparisons**: Inspecting `Counter` summaries of palette indices for both predictions and properly quantized targets quickly highlighted the mismatch.
- **Unit-test-style scripts**: Small inline Python snippets (`python - <<'PY'`) were invaluable for probing logits, biases, and quantization behaviour without touching the training loop.
- **`git log -S` searches**: Tracking when `input_domain="unit11"` was introduced let us pinpoint the exact regression commit.

## What Didn’t Help Much
- Staring at large validation JSON dumps without focused aggregation: the raw percentages were correct, but without comparing them to predictions it was easy to miss the pattern.
- Tweaking inference decoding or controller application logic: those layers simply reflected whatever the head predicted, so they offered no leverage once verified.

## TL;DR
If the model is spamming up-right or refuses to change direction, first confirm the training labels were quantized in `[0,1]` mode, then verify that inference isn’t feeding the model its own previous outputs. Correct quantization + preserving real controller inputs restores sane stick behaviour.
