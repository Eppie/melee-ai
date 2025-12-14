# PPO Tests

Comprehensive test suite for the PPO training system.

## Running Tests

```bash
# Run all PPO tests
pytest ppo/tests/ -v

# Run specific test file
pytest ppo/tests/test_ppo_loss.py -v

# Run with coverage
pytest ppo/tests/ --cov=ppo --cov-report=html
```

## Test Categories

### 1. PPO Loss Tests (`test_ppo_loss.py`)
- Action log probability computation
- Entropy calculation
- Full PPO loss with all components
- Button type conversion (uint8 → bool)
- Clipping behavior

### 2. Rollout Tests (`test_rollout.py`)
- Rollout buffer initialization
- Frame appending
- GAE advantage computation
- Windowed batch creation
- Action extraction from batches
- Multi-rollout batching

### 3. Coordinator Shape Tests (`test_coordinator_shapes.py`)
- Action shape extraction (final timestep)
- Feature scaling/unscaling
- Memory estimation
- Tensor cleanup

## Common Issues Caught by Tests

1. **Shape Mismatches**: Tests verify all tensor shapes match between:
   - Features → Model input
   - Model output → Loss computation
   - Actions (full sequence vs final timestep)

2. **Type Errors**: Tests catch:
   - uint8 vs bool for buttons
   - Variable name shadowing (F vs torch.nn.functional.F)

3. **Missing Features**: Tests verify:
   - All loss components present

4. **Memory Issues**: Tests check:
   - Tensor deletion works
   - Memory estimates are reasonable

## Adding New Tests

When adding new PPO functionality:

1. Write tests FIRST (TDD)
2. Test shape transformations at boundaries
3. Test with small models for speed
4. Use `pytest.mark.slow` for expensive tests
5. Add docstrings explaining what's being tested

## CI Integration

These tests should run before:
- Commits to main branch
- PR merges
- Training launches

```bash
# Pre-commit hook
pytest ppo/tests/ -v --tb=short || exit 1
```
