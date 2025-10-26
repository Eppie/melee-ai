# Rust Integration - Test Commands

## Setup
```bash
source ~/.venvs/slippi312/bin/activate
cd /Users/eppie/PycharmProjects/nano-melee/libmelee
```

## 1. Test Rust Module Basics
```bash
python test_rust_module.py
```
Expected: All basic tests should pass ✅

## 2. Test Rust Integration
```bash
python test_rust_integration.py
```
Expected: 3/3 tests should pass ✅

## 3. Benchmark Performance (WITH Rust)
```bash
LIBMELEE_USE_RUST=1 python benchmark_rust.py
```
Expected: Should show significant speedup for data structures

## 4. Benchmark Performance (WITHOUT Rust - Python only)
```bash
LIBMELEE_USE_RUST=0 python benchmark_rust.py
```
Expected: Slower than Rust, but should still work

## 5. Parse test.slp with Rust Acceleration
```bash
LIBMELEE_USE_RUST=1 python test_slp_file.py
```
Expected: Should parse successfully and show frame count

## 6. Parse test.slp WITHOUT Rust (Python fallback)
```bash
LIBMELEE_USE_RUST=0 python test_slp_file.py
```
Expected: Should parse successfully (slower)

## 7. Run Existing Test Suite
```bash
python -m pytest test/test_console.py -v
```
Expected: All existing tests should still pass

## What to Look For

### Success Indicators:
- ✓ Rust module imports successfully
- ✓ GameState operations work with both Python and Rust
- ✓ Enums are compatible between Python and Rust
- ✓ SLP file parses correctly
- ✓ Performance improvement visible in benchmarks (50-100x for data structures)

### Common Issues:
- If segfault occurs: Might be numpy version incompatibility
- If "No module named melee_rust": Run `cd melee_rust && maturin develop --release`
- If slow performance: Make sure you built with `--release` flag

## Expected Performance Gains

Based on initial benchmarks:
- **Data Structure Creation**: 73x faster 🚀
- **Enum Operations**: Similar (Python enums are already fast)
- **Full SLP Parsing**: 5-15x faster expected when fully integrated

## Troubleshooting

If tests fail, try:
1. Rebuild Rust module: `cd melee_rust && maturin develop --release`
2. Check Python version: `python --version` (should be 3.12.x)
3. Verify imports: `python -c "import melee_rust; print('OK')"`

