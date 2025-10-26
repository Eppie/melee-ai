# Rust Acceleration for libmelee

This directory contains a Rust implementation of the performance-critical SLP parsing code, which provides significant speedups over the pure Python implementation.

## Installation

### Development Installation

1. Ensure Rust is installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install maturin:
```bash
cargo install maturin
```

3. Build and install the Rust extension:
```bash
cd melee_rust
source ../venv/bin/activate  # If using a venv
maturin develop --release
```

The `--release` flag is important for performance - it enables optimizations.

### Using the Rust Module

The Rust module provides the same data structures as the Python version:

```python
import melee_rust

# Create gamestate structures
gamestate = melee_rust.GameState()
player = melee_rust.PlayerState()

# Use enums
character = melee_rust.Character.Fox
stage = melee_rust.Stage.FinalDestination

# Parse SLP events (low-level API)
event_bytes = b'...'  # Raw SLP event data
frame_ended = melee_rust.parse_events(event_bytes, gamestate)
```

## Architecture

The Rust implementation consists of:

- **`enums.rs`**: All game enums (Character, Action, Stage, etc.)
- **`gamestate.rs`**: Data structures for game state (GameState, PlayerState, etc.)
- **`parser.rs`**: Binary parsing logic for SLP events
- **`lib.rs`**: PyO3 bindings to expose Rust to Python

## Performance

Expected performance improvements:

- **5-10x** faster SLP parsing
- **3-5x** faster overall frame processing
- **Reduced memory allocations** in hot loops

## Current Status

✅ Core types implemented and exposed to Python
✅ Binary parser for all major event types
✅ PyO3 bindings working
✅ Can parse POST_FRAME, PRE_FRAME, ITEM_UPDATE events
⚠️ Not yet integrated into Console.step() (requires state management)

## Future Integration

To fully integrate into `console.py`, the Rust parser needs:

1. State management for `eventsize`, `_frame`, `_current_stage`, etc.
2. Handling of controller flushing and side effects
3. Support for `_use_manual_bookends` mode
4. Menu event handling

For now, the Rust module can be used in custom implementations or for replay parsing where you want maximum performance.

## Building for Distribution

To build wheels for distribution:

```bash
cd melee_rust
maturin build --release
```

This creates a wheel in `target/wheels/` that can be installed with pip.

## Troubleshooting

### Import Error

If you get `ImportError: No module named 'melee_rust'`, make sure you've run `maturin develop` in the `melee_rust` directory while your Python environment is activated.

### SSL Certificate Errors

If you get SSL errors when downloading crates, you may need to install maturin via cargo instead:
```bash
cargo install maturin
```

### Compilation Errors

Make sure you have:
- Rust 1.70 or newer
- A C compiler (gcc, clang, or MSVC)
- Python development headers

## Development Workflow

1. Make changes to Rust code in `melee_rust/src/`
2. Rebuild: `cd melee_rust && maturin develop --release`
3. Test changes: `python test_rust_module.py`
4. Run benchmarks: `python test/perf_test.py`

## Environment Variables

- `LIBMELEE_USE_RUST=0`: Disable Rust acceleration (use Python implementation)
- `LIBMELEE_USE_RUST=1`: Enable Rust acceleration (default if available)

