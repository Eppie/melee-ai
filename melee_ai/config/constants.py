"""
Constants and palettes used throughout the Melee AI pipeline.
"""

import numpy as np

# Maximum frames in a replay (8 minutes at 60 FPS + buffer)
MAX_FRAMES: int = (60 * 60 * 8) + 123  # Full 8 minute replay

# Fox main stick quantization palette (64 positions)
FOX_STICK_64: list[tuple[float, float]] = [
    # --- Essentials ---
    (0.0, -1.0),  # 01 Hard down – ASDI-down, CC, fast-fall
    (-1.0, 0.0),  # 02 Hard left – max drift, horiz DI
    (0.0, 1.0),  # 03 Hard up – survival vs horizontals, Firefox charge positioning
    (1.0, 0.0),  # 04 Hard right – max drift, horiz DI
    (0.0, 0.0),  # 05 Neutral – buffering, re-center

    # --- Shield-drop (engine accepts -0.6875..-0.6625; one precise value suffices) ---
    (0.0, -0.6750),  # 06 Shield-drop Y (canonical pick in the allowed band)

    # --- Wavedash / waveland / ledgedash (down-toward). Keep both shallow (distance) and ~45° (Fox ledgedash) ---
    (0.9500, -0.2875),  # 07 16.84° down-right – shallowest legal; max slide if grounded
    (0.9300, -0.3500),  # 08 ~21° down-right – shallow WD
    (0.8625, -0.5000),  # 09 30° down-right – standard long WD
    (0.8125, -0.5750),  # 10 ~35° down-right – distance/consistency trade
    (0.7625, -0.6375),  # 11 ~40° down-right – reliable ground contact
    (0.7000, -0.7000),  # 12 45° down-right – Fox ledgedash (~45° best GALINT)
    (0.6375, -0.7625),  # 13 50° down-right – safer vs slants
    (-0.9500, -0.2875),  # 14 16.84° down-left – mirror
    (-0.9300, -0.3500),  # 15 ~21° down-left – mirror
    (-0.8625, -0.5000),  # 16 30° down-left – mirror
    (-0.8125, -0.5750),  # 17 ~35° down-left – mirror
    (-0.7625, -0.6375),  # 18 ~40° down-left – mirror
    (-0.7000, -0.7000),  # 19 45° down-left – Fox ledgedash mirror
    (-0.6375, -0.7625),  # 20 50° down-left – mirror

    # --- Firefox (Up+B) right/up – dense angles for unfair recoveries ---
    (0.9000, 0.4125),  # 21 ~24.5° – shallow Firefox
    (0.8625, 0.5000),  # 22 30°
    (0.8125, 0.5750),  # 23 35°
    (0.7625, 0.6375),  # 24 40°
    (0.7000, 0.7000),  # 25 45° – baseline diagonal
    (0.6375, 0.7625),  # 26 50°
    (0.5750, 0.8125),  # 27 55°
    (0.5000, 0.8625),  # 28 60°
    (0.4125, 0.9000),  # 29 65°
    (0.3500, 0.9300),  # 30 70°
    (0.2875, 0.9500),  # 31 75° – steep threader

    # --- Firefox (Up+B) left/up – mirrors ---
    (-0.9000, 0.4125),  # 32 ~24.5° left
    (-0.8625, 0.5000),  # 33 30° left
    (-0.8125, 0.5750),  # 34 35° left
    (-0.7625, 0.6375),  # 35 40° left
    (-0.7000, 0.7000),  # 36 45° left
    (-0.6375, 0.7625),  # 37 50° left
    (-0.5750, 0.8125),  # 38 55° left
    (-0.5000, 0.8625),  # 39 60° left
    (-0.4125, 0.9000),  # 40 65° left
    (-0.3500, 0.9300),  # 41 70° left
    (-0.2875, 0.9500),  # 42 75° left

    # --- Ultra-steep Firefox angles (beyond rectangle-legal; superhuman recoveries) ---
    (0.1750, 0.9750),  # 43 ~80° right/up – high thread under/over guards
    (0.0875, 0.9875),  # 44 ~85° right/up – near-vertical sweetspot
    (-0.1750, 0.9750),  # 45 ~80° left/up – mirror
    (-0.0875, 0.9875),  # 46 ~85° left/up – mirror

    # --- Universal 45° diagonals (DI/ASDI, drift, recoveries) ---
    (0.7071, 0.7071),  # 47 45° up-right – survival DI corner (approx; <=1 magnitude)
    (-0.7071, 0.7071),  # 48 135° up-left – mirror
    (-0.7071, -0.7071),  # 49 225° down-left – combo DI down-away
    (0.7071, -0.7071),  # 50 315° down-right – combo DI down-away

    # --- DI ring (8-point, ~22.5° steps; grid-aligned to 0.925/0.375 for unit radius) ---
    (0.9250, 0.3750),  # 51 ~22.5° – shallow-KB survival DI
    (0.3750, 0.9250),  # 52 ~67.5°
    (-0.3750, 0.9250),  # 53 ~112.5°
    (-0.9250, 0.3750),  # 54 ~157.5°
    (-0.9250, -0.3750),  # 55 ~202.5°
    (-0.3750, -0.9250),  # 56 ~247.5°
    (0.3750, -0.9250),  # 57 ~292.5°
    (0.9250, -0.3750),  # 58 ~337.5°

    # --- Steep ASDI/slide-off helpers (mirroring rectangle C-stick diagonals) ---
    (0.5250, 0.8500),  # 59 Up-right steep ASDI / platform slide-off
    (-0.5250, 0.8500),  # 60 Up-left steep ASDI / platform slide-off

    # --- Exact thresholds of axis activation (tilts, walk, ambiguous DI, pivot control) ---
    (0.6625, 0.0),  # 61 X=+0.6625 – tilt/walk cutoff; avoids X-smash
    (-0.6625, 0.0),  # 62 X=-0.6625 – mirror
    (0.2875, 0.0),  # 63 X=+0.2875 – minimal axis activation; dash/turn micro-timing
    (-0.2875, 0.0),  # 64 X=-0.2875 – mirror
]

# C-stick quantization palette (9 positions)
C_STICK_XY_CLUSTER_CENTERS_V0_1: np.ndarray = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.0, 1.0],
    [-0.7, -0.7],
    [0.7, -0.7],
    [0.7, 0.7],
    [-0.7, 0.7],
])

# Additional stick quantization palette from schema.py
STICK_XY_CLUSTER_CENTERS_V2 = (
    np.array([
        # neutral
        [0.0, 0.0],
        # partial tilt
        [0.35, 0.0],
        [-0.35, 0.0],
        [0.0, 0.35],
        [0.0, -0.35],
        # tilt
        [0.675, 0.0],
        [-0.675, 0.0],
        [0.0, 0.675],
        [0.0, -0.675],
        # full press (dash / smash attack)
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        # 17º / perfect wave/ledgedash
        [0.95, -0.3],
        [-0.95, -0.3],
        # 17º
        [0.95, 0.3],
        [-0.95, 0.3],
        # 30º / downward/up-angled f-smash
        [0.85, -0.5],
        [0.85, 0.5],
        [-0.85, -0.5],
        [-0.85, 0.5],
        # 45º + shield drops
        [0.7, -0.7],
        [-0.7, -0.7],
        [0.7, 0.7],
        [-0.7, 0.7],
        # up-/down-angled f-tilts
        [0.5, 0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
        # 60º
        [0.5, 0.85],
        [-0.5, 0.85],
        [0.5, -0.85],
        [-0.5, -0.85],
        # 72.5º
        [0.3, -0.95],
        [0.3, 0.95],
        [-0.3, -0.95],
        [-0.3, 0.95],
    ]) / 2 + 0.5
)

# Quick sanity checks
assert len(FOX_STICK_64) == 64
assert all(x * x + y * y <= 1.0 + 1e-9 for x, y in FOX_STICK_64), "All Fox stick positions must be inside unit circle"

assert C_STICK_XY_CLUSTER_CENTERS_V0_1.shape == (9, 2), "C-stick palette must have shape (9, 2)"
