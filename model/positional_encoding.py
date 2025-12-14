import torch
from torch import Tensor


def get_alibi_biases(
    num_heads: int, max_seq_len: int, device: torch.device = None
) -> Tensor:
    """
    Compute ALiBi (Attention with Linear Biases) position biases.

    ALiBi adds a static bias to attention scores based on the distance between tokens:
        bias[i, j] = -slope * |i - j|

    Each attention head gets a different slope, computed as a geometric sequence.
    This allows the model to extrapolate to longer sequences than seen during training.

    Parameters
    ----------
    num_heads : int
        Number of attention heads
    max_seq_len : int
        Maximum sequence length to precompute biases for
    device : torch.device, optional
        Device to create the tensor on. If None, uses CPU.

    Returns
    -------
    torch.Tensor
        Shape (1, num_heads, max_seq_len, max_seq_len) containing the biases.
        The bias at position [b, h, i, j] represents the bias for head h when
        attending from position i to position j.

    Notes
    -----
    - Slopes are computed as 2^(-(8*k/num_heads)) for k in [1, 2, ..., num_heads]
    - This creates a geometric progression from 2^(-8/num_heads) to 2^(-8)
    - The biases are negative and proportional to distance, encouraging local attention
    - Biases are 0 on the diagonal (i=j) and become more negative as distance increases

    Example
    -------
    For num_heads=4, max_seq_len=4:
        slopes = [2^(-2), 2^(-4), 2^(-6), 2^(-8)] ≈ [0.25, 0.0625, 0.0156, 0.0039]

        For head 0 (slope ≈ 0.25):
            [[0.00, -0.25, -0.50, -0.75],
             [0.00,  0.00, -0.25, -0.50],
             [0.00,  0.00,  0.00, -0.25],
             [0.00,  0.00,  0.00,  0.00]]
    """
    if device is None:
        device = torch.device("cpu")

    # Compute slopes for each head as a geometric sequence
    # Formula: 2^(-(8*k/num_heads)) for k in [1, 2, ..., num_heads]
    slopes = torch.pow(
        2.0,
        -8.0
        * torch.arange(1, num_heads + 1, dtype=torch.float32, device=device)
        / num_heads,
    )  # Shape: (num_heads,)

    # Create position indices
    positions = torch.arange(
        max_seq_len, dtype=torch.float32, device=device
    )  # Shape: (max_seq_len,)

    # Compute pairwise distances: |i - j|
    # positions[:, None] creates (max_seq_len, 1), positions[None, :] creates (1, max_seq_len)
    # Broadcasting gives us (max_seq_len, max_seq_len) matrix of distances
    distances = torch.abs(
        positions[:, None] - positions[None, :]
    )  # Shape: (max_seq_len, max_seq_len)

    # Apply slopes: bias[h, i, j] = -slope[h] * distance[i, j]
    # slopes[:, None, None] creates (num_heads, 1, 1)
    # distances[None, :, :] creates (1, max_seq_len, max_seq_len)
    # Broadcasting gives us (num_heads, max_seq_len, max_seq_len)
    biases = (
        -slopes[:, None, None] * distances[None, :, :]
    )  # Shape: (num_heads, max_seq_len, max_seq_len)

    # Add batch dimension: (1, num_heads, max_seq_len, max_seq_len)
    biases = biases.unsqueeze(0)

    return biases


def apply_rotary_emb(states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to a 4-D multi-head tensor.

    This implements a per-dimension 2D rotation on the last axis by splitting the
    head dimension into two halves and rotating each (x1_i, x2_i) pair using
    element-wise cos/sin for the corresponding time step and feature index:

        Let states have shape (B, T, H, 2d) and write:
            x1 = states[..., :d]   # first half
            x2 = states[..., d:]   # second half

        Then the rotation is
            y1 =  x1 * cos + x2 * sin
            y2 = -x1 * sin + x2 * cos

        and the result is `out = concat([y1, y2], dim=-1)` cast back to states.dtype.

    Parameters
    ----------
    states : torch.Tensor
        Shape (B, T, H, 2d). Typically the projected queries or keys from
        multi-head attention. The last dimension must be even (2d).
    cos : torch.Tensor
        Cosine factors with last dimension d. Must be broadcastable to
        x1/x2's shape (B, T, H, d). Common shapes:
          • (T, 1, d)  — shared across batch and heads
          • (1, T, 1, d)
          • (B, T, H, d)
        It is fine (and common) for `cos`/`sin` to be float32 while `states`
        are float16; the output is cast back to `states.dtype`.
    sin : torch.Tensor
        Sine factors, same shape/broadcasting rules as `cos`.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, T, H, 2d) with RoPE applied, dtype matching `states`.

    Shape rules & notes
    -------------------
    • `states.ndim` must be 4 and `states.shape[-1]` must be even.
    • `cos` and `sin` must have last dimension `d = states.shape[-1] // 2`.
    • Broadcasting usually sets `cos/sin` per-time-step and per-feature while
      sharing across batch and heads.

    Worked example (step-by-step, with concrete values)
    ---------------------------------------------------
    Suppose:
      B=1, T=2, H=1, 2d=4  →  d=2
      states =
        [[[[  1.,   2.,   3.,   4.]],      # t=0: (x1=[1,2],  x2=[3,4])
          [[ 10.,  20.,  30.,  40.]]]]     # t=1: (x1=[10,20], x2=[30,40])

      Choose cos/sin so that at t=0 we rotate by 90° (cos=0, sin=1) for each
      pair, and at t=1 we rotate by 0° (cos=1, sin=0). Using a broadcastable
      (T, 1, d) layout:

        cos =
          [[[0., 0.]],   # t=0
           [[1., 1.]]]   # t=1
        sin =
          [[[1., 1.]],   # t=0
           [[0., 0.]]]   # t=1

    1) Split the last dimension:
         x1 = states[..., :2] =
           [[[[ 1.,  2.]],
             [[10., 20.]]]]

         x2 = states[..., 2:] =
           [[[[ 3.,  4.]],
             [[30., 40.]]]]

    2) Rotate:
         y1 = x1 * cos + x2 * sin
            = at t=0: [1,2]*[0,0] + [3,4]*[1,1] = [3,4]
              at t=1: [10,20]*[1,1] + [30,40]*[0,0] = [10,20]

         y2 = -x1 * sin + x2 * cos
            = at t=0: -[1,2]*[1,1] + [3,4]*[0,0] = [-1,-2]
              at t=1: -[10,20]*[0,0] + [30,40]*[1,1] = [30,40]

    3) Concatenate halves:
         out = concat([y1, y2], dim=-1) =
           [[[[  3.,   4.,  -1.,  -2.]],
             [[ 10.,  20.,  30.,  40.]]]]

       Note how t=0 performed a 90° rotation ( (x1,x2) --> (x2, -x1) ), while
       t=1 (0°) left the vector unchanged.


    """
    assert states.ndim == 4  # multihead attention
    d = states.shape[3] // 2
    x1, x2 = states[..., :d], states[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(states.dtype)  # ensure input/output dtypes match
    return out
