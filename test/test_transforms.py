import numpy as np
import torch

from transforms.numpy_transforms import (
    sticks01_to_unit11_np,
    unit11_to_sticks01_np,
    quantize_unit11_to_palette_np,
)
from transforms.python_transforms import (
    sticks01_to_unit11_py,
    unit11_to_sticks01_py,
    quantize_unit11_to_palette_py,
)
from transforms.pytorch_transforms import (
    sticks01_to_unit11_torch,
    unit11_to_sticks01_torch,
    quantize_unit11_to_palette_torch,
)


def test_sticks01_to_unit11_equivalence():
    # Python
    py_val = 0.25
    py_result = sticks01_to_unit11_py(py_val)

    # NumPy
    np_val = np.array([0.25, 0.5, 0.75])
    np_result = sticks01_to_unit11_np(np_val)

    # PyTorch
    torch_val = torch.tensor([0.25, 0.5, 0.75])
    torch_result = sticks01_to_unit11_torch(torch_val)

    assert py_result == -0.5
    np.testing.assert_allclose(np_result, np.array([-0.5, 0.0, 0.5]))
    torch.testing.assert_close(torch_result, torch.tensor([-0.5, 0.0, 0.5]))

    # Check equivalence between libraries for a single value
    assert sticks01_to_unit11_py(0.5) == sticks01_to_unit11_np(np.array(0.5))
    assert sticks01_to_unit11_py(0.5) == sticks01_to_unit11_torch(torch.tensor(0.5))


def test_unit11_to_sticks01_equivalence():
    # Python
    py_val = -0.5
    py_result = unit11_to_sticks01_py(py_val)

    # NumPy
    np_val = np.array([-0.5, 0.0, 0.5])
    np_result = unit11_to_sticks01_np(np_val)

    # PyTorch
    torch_val = torch.tensor([-0.5, 0.0, 0.5])
    torch_result = unit11_to_sticks01_torch(torch_val)

    assert py_result == 0.25
    np.testing.assert_allclose(np_result, np.array([0.25, 0.5, 0.75]))
    torch.testing.assert_close(torch_result, torch.tensor([0.25, 0.5, 0.75]))

    # Check equivalence between libraries for a single value
    assert unit11_to_sticks01_py(0.0) == unit11_to_sticks01_np(np.array(0.0))
    assert unit11_to_sticks01_py(0.0) == unit11_to_sticks01_torch(torch.tensor(0.0))


def test_quantize_unit11_to_palette_equivalence():
    palette = [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ]
    point = [0.1, -0.2]

    # Python
    py_quant, py_idx = quantize_unit11_to_palette_py(point, palette, return_index=True)

    # NumPy
    np_palette = np.array(palette, dtype=np.float32)
    np_point = np.array(point, dtype=np.float32)
    np_quant, np_idx = quantize_unit11_to_palette_np(
        np_point, np_palette, return_index=True
    )

    # PyTorch
    torch_palette = torch.tensor(palette, dtype=torch.float32)
    torch_point = torch.tensor(point, dtype=torch.float32)
    torch_quant, torch_idx = quantize_unit11_to_palette_torch(
        torch_point, torch_palette, return_index=True
    )

    expected_quant = [0.0, 0.0]
    expected_idx = 4

    assert py_idx == expected_idx
    assert py_quant == expected_quant

    assert np_idx == expected_idx
    np.testing.assert_allclose(np_quant, expected_quant, atol=1e-6)

    assert torch_idx == expected_idx
    torch.testing.assert_close(
        torch_quant, torch.tensor(expected_quant, dtype=torch.float32)
    )

    # Check equivalence
    assert py_idx == np_idx
    assert py_idx == torch_idx.item()
    np.testing.assert_allclose(np.array(py_quant), np_quant, atol=1e-6)
    torch.testing.assert_close(torch.tensor(py_quant), torch_quant)

    # Test without returning index
    py_quant_no_idx = quantize_unit11_to_palette_py(point, palette)
    np_quant_no_idx = quantize_unit11_to_palette_np(np_point, np_palette)
    torch_quant_no_idx = quantize_unit11_to_palette_torch(torch_point, torch_palette)

    assert py_quant_no_idx == expected_quant
    np.testing.assert_allclose(np_quant_no_idx, expected_quant, atol=1e-6)
    torch.testing.assert_close(
        torch_quant_no_idx, torch.tensor(expected_quant, dtype=torch.float32)
    )


from transforms.numpy_transforms import scale_np
from transforms.python_transforms import scale
from transforms.pytorch_transforms import scale_torch


def test_scale_equivalence():
    # Test with a single scalar
    val = 5.0
    factor = 2.0
    expected = 10.0

    py_result = scale(val, factor)
    np_result = scale_np(np.array(val), factor)
    torch_result = scale_torch(torch.tensor(val), factor)

    assert py_result == expected
    np.testing.assert_allclose(np_result, expected)
    torch.testing.assert_close(torch_result, torch.tensor(expected))

    assert py_result == np_result
    assert py_result == torch_result.item()

    # Test with arrays/tensors
    arr = [1.0, 2.0, 3.0]
    factor = 3.0
    expected_arr = [3.0, 6.0, 9.0]

    np_arr = np.array(arr)
    np_result_arr = scale_np(np_arr, factor)

    torch_arr = torch.tensor(arr)
    torch_result_arr = scale_torch(torch_arr, factor)

    np.testing.assert_allclose(np_result_arr, np.array(expected_arr))
    torch.testing.assert_close(torch_result_arr, torch.tensor(expected_arr))

    np.testing.assert_allclose(np_result_arr, torch_result_arr.numpy())


from transforms.numpy_transforms import bit01_to_sign11_np
from transforms.python_transforms import bit01_to_sign11
from transforms.pytorch_transforms import bit01_to_sign11_torch


def test_bit01_to_sign11_equivalence():
    # Test with 0.0
    val_0 = 0.0
    expected_0 = -1.0

    py_result_0 = bit01_to_sign11(val_0)
    np_result_0 = bit01_to_sign11_np(np.array(val_0))
    torch_result_0 = bit01_to_sign11_torch(torch.tensor(val_0))

    assert py_result_0 == expected_0
    np.testing.assert_allclose(np_result_0, expected_0)
    torch.testing.assert_close(torch_result_0, torch.tensor(expected_0))

    assert py_result_0 == np_result_0
    assert py_result_0 == torch_result_0.item()

    # Test with 1.0
    val_1 = 1.0
    expected_1 = 1.0

    py_result_1 = bit01_to_sign11(val_1)
    np_result_1 = bit01_to_sign11_np(np.array(val_1))
    torch_result_1 = bit01_to_sign11_torch(torch.tensor(val_1))

    assert py_result_1 == expected_1
    np.testing.assert_allclose(np_result_1, expected_1)
    torch.testing.assert_close(torch_result_1, torch.tensor(expected_1))

    assert py_result_1 == np_result_1
    assert py_result_1 == torch_result_1.item()

    # Test with an array/tensor
    arr = [0.0, 1.0, 0.0]
    expected_arr = [-1.0, 1.0, -1.0]

    np_arr = np.array(arr)
    np_result_arr = bit01_to_sign11_np(np_arr)

    torch_arr = torch.tensor(arr)
    torch_result_arr = bit01_to_sign11_torch(torch_arr)

    np.testing.assert_allclose(np_result_arr, np.array(expected_arr))
    torch.testing.assert_close(torch_result_arr, torch.tensor(expected_arr))

    np.testing.assert_allclose(np_result_arr, torch_result_arr.numpy())
