"""Time evolution of quantum state.
Refer: https://gitee.com/forcekeng/quditop
"""
import math
import torch
from torch import Tensor
from typing import List, Tuple
from .utils import get_complex_tuple


def get_general_controlled_gate_cmatrix(u, dim: int, ctrl_states: List):
    """Get the matrix of general controlled gate.
    Args:
        u: Gate matrix.
        index_ctrl: The controlled state.
        k_qudits: The number of qudits the `u` acts on, including objective and control qudits.
        target_index: It determines where to put the input matrix `u`.
    """
    assert (isinstance(ctrl_states, List) and len(ctrl_states) >= 1), "The `ctrl_states` should be non-empty list."

    assert (max(ctrl_states) < dim), "The maximum element in `ctrl_states` should less than `dim`."

    u_re, u_im = get_complex_tuple(u)
    assert (len(u_re.shape) == 2 and u_re.shape[0] == u_re.shape[1]), "The input matrix `u` should be a square matrix."

    r = u_re.shape[0]
    k_obj_qudits = round(math.log(r, dim))
    k_ctrl_qudits = len(ctrl_states)
    k_qudits = k_obj_qudits + k_ctrl_qudits
    re = torch.eye(dim**k_qudits)
    im = torch.zeros((dim**k_qudits, dim**k_qudits))
    idx = dim**k_obj_qudits * int("".join(str(c) for c in ctrl_states), dim)
    re[idx:idx + r, idx:idx + r] = u_re
    im[idx:idx + r, idx:idx + r] = u_im
    return re, im


def evolution(op_mat: Tensor, qs: Tensor, target_indices: List[int]) -> Tensor:
    """Get the new quantum state after applying specific operation(gate or matrix).
    Refer: `https://pyquil-docs.rigetti.com/en/stable/_modules/pyquil/simulation/_numpy.html`
    Args:
        op_mat: The operation matrix that change the quantum state.
        qs: Current quantum state.
        target_indices: The qudits that `op_mat` acts on.
    Returns:
        The new quantum state.
    """
    k = len(target_indices)
    d = len(qs.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in target_indices)
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_indices):
        output_indices[t] = w
    return torch.einsum(op_mat, input_indices, qs, data_indices, output_indices)


def evolution_complex(op_mat: Tuple, qs: Tuple, target_indices: List[int]) -> Tensor:
    """Get the new quantum state after applying specific operation(gate or matrix).
    Since the auto-difference of complex number is not supported in PyTorch, Here just decompose the complex
    matrix as a tuple (real, imag) which represents the real part and imaginary part respectively.
    Args:
        op_mat: The operation matrix that change the quantum state.
        qs: Current quantum state.
        target_indices: The qudits that `op_mat` acts on.
    Returns:
        The new quantum state.
    """
    op_mat_real, op_mat_imag = op_mat
    qs_real, qs_imag = qs
    qs2_real = evolution(op_mat_real, qs_real, target_indices) \
        - evolution(op_mat_imag, qs_imag, target_indices)
    qs2_imag = evolution(op_mat_real, qs_imag, target_indices) \
        + evolution(op_mat_imag, qs_real, target_indices)
    return qs2_real, qs2_imag

__all__ = ["get_general_controlled_gate_cmatrix", "evolution", "evolution_complex"]