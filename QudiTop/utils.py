"""Utility functions.
Refer: https://github.com/forcekeng/QudiTop
"""
import math
import torch
import numpy as np
from torch import Tensor
from .global_var import DTYPE
from typing import List, Tuple


def _fill_fist_sep(string, sep, length, fill_char=' '):
    poi = string.find(sep)
    if length < poi:
        raise Exception(f"Original length is {poi}, can not fill it to length {length}.")
    return string[:poi] + fill_char * (length - poi) + string[poi:]


def _check_str(string, name):
    if not isinstance(string, str):
        raise TypeError(f"{name} requires str, but get {type(string)}!")


def ket(i, dim):
    """Get the numerical column vector.
    Args:
        i: Value of ket.
        dim: Dimension of qudits.
    """
    vec = torch.zeros((dim, 1), dtype=DTYPE)
    vec[i, 0] = 1
    return vec


def bra(i, dim):
    """Get the numerical row vector.
    Args:
        i: Value of ket.
        dim: Dimension of qudits.
    """
    vec = torch.zeros((1, dim), dtype=DTYPE)
    vec[0, i] = 1
    return vec


def check_unitary(u: Tensor, atol=1e-8):
    """Check if input matrix is unitary."""
    u = np.array(u)
    if u.ndim != 2:
        raise ValueError(f"Input matrix must be 2-D matrix, but got shape {u.shape}.")
    v = np.matmul(u, u.conj().T)
    flag = np.isclose(v.real, np.eye(len(u)), atol=atol).all()
    if flag and np.iscomplexobj(v):
        flag = flag and np.isclose(v.imag, np.zeros_like(u), atol=atol).all()
    return flag


def get_complex_tuple(mat, shape=None):
    """Convert the input matrix to a tuple which represents the complex matrix.
    Args:
        mat: Input matrix, which can be real or complex, and data type can be numpy.ndarray or torch.Tensor.
        shape: If not None, reshape the `mat` shape.
    """
    if isinstance(mat, (Tuple, List)):
        assert len(mat) == 2, "The input is real and imaginary part respectively."
        re, im = mat
        if isinstance(re, np.ndarray):
            re = torch.tensor(re, dtype=DTYPE)
            im = torch.tensor(im, dtype=DTYPE)
        elif isinstance(re, Tensor):
            re = re.clone().type(DTYPE)
            im = im.clone().type(DTYPE)
    elif isinstance(mat, np.ndarray):
        if np.iscomplexobj(mat):
            re = torch.tensor(mat.real, dtype=DTYPE)
            im = torch.tensor(mat.imag, dtype=DTYPE)
        else:
            re = torch.tensor(mat, dtype=DTYPE)
            im = torch.zeros_like(re, dtype=DTYPE)
    elif isinstance(mat, Tensor):
        if torch.is_complex(mat):
            re = mat.real.clone().type(DTYPE)
            im = mat.imag.clone().type(DTYPE)
        else:
            re = mat.clone().type(DTYPE)
            im = torch.zeros_like(re, dtype=DTYPE)
    else:
        raise TypeError(f"The type of input `mat` should be numpy.ndarray, torch.Tensor or Tuple[re, im], but got type {type(mat)}.")
    if shape:
        re = re.reshape(shape)
        im = im.reshape(shape)
    return re, im


def bprint(strings: list, align=":", title='', v_around='=', h_around='|', fill_char=' '):
    """Print the information in block shape.
    Refer: https://gitee.com/mindspore/mindquantum
    """
    if not isinstance(strings, list):
        raise TypeError(f"strings requires a list, but get {type(strings)}")
    for string in strings:
        _check_str(string, "string")
    _check_str(align, 'align')
    _check_str(title, 'title')
    _check_str(v_around, 'v_around')
    _check_str(h_around, 'h_around')
    _check_str(fill_char, 'fill_char')
    maxmim_len = strings[0].find(align)
    for sub_str in strings:
        m_poi = sub_str.find(align)
        if m_poi > maxmim_len:
            maxmim_len = m_poi
    strings = [_fill_fist_sep(i, align, maxmim_len, fill_char) for i in strings]
    n_around = 3
    title = v_around * n_around + title + v_around * n_around
    maxmim = max(len(i) for i in strings)
    if len(title) > maxmim:
        len_total = (len(title) - maxmim) // 2 + (len(title) - maxmim) % 2
        strings = [h_around + ' ' * len_total + i + ' ' * (len(title) - len(i) - len_total) + h_around for i in strings]
        title = h_around + title + h_around
    else:
        len_total = (maxmim - len(title)) // 2 + (maxmim - len(title)) % 2
        title = v_around + v_around * len_total + \
            title + v_around * len_total + v_around
        strings = [h_around + i + ' ' * (len(title) - 2 - len(i)) + h_around for i in strings]
    bot = v_around + v_around * (len(title) - 2) + v_around
    output = []
    output.append(title)
    output.extend(strings)
    output.append(bot)
    return output


def str_special(str_pr):
    """Represent the string in more concise way.
    Refer: https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim
    """
    special = {'': 1, 'π': np.pi, '√2': np.sqrt(2), '√3': np.sqrt(3), '√5': np.sqrt(5)}
    if isinstance(str_pr, (int, str)):
        return str(str_pr)
    elif str_pr % 1 == 0:
        return str(int(str_pr))
    div = -1 if str_pr < 0 else 1
    str_pr *= -1 if str_pr < 0 else 1
    for key, val in special.items():
        if isinstance(str_pr, str):
            break
        if np.isclose(str_pr / val % 1, 0):
            div *= int(str_pr / val)
            str_pr = key if div == 1 else f'-{key}' if div == - \
                1 else f'{div}{key}'
        elif np.isclose(val / str_pr % 1, 0):
            div *= int(val / str_pr)
            key = 1 if val == 1 else key
            str_pr = f'{key}/{div}' if div > 0 else f'-{key}/{-div}'
    if isinstance(str_pr, str):
        return str_pr
    return str(round(str_pr * div, 4))


def str_ket(dim: int, state: np.ndarray) -> str:
    """Get ket format of the qudit state.
    Refer: https://github.com/GhostArtyom/QuditVQE/tree/main/QuditSim
    """
    if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
        state = state.flatten()
    if state.ndim != 1:
        raise ValueError(f'State requires a 1-D ndarray, but get {state.shape}')
    nq = round(math.log(len(state), dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong state shape {state.shape} is not a power of {dim}')
    nq = int(nq)
    tol = 1e-8
    string = []
    for ind, val in enumerate(state):
        base = np.base_repr(ind, dim).zfill(nq)
        real = np.real(val)
        imag = np.imag(val)
        str_real = str_special(real)
        str_imag = str_special(imag)
        if np.abs(val) < tol:
            continue
        if np.abs(real) < tol:
            string.append(f'{str_imag}j¦{base}⟩')
            continue
        if np.abs(imag) < tol:
            string.append(f'{str_real}¦{base}⟩')
            continue
        if str_imag.startswith('-'):
            string.append(f'{str_real}{str_imag}j¦{base}⟩')
        else:
            string.append(f'{str_real}+{str_imag}j¦{base}⟩')
    return '\n'.join(string)


__all__ = ['ket', 'bra', 'check_unitary', 'get_complex_tuple', 'bprint', 'str_special', 'str_ket']
