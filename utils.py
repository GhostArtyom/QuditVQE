import os
import numpy as np
from math import log
from functools import reduce
from typing import List, Union
from fractions import Fraction
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from scipy.io import loadmat, savemat
from numpy.linalg import det, eigh, norm, svd
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RX, RY, RZ, Rxx, Ryy, Rzz, U3, GlobalPhase, PhaseShift, UnivMathGate

DTYPE = np.float64
CDTYPE = np.complex128
opt_basis = ['zyz', 'u3']
A = np.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]])
M = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / np.sqrt(2)


def updatemat(name: str, save: dict):
    if os.path.exists(name):
        load = loadmat(name)
        load.update(save)
        savemat(name, load)
    else:
        savemat(name, save)


def dict_file(path: str) -> dict:
    dict_file = {}
    for root, _, files in os.walk(path):
        i = 1
        for name in sorted(files):
            subfolder = os.path.split(root)[-1]
            dict_file[f'{subfolder}_{i}'] = name
            i += 1
    return dict_file


def is_power_of_two(num: int) -> bool:
    if not isinstance(num, (int, np.int64)):
        num = round(num, 12)
        if num % 1 != 0:
            raise ValueError(f'Wrong number type {num} {type(num)}')
        num = int(num)
    return (num & (num - 1) == 0) and num != 0


def is_unitary(mat: np.ndarray) -> bool:
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        dim = mat.shape[0]
        return np.allclose(np.eye(dim), mat @ mat.conj().T)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def is_hermitian(mat: np.ndarray) -> bool:
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        return np.allclose(mat, mat.conj().T)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def approx_matrix(mat: np.ndarray, tol: float = 1e-15):
    if np.iscomplexobj(mat):
        mat_real = np.real(mat)
        mat_imag = np.imag(mat)
        mat_real[np.abs(mat_real) < tol] = 0
        mat_imag[np.abs(mat_imag) < tol] = 0
        mat = mat_real + 1j * mat_imag
        return mat_real if np.all(mat_imag == 0) else mat
    mat[np.abs(mat) < tol] = 0
    return mat


def random_qudit(dim: int, ndim: int = 1) -> np.ndarray:
    '''Generate random one-qudit state or matrix'''
    if ndim == 1:
        qudit = np.random.rand(dim) + 1j * np.random.rand(dim)
    elif ndim == 2:
        qudit = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    else:
        raise ValueError(f'Wrong qudit ndim {ndim}')
    qudit /= norm(qudit)
    return qudit


def random_qudits(dim: int, n_qudits: int, ndim: int = 1) -> np.ndarray:
    '''Generate random n-qudit states or matrices'''
    qudit_list = [random_qudit(dim, ndim) for _ in range(n_qudits)]
    qudits = reduce(np.kron, qudit_list)
    qudits /= norm(qudits)
    return qudits


def str_special(pr_str: Union[str, int, float]) -> str:
    special = {'': 1, 'π': np.pi, '√2': np.sqrt(2), '√3': np.sqrt(3), '√5': np.sqrt(5)}
    if isinstance(pr_str, (int, str)):
        return str(pr_str)
    elif pr_str % 1 == 0:
        return str(int(pr_str))
    coef = -1 if pr_str < 0 else 1
    pr_str *= -1 if pr_str < 0 else 1
    for k, v in special.items():
        frac = Fraction(pr_str / v).limit_denominator(100)
        multi = round(pr_str / v, 4)
        divisor = round(v / pr_str, 4)
        if np.isclose(multi % 1, 0):
            coef *= int(multi)
            pr_str = k if coef == 1 else f'-{k}' if coef == -1 else f'{coef}{k}'
            break
        elif np.isclose(divisor % 1, 0):
            coef *= int(divisor)
            k = 1 if v == 1 else k
            pr_str = f'{k}/{coef}' if coef > 0 else f'-{k}/{-coef}'
            break
        elif abs(pr_str / v - frac) < 1e-6:
            x, y = frac.numerator, frac.denominator
            x = '' if x == 1 else x
            pr_str = f'{x}{k}/{y}' if coef > 0 else f'-{x}{k}/{y}'
            break
    if isinstance(pr_str, str):
        return pr_str
    return str(round(pr_str * coef, 4))


def str_ket(state: np.ndarray, dim: int = 2) -> str:
    '''Get ket format of the qudit state'''
    if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
        state = state.flatten()
    if state.ndim != 1:
        raise ValueError(f'State requires a 1-D ndarray, but get {state.shape}')
    nq = round(log(len(state), dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong state shape {state.shape} is not a power of {dim}')
    nq = int(nq)
    tol = 1e-8
    string = []
    for ind, val in enumerate(state):
        base = np.base_repr(ind, dim).zfill(nq)
        real = np.real(val)
        imag = np.imag(val)
        real_str = str_special(real)
        imag_str = str_special(imag)
        if np.abs(val) < tol:
            continue
        if np.abs(real) < tol:
            string.append(f'{imag_str}j¦{base}⟩')
            continue
        if np.abs(imag) < tol:
            string.append(f'{real_str}¦{base}⟩')
            continue
        if imag_str.startswith('-'):
            string.append(f'{real_str}{imag_str}j¦{base}⟩')
        else:
            string.append(f'{real_str}+{imag_str}j¦{base}⟩')
    print('\n'.join(string))
    print(state)


def decompose_zyz(mat: np.ndarray):
    phase = -np.angle(det(mat)) / 2
    matU = np.exp(1j * phase) * mat
    cos = np.sqrt(np.real(matU[0, 0] * matU[1, 1]))
    if abs(cos) > 1:
        cos = np.clip(cos, -1, 1)
    theta = 2 * np.arccos(cos)
    phi = np.angle(matU[1, 1]) + np.angle(matU[1, 0])
    lam = np.angle(matU[1, 1]) - np.angle(matU[1, 0])
    return phase, theta, phi, lam


def decompose_u3(mat: np.ndarray):
    phase, theta, phi, lam = decompose_zyz(mat)
    phase += (phi + lam) / 2
    return phase, theta, phi, lam


def one_qubit_decompose(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = True, with_params: bool = True):
    name_phase = gate.name + '_phase'
    name_theta = gate.name + '_theta'
    name_phi = gate.name + '_phi'
    name_lam = gate.name + '_lam'
    mat = gate.matrix()
    if mat.shape != (2, 2):
        raise ValueError('Gate is not one-qubit')
    if not np.allclose(np.eye(2), mat @ mat.conj().T):
        raise ValueError('Gate is not unitary')
    circ = Circuit()
    obj = gate.obj_qubits
    if basis == 'zyz':
        phase, theta, phi, lam = decompose_zyz(mat)
        circ += RZ(name_lam).on(obj)
        circ += RY(name_theta).on(obj)
        circ += RZ(name_phi).on(obj)
    elif basis == 'u3':
        circ += U3(name_theta, name_phi, name_lam).on(obj)
        phase, theta, phi, lam = decompose_u3(mat)
    else:
        raise ValueError(f'Wrong basis {basis} is not in {opt_basis}')
    if with_phase:
        circ += GlobalPhase(name_phase).on(obj)
        pr = {name_phase: phase, name_phi: phi, name_theta: theta, name_lam: lam}
    else:
        pr = {name_phi: phi, name_theta: theta, name_lam: lam}
    return (circ, pr) if with_params else circ.apply_value(pr)


def simult_svd(mat1: np.ndarray, mat2: np.ndarray, is_complex: bool = True):
    d = mat1.shape[0]
    u_a, d_a, v_a_h = svd(mat1)
    u_a_h = u_a.conj().T
    v_a = v_a_h.conj().T
    if np.count_nonzero(d_a) != d:
        raise ValueError('Mat1 is not full-rank')
    # g commutes with d
    g = u_a_h @ mat2 @ v_a
    # because g is hermitian, eigen-decomposition is its spectral decomposition
    _, p = eigh(g)  # p is unitary or orthogonal
    u = u_a @ p
    v = v_a @ p
    # ensure det(u_a) == det(v_a) == +1
    if det(u) < 0:
        u[:, 0] *= -1
    if det(v) < 0:
        v[:, 0] *= -1
    d1 = u.conj().T @ mat1 @ v
    d2 = u.conj().T @ mat2 @ v
    d = d1 + 1j * d2
    return (u, v, d) if is_complex else (u, v, d1, d2)


def kron_factor_4x4_to_2x2s(mat: np.ndarray):
    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(mat[t]))
    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype=CDTYPE)
    f2 = np.zeros((2, 2), dtype=CDTYPE)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = mat[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = mat[a ^ i, b ^ j]
    # Rescale factors to have unit determinants.
    f1 /= np.sqrt(det(f1)) or 1
    f2 /= np.sqrt(det(f2)) or 1
    # Determine global phase.
    div = f1[a >> 1, b >> 1] * f2[a & 1, b & 1]
    if div == 0:
        raise ZeroDivisionError('Div cannot be 0')
    g = mat[a, b] / div
    if np.real(g) < 0:
        f1 *= -1
        g = -g
    return f1, f2


def two_qubit_decompose(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = True, with_params: bool = True):
    name_rxx = gate.name + '_Rxx'
    name_ryy = gate.name + '_Ryy'
    name_rzz = gate.name + '_Rzz'
    name_phase = gate.name + '_phase'
    mat = gate.matrix()
    if mat.shape != (4, 4):
        raise ValueError('Gate is not two-qubit')
    if not np.allclose(np.eye(4), mat @ mat.conj().T):
        raise ValueError('Gate is not unitary')
    ur = np.real(M.conj().T @ mat @ M)
    ui = np.imag(M.conj().T @ mat @ M)
    QL, QR, D = simult_svd(ur, ui)
    A1, A0 = kron_factor_4x4_to_2x2s(M @ QL @ M.conj().T)
    B1, B0 = kron_factor_4x4_to_2x2s(M @ QR.T @ M.conj().T)
    k = (A.T / 4) @ np.angle(np.diag(D))
    pr = {name_rxx: -2 * k[1], name_ryy: -2 * k[2], name_rzz: -2 * k[3], name_phase: k[0] / -2}
    circ = Circuit()
    circ_d = Circuit()
    obj0, obj1 = gate.obj_qubits
    circ += UnivMathGate(gate.name + '_B0', B0).on(obj0)
    circ += UnivMathGate(gate.name + '_B1', B1).on(obj1)
    circ += Rxx(name_rxx).on([obj0, obj1])
    circ += Ryy(name_ryy).on([obj0, obj1])
    circ += Rzz(name_rzz).on([obj0, obj1])
    if with_phase:
        circ += GlobalPhase(name_phase).on(obj0)
        circ += GlobalPhase(name_phase).on(obj1)
    circ += UnivMathGate(gate.name + '_A0', A0).on(obj0)
    circ += UnivMathGate(gate.name + '_A1', A1).on(obj1)
    for g in circ:
        if len(g.obj_qubits) == 1 and isinstance(g, UnivMathGate):
            if basis == 'zyz':
                gate_d, para = one_qubit_decompose(g, 'zyz', False)
            elif basis == 'u3':
                gate_d, para = one_qubit_decompose(g, 'u3', with_phase)
            else:
                raise ValueError(f'Wrong basis {basis} is not in {opt_basis}')
            circ_d += gate_d
            pr.update(para)
        else:
            circ_d += g
    return (circ_d, pr) if with_params else circ_d.apply_value(pr)


def two_level_unitary_synthesis(dim: int, basis: str, ind: List[int], pr_str: List[str], obj: List[int]) -> Circuit:
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    if len(ind) != 2:
        raise ValueError(f'The qudit unitary index length {len(ind)} should be 2.')
    if len(set(ind)) != len(ind):
        raise ValueError(f'The qudit unitary index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) >= dim:
        raise ValueError(f'The qudit unitary index {ind} should in 0 to {dim-1}.')
    if len(pr_str) != dim:
        raise ValueError(f'The qudit unitary params length {len(pr_str)} should be {dim}.')
    if ind == [0, 1]:
        corr = Circuit() + X(obj[0], obj[1]) + RY(-np.pi / 2).on(obj[1], obj[0]) + X(obj[1])
    elif ind == [0, 2]:
        corr = Circuit() + X(obj[1], obj[0]) + X(obj[1])
    elif ind == [1, 2]:
        corr = Circuit() + X(obj[0], obj[1]) + RY(np.pi / 2).on(obj[1], obj[0]) + X(obj[0])
    circ = Circuit() + corr
    if basis == 'zyz':
        circ += RZ(pr_str[0]).on(obj[0], obj[1])
        circ += RY(pr_str[1]).on(obj[0], obj[1])
        circ += RZ(pr_str[2]).on(obj[0], obj[1])
    elif basis == 'u3':
        theta, phi, lam = pr_str
        circ += U3(theta, phi, lam).on(obj[0], obj[1])
    else:
        raise ValueError(f'{basis} is not a supported decomposition method of {opt_basis}.')
    circ += corr.hermitian()
    return circ


def single_qudit_unitary_synthesis(dim: int, basis: str, name: str, obj: List[int]) -> Circuit:
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    circ = Circuit()
    index = [[0, 1], [0, 2], [1, 2]]
    if basis == 'zyz':
        for i, ind in enumerate(index):
            pr_ind = f'{"".join(str(i) for i in ind)}_{i}'
            pr_str = [f'{name}RZ{pr_ind}', f'{name}RY{pr_ind}', f'{name}Rz{pr_ind}']
            circ += two_level_unitary_synthesis(dim, basis, ind, pr_str, obj)
    elif basis == 'u3':
        for i, ind in enumerate(index):
            pr_ind = f'{"".join(str(i) for i in ind)}_{i}'
            pr_str = [f'{name}𝜃{pr_ind}', f'{name}𝜑{pr_ind}', f'{name}𝜆{pr_ind}']
            circ += two_level_unitary_synthesis(dim, basis, ind, pr_str, obj)
    else:
        raise ValueError(f'{basis} is not a supported decomposition method of {opt_basis}.')
    return circ


def controlled_rotation_synthesis(dim: int, ind: List[int], name: str, obj: int, ctrl: List[int], state: int) -> Circuit:
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    if state == 0:
        corr = Circuit() + X(ctrl[1]) + X(ctrl[2])
    elif state == 1:
        corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2])
    elif state == 2:
        corr = Circuit()
    if ind == [0, 1]:
        corr = corr + X(obj, ctrl) + RY(-np.pi / 2).on(ctrl[0], [obj] + ctrl[1:]) + X(ctrl[0], ctrl[1:])
    elif ind == [0, 2]:
        corr = corr + X(ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
    elif ind == [1, 2]:
        corr = corr + X(obj, ctrl) + RY(np.pi / 2).on(ctrl[0], [obj] + ctrl[1:]) + X(obj, ctrl[1:])
    circ = Circuit() + corr
    if 'RX' in name:
        circ += RX(name).on(obj, ctrl)
    elif 'RY' in name:
        circ += RY(name).on(obj, ctrl)
    elif 'RZ' in name:
        circ += RZ(name).on(obj, ctrl)
    elif 'GP' in name:
        circ += GlobalPhase(name).on(obj, ctrl)
    elif 'PS' in name:
        circ += PhaseShift(name).on(obj, ctrl)
    circ += corr.hermitian()
    return circ


def controlled_diagonal_synthesis(dim: int, name: str, obj: int, ctrl: List[int], state: int) -> Circuit:
    if dim != 3:
        raise ValueError('Only works when dim = 3')
    if state == 0:
        corr = Circuit() + X(ctrl[1]) + X(ctrl[2])
    elif state == 1:
        corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2])
    elif state == 2:
        corr = Circuit()
    ind01 = Circuit() + X(obj, ctrl) + RY(-np.pi / 2).on(ctrl[0], [obj] + ctrl[1:]) + X(ctrl[0], ctrl[1:])
    ind02 = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
    ind12 = Circuit() + X(obj, ctrl) + RY(np.pi / 2).on(ctrl[0], [obj] + ctrl[1:]) + X(obj, ctrl[1:])
    circ = Circuit() + corr
    circ = circ + ind01 + RZ(f'{name}RZ01').on(obj, ctrl) + ind01.hermitian()
    circ = circ + ind02 + RZ(f'{name}RZ02').on(obj, ctrl) + ind02.hermitian()
    circ += PhaseShift(name).on(ctrl[1], ctrl[2])
    circ += corr.hermitian()
    return circ


def qutrit_symmetric_ansatz(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = False) -> Circuit:
    dim = 3
    basis = basis.lower()
    if basis not in opt_basis:
        raise ValueError(f'{basis} is not a supported decomposition method of {opt_basis}')
    if gate.ctrl_qubits:
        raise ValueError(f'Currently not applicable for a controlled gate {gate}')
    circ = Circuit()
    obj = gate.obj_qubits
    name = f'{gate.name}_'
    if len(obj) == 2:
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}', obj)
    elif len(obj) == 4:
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U1_', obj[:2])
        circ += controlled_diagonal_synthesis(dim, f'{name}CD1_', obj[0], obj[1:], 1)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U2_', obj[:2])
        circ += controlled_diagonal_synthesis(dim, f'{name}CD2_', obj[0], obj[1:], 2)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U3_', obj[:2])
        circ += controlled_rotation_synthesis(dim, [1, 2], f'{name}RY1_1', obj[-1], obj[::-1][1:], 1)
        circ += controlled_rotation_synthesis(dim, [1, 2], f'{name}RY1_2', obj[-1], obj[::-1][1:], 2)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U4_', obj[:2])
        circ += controlled_diagonal_synthesis(dim, f'{name}CD3_', obj[0], obj[1:], 2)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U5_', obj[:2])
        circ += controlled_rotation_synthesis(dim, [0, 1], f'{name}RY2_1', obj[-1], obj[::-1][1:], 1)
        circ += controlled_rotation_synthesis(dim, [0, 1], f'{name}RY2_2', obj[-1], obj[::-1][1:], 2)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U6_', obj[:2])
        circ += controlled_diagonal_synthesis(dim, f'{name}CD4_', obj[0], obj[1:], 0)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U7_', obj[:2])
        circ += controlled_rotation_synthesis(dim, [1, 2], f'{name}RY3_1', obj[-1], obj[::-1][1:], 1)
        circ += controlled_rotation_synthesis(dim, [1, 2], f'{name}RY3_2', obj[-1], obj[::-1][1:], 2)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U8_', obj[:2])
        circ += controlled_diagonal_synthesis(dim, f'{name}CD5_', obj[0], obj[1:], 2)
        circ += single_qudit_unitary_synthesis(dim, basis, f'{name}U9_', obj[:2])
    else:
        raise ValueError('Currently only applicable when the n_qutrits is 1 or 2, which means the n_qubits must be 2 or 4.')
    if with_phase:
        for i in obj:
            circ += GlobalPhase(f'{name}phase').on(i)
    return circ


def circuit_depth(circ: Circuit) -> int:
    nq = circ.n_qubits
    depth = [0] * nq
    for i in range(nq):
        for g in circ:
            if i in g.obj_qubits or i in g.ctrl_qubits:
                depth[i] += 1
    return max(depth)


def partial_trace(rho: np.ndarray, dim: int, ind: int) -> np.ndarray:
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if rho.ndim != 1 and rho.ndim != 2:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if not isinstance(dim, (int, np.int64)):
        raise ValueError(f'Wrong dimension type {dim} {type(dim)}')
    if not isinstance(ind, (int, np.int64)):
        raise ValueError(f'Wrong index type {ind} {type(ind)}')
    n = rho.shape[0]
    m = n // dim
    if n == dim and rho.ndim == 1:
        return rho.conj() @ rho
    elif n == dim and rho.ndim == 2:
        return np.trace(rho)
    nq = round(log(m, dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong matrix size {n} is not a power of {dim}')
    nq = int(nq)
    if ind < 0 or ind > nq:
        raise ValueError(f'Wrong index {ind} is not in 0 to {nq}')
    pt = csr_matrix((m, m), dtype=CDTYPE)
    for k in range(dim):
        i_ = np.zeros(m, dtype=np.int64)
        for i in range(m):
            ii = np.base_repr(i, dim).zfill(nq)
            i_[i] = int(ii[:ind] + str(k) + ii[ind:], dim)
        psi = csr_matrix((np.ones(m), (np.arange(m), i_)), shape=(m, n))
        if rho.ndim == 1:
            temp = psi.dot(csr_matrix(rho).T)
            pt += temp.dot(temp.conj().T)
        elif rho.ndim == 2:
            pt += psi.dot(csr_matrix(rho)).dot(psi.conj().T)
    return pt.toarray()


def reduced_density_matrix(rho: np.ndarray, dim: int, position: List[int]) -> np.ndarray:
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if rho.ndim != 1 and rho.ndim != 2:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if not isinstance(dim, (int, np.int64)):
        raise ValueError(f'Wrong dimension type {dim} {type(dim)}')
    if isinstance(position, (int, np.int64)):
        position = [position]
    n = rho.shape[0]
    nq = round(log(n, dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong matrix size {n} is not a power of {dim}')
    nq = int(nq)
    p = [x for x in range(nq) if x not in position]
    for ind in p[::-1]:
        rho = partial_trace(rho, dim, ind)
    return rho


def fidelity(rho: np.ndarray, sigma: np.ndarray, sqrt: bool = False) -> float:
    state = {'rho': rho, 'sigma': sigma}
    for key, mat in state.items():
        if mat.ndim == 2 and (mat.shape[0] == 1 or mat.shape[1] == 1):
            mat = mat.flatten()
            state[key] = mat
        if mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
            raise ValueError(f'Wrong {key} shape {mat.shape}')
        if mat.ndim != 1 and mat.ndim != 2:
            raise ValueError(f'Wrong {key} shape {mat.shape}')
    rho, sigma = state.values()
    if rho.shape[0] != sigma.shape[0]:
        raise ValueError(f'Mismatch state shape: rho {rho.shape}, sigma {sigma.shape}')
    if rho.ndim == 1 and sigma.ndim == 1:
        f = np.abs(rho.conj() @ sigma)
        return f if sqrt else f**2
    elif rho.ndim == 1 and sigma.ndim == 2:
        f = np.real(rho.conj() @ sigma @ rho)
        return np.sqrt(f) if sqrt else f
    elif rho.ndim == 2 and sigma.ndim == 1:
        f = np.real(sigma.conj() @ rho @ sigma)
        return np.sqrt(f) if sqrt else f
    elif rho.ndim == 2 and sigma.ndim == 2:
        f = np.real(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))
        return f if sqrt else f**2
    else:
        raise ValueError(f'Wrong state ndim: rho {rho.ndim}, sigma {sigma.ndim}')


def symmetric_index(dim: int, n_qudits: int) -> dict:
    if not isinstance(dim, (int, np.int64)):
        raise ValueError(f'Wrong dimension type {dim} {type(dim)}')
    if not isinstance(n_qudits, (int, np.int64)):
        raise ValueError(f'Wrong n_qudits type {n_qudits} {type(n_qudits)}')
    if n_qudits == 1:
        ind = {}
        for i in range(2**(dim - 1)):
            num1 = bin(i).count('1')
            if num1 in ind:
                ind[num1].append(i)
            else:
                ind[num1] = [i]
    else:
        ind, ind_ = {}, {}
        for i in range(2**(dim - 1)):
            num1 = bin(i).count('1')
            i_ = bin(i)[2::].zfill(dim - 1)
            if num1 in ind_:
                ind_[num1].append(i_)
            else:
                ind_[num1] = [i_]
        for i in range(dim**n_qudits):
            multi = ['']
            base = np.base_repr(i, dim).zfill(n_qudits)
            for j in range(n_qudits):
                multi = [x + y for x in multi for y in ind_[int(base[j])]]
            ind[i] = [int(x, 2) for x in multi]
    return ind


def is_symmetric(mat: np.ndarray, n_qudits: int = 1) -> bool:
    if mat.ndim == 2 and (mat.shape[0] == 1 or mat.shape[1] == 1):
        mat = mat.flatten()
    if mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
        raise ValueError(f'Wrong matrix shape {mat.shape}')
    if mat.ndim != 1 and mat.ndim != 2:
        raise ValueError(f'Wrong matrix shape {mat.shape}')
    is_sym = True
    n = mat.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f'Wrong matrix size {n} is not a power of 2')
    nq = int(np.log2(n))
    dim = nq // n_qudits + 1
    if nq % n_qudits == 0 and nq != n_qudits:
        ind = symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape} or n_qudits {n_qudits}')
    if mat.ndim == 1:
        for i in range(dim**n_qudits):
            i_ = ind[i]
            if len(i_) != 1:
                a = mat[i_]
                is_sym = is_sym & np.allclose(a, a[0])
    elif mat.ndim == 2:
        for i in range(dim**n_qudits):
            i_ = ind[i]
            for j in range(dim**n_qudits):
                j_ = ind[j]
                if len(i_) != 1 or len(j_) != 1:
                    a = mat[np.ix_(i_, j_)]
                    is_sym = is_sym & np.allclose(a, a[0][0])
    return is_sym


def symmetric_decoding(qubit: np.ndarray, n_qudits: int = 1) -> np.ndarray:
    if qubit.ndim == 2 and (qubit.shape[0] == 1 or qubit.shape[1] == 1):
        qubit = qubit.flatten()
    if qubit.ndim == 2 and qubit.shape[0] != qubit.shape[1]:
        raise ValueError(f'Wrong qubit state shape {qubit.shape}')
    if qubit.ndim != 1 and qubit.ndim != 2:
        raise ValueError(f'Wrong qubit state shape {qubit.shape}')
    n = qubit.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f'Wrong qubit state size {n} is not a power of 2')
    nq = int(np.log2(n))
    dim = nq // n_qudits + 1
    if nq % n_qudits == 0 and nq != n_qudits:
        ind = symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong qubit state shape {qubit.shape} or n_qudits {n_qudits}')
    if qubit.ndim == 1:
        qudit = np.zeros(dim**n_qudits, dtype=CDTYPE)
        for i in range(dim**n_qudits):
            i_ = ind[i]
            qubit_i = qubit[i_]
            if np.allclose(qubit_i, qubit_i[0]):
                qudit[i] = qubit_i[0] * np.sqrt(len(i_))
            else:
                raise ValueError('Qubit state is not symmetric')
    elif qubit.ndim == 2:
        qudit = np.zeros([dim**n_qudits, dim**n_qudits], dtype=CDTYPE)
        for i in range(dim**n_qudits):
            i_ = ind[i]
            for j in range(dim**n_qudits):
                j_ = ind[j]
                qubit_ij = qubit[np.ix_(i_, j_)]
                if np.allclose(qubit_ij, qubit_ij[0][0]):
                    div = np.sqrt(len(i_)) * np.sqrt(len(j_))
                    qudit[i, j] = qubit_ij[0][0] * div
                else:
                    raise ValueError('Qubit state is not symmetric')
    return qudit


def symmetric_encoding(qudit: np.ndarray, n_qudits: int = 1, is_csr: bool = False) -> np.ndarray:
    if qudit.ndim == 2 and (qudit.shape[0] == 1 or qudit.shape[1] == 1):
        qudit = qudit.flatten()
    if qudit.ndim == 2 and qudit.shape[0] != qudit.shape[1]:
        raise ValueError(f'Wrong qudit state shape {qudit.shape}')
    if qudit.ndim != 1 and qudit.ndim != 2:
        raise ValueError(f'Wrong qudit state shape {qudit.shape}')
    dim = round(qudit.shape[0]**(1 / n_qudits), 12)
    if dim % 1 == 0:
        dim = int(dim)
        n = 2**((dim - 1) * n_qudits)
        ind = symmetric_index(dim, n_qudits)
    else:
        raise ValueError(f'Wrong qudit state shape {qudit.shape} or n_qudits {n_qudits}')
    if qudit.ndim == 1:
        qubit = csr_matrix((n, 1), dtype=CDTYPE)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            i_ = (ind_i, np.zeros(num_i))
            qubit += csr_matrix((data, i_), shape=(n, 1))
        if not is_csr:
            qubit = qubit.toarray().flatten()
    elif qudit.ndim == 2:
        qubit = csr_matrix((n, n), dtype=CDTYPE)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            for j in range(dim**n_qudits):
                ind_j = ind[j]
                num_j = len(ind_j)
                i_ = np.repeat(ind_i, num_j)
                j_ = np.tile(ind_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubit += csr_matrix((data, (i_, j_)), shape=(n, n))
        if not is_csr:
            qubit = qubit.toarray()
    return qubit
