import os
import numpy as np
from math import log
from typing import List
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from numpy.linalg import det, eigh, svd
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RX, RY, RZ, Rxx, Ryy, Rzz, U3, GlobalPhase, UnivMathGate

DTYPE = np.float64
CDTYPE = np.complex128
opt_basis = ['zyz', 'u3']
A = np.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]])
M = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / np.sqrt(2)


def dict_file(path):
    dict_file = {}
    for root, dirs, files in os.walk(path):
        i = 1
        for name in sorted(files):
            subfolder = os.path.split(root)[-1]
            dict_file[f'{subfolder}_{i}'] = name
            i += 1
    return dict_file

def is_power_of_two(num: int) -> bool:
    if not isinstance(num, int):
        num = round(num, 12)
        if num % 1 != 0:
            raise ValueError(f'Wrong number type {num} {type(num)}')
        num = int(num)
    return (num & (num - 1) == 0) and num != 0


def is_unitary(mat: np.ndarray) -> bool:
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        d = mat.shape[0]
        return np.allclose(np.eye(d), mat @ mat.conj().T)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def is_hermitian(mat: np.ndarray) -> bool:
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        return np.allclose(mat, mat.conj().T)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def str_special(str_pr):
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
            str_pr = key if div == 1 else f'-{key}' if div == -1 else f'{div}{key}'
        elif np.isclose(val / str_pr % 1, 0):
            div *= int(val / str_pr)
            key = 1 if val == 1 else key
            str_pr = f'{key}/{div}' if div > 0 else f'-{key}/{-div}'
    if isinstance(str_pr, str):
        return str_pr
    return str(round(str_pr * div, 4))


def str_ket(dim: int, state: np.ndarray) -> str:
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
    # return '\n'.join(string)
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


def one_qubit_decompose(gate: UnivMathGate,
                        basis: str = 'zyz',
                        with_phase: bool = True,
                        with_params: bool = True) -> Circuit:
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


def simult_svd(mat1: np.ndarray, mat2: np.ndarray):
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
    return u, v, d


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


def two_qubit_decompose(gate: UnivMathGate,
                        basis: str = 'zyz',
                        with_phase: bool = True,
                        with_params: bool = True) -> Circuit:
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


def Uind(basis, d, ind, pr, obj):
    if d != 3:
        raise ValueError('Only works when d = 3')
    if len(ind) != 2:
        raise ValueError(f'U3 index length {len(ind)} should be 2')
    if len(set(ind)) != len(ind):
        raise ValueError(f'U3 index {ind} cannot be repeated')
    if min(ind) < 0 or max(ind) >= d:
        raise ValueError(f'U3 index {ind} should in 0 to {d-1}')
    if len(pr) != 3:
        raise ValueError(f'U3 params length {len(pr)} should be 3')
    circ = Circuit()
    if ind == [0, 1]:
        corr = Circuit() + X(obj[1], obj[0]) + RY(np.pi / 2).on(obj[0], obj[1]) + X(obj[1], obj[0]) + X(obj[1])
    elif ind == [0, 2]:
        corr = Circuit() + X(obj[0]) + X(obj[1], obj[0]) + X(obj[0])
    elif ind == [1, 2]:
        corr = Circuit() + X(obj[1], obj[0]) + RY(-np.pi / 2).on(obj[0], obj[1]) + X(obj[1], obj[0])
    circ += corr
    if basis == 'zyz':
        circ += RZ(pr[0]).on(obj[0], obj[1])
        circ += RY(pr[1]).on(obj[0], obj[1])
        circ += RZ(pr[2]).on(obj[0], obj[1])
    elif basis == 'u3':
        theta, phi, lam = pr
        circ += U3(theta, phi, lam).on(obj[0], obj[1])
    else:
        raise ValueError(f'Wrong basis {basis} is not in {opt_basis}')
    circ += corr.hermitian()
    return circ


def Ub(basis, d, name, obj):
    circ = Circuit()
    index = [[0, 1], [0, 2], [1, 2]]
    if basis == 'zyz':
        for i, ind in enumerate(index):
            str_pr = f'{"".join(str(i) for i in ind)}_{i}'
            pr = [f'{name}RZ{str_pr}', f'{name}RY{str_pr}', f'{name}Rz{str_pr}']
            circ += Uind(basis, d, ind, pr, obj)
    elif basis == 'u3':
        for i, ind in enumerate(index):
            str_pr = f'{"".join(str(i) for i in ind)}_{i}'
            pr = [f'{name}𝜃{str_pr}', f'{name}𝜑{str_pr}', f'{name}𝜆{str_pr}']
            circ += Uind(basis, d, ind, pr, obj)
    else:
        raise ValueError(f'Wrong basis {basis} is not in {opt_basis}')
    return circ


def GCRb(d, ind, name, obj, ctrl, state):
    if d != 3:
        raise ValueError('Only works when d = 3')
    circ = Circuit()
    if state == 0:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(obj, ctrl[1:]) + X(ctrl[0], ctrl[1:] + [obj]) + X(
                obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(-np.pi / 2).on(
                obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj])
    elif state == 1:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(
                np.pi / 2).on(obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(obj, ctrl[1:]) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(
                -np.pi / 2).on(obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj])
    elif state == 2:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + RY(np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(obj, ctrl[1:]) + X(ctrl[0], ctrl[1:] + [obj]) + X(obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + RY(-np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj])
    circ += corr
    if 'RX' in name:
        circ = circ + RX(name).on(obj, ctrl)
    elif 'RY' in name:
        circ = circ + RY(name).on(obj, ctrl)
    elif 'RZ' in name:
        circ = circ + RZ(name).on(obj, ctrl)
    elif 'GP' in name:
        circ = circ + GlobalPhase(name).on(obj, ctrl)
    circ += corr.hermitian()
    return circ


def GCPb(d, name, obj, ctrl, state):
    if d != 3:
        raise ValueError('Only works when d = 3')
    circ = Circuit()
    if state == 0:
        corr = Circuit()
    elif state == 1:
        corr = Circuit()
    elif state == 2:
        corr = Circuit()
    circ += corr
    circ = circ + GlobalPhase(name).on(obj, ctrl)
    circ += corr.hermitian()
    return circ


def Cb(d, name, obj, ctrl, state):
    if d != 3:
        raise ValueError('Only works when d = 3')
    circ = Circuit()
    circ += GCRb(d, [0, 1], f'{name}RZ01', obj, ctrl, state)
    circ += GCRb(d, [0, 2], f'{name}RZ02', obj, ctrl, state)
    circ += GCRb(d, [0, 1], f'{name}GP', obj, ctrl, state)
    circ += GCRb(d, [0, 2], f'{name}GP', obj, ctrl, state)
    circ += GCRb(d, [1, 2], f'{name}GP', obj, ctrl, state)
    return circ


def qutrit_symmetric_ansatz(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = False):
    name = f'{gate.name}_'
    obj = gate.obj_qubits
    circ = Circuit()
    if len(obj) == 2:
        circ += Ub(basis, 3, f'{name}', obj)
    elif len(obj) == 4:
        circ += Ub(basis, 3, f'{name}U1_', obj[:2])
        circ += Cb(3, f'{name}C1_', obj[0], obj[1:], 1)
        circ += Ub(basis, 3, f'{name}U2_', obj[:2])
        circ += Cb(3, f'{name}C2_', obj[0], obj[1:], 2)
        circ += Ub(basis, 3, f'{name}U3_', obj[:2])
        circ += GCRb(3, [1, 2], f'{name}RY12', obj[-1], obj[::-1][1:], 2)
        circ += GCRb(3, [1, 2], f'{name}RY11', obj[-1], obj[::-1][1:], 1)
        circ += Ub(basis, 3, f'{name}U4_', obj[:2])
        circ += Cb(3, f'{name}C3_', obj[0], obj[1:], 2)
        circ += Ub(basis, 3, f'{name}U5_', obj[:2])
        circ += GCRb(3, [0, 1], f'{name}RY22', obj[-1], obj[::-1][1:], 2)
        circ += GCRb(3, [0, 1], f'{name}RY21', obj[-1], obj[::-1][1:], 1)
        circ += Ub(basis, 3, f'{name}U6_', obj[:2])
        circ += Cb(3, f'{name}C4_', obj[0], obj[1:], 0)
        circ += Ub(basis, 3, f'{name}U7_', obj[:2])
        circ += GCRb(3, [1, 2], f'{name}RY32', obj[-1], obj[::-1][1:], 2)
        circ += GCRb(3, [1, 2], f'{name}RY31', obj[-1], obj[::-1][1:], 1)
        circ += Ub(basis, 3, f'{name}U8_', obj[:2])
        circ += Cb(3, f'{name}C5_', obj[0], obj[1:], 2)
        circ += Ub(basis, 3, f'{name}U9_', obj[:2])
    else:
        raise ValueError('Only works when number of qutrits <= 2')
    if with_phase:
        for i in obj:
            circ += GlobalPhase(f'{name}phase').on(i)
    return circ


def partial_trace(rho: np.ndarray, d: int, ind: int) -> np.ndarray:
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if rho.ndim != 1 and rho.ndim != 2:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if not isinstance(d, int):
        raise ValueError(f'Wrong dimension type {d} {type(d)}')
    if not isinstance(ind, int):
        raise ValueError(f'Wrong index type {ind} {type(ind)}')
    n = rho.shape[0]
    m = n // d
    if n == d and rho.ndim == 1:
        return rho.conj() @ rho
    elif n == d and rho.ndim == 2:
        return np.trace(rho)
    nq = round(log(m, d), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong matrix size {n} is not a power of {d}')
    nq = int(nq)
    if ind < 0 or ind > nq:
        raise ValueError(f'Wrong index {ind} is not in 0 to {nq}')
    pt = csr_matrix((m, m), dtype=CDTYPE)
    for k in range(d):
        i_ = np.zeros(m, dtype=np.int64)
        for i in range(m):
            ii = np.base_repr(i, d).zfill(nq)
            i_[i] = int(ii[:ind] + str(k) + ii[ind:], d)
        psi = csr_matrix((np.ones(m), (np.arange(m), i_)), shape=(m, n))
        if rho.ndim == 1:
            temp = psi.dot(csr_matrix(rho).T)
            pt += temp.dot(temp.conj().T)
        elif rho.ndim == 2:
            pt += psi.dot(csr_matrix(rho)).dot(psi.conj().T)
    return pt.toarray()


def reduced_density_matrix(rho: np.ndarray, d: int, position: List[int]) -> np.ndarray:
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if rho.ndim == 2 and rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if rho.ndim != 1 and rho.ndim != 2:
        raise ValueError(f'Wrong state shape {rho.shape}')
    if not isinstance(d, int):
        raise ValueError(f'Wrong dimension type {d} {type(d)}')
    if isinstance(position, int):
        position = [position]
    n = rho.shape[0]
    nq = round(log(n, d), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong matrix size {n} is not a power of {d}')
    nq = int(nq)
    p = [x for x in range(nq) if x not in position]
    for ind in p[::-1]:
        rho = partial_trace(rho, d, ind)
    return rho


def fidelity(rho: np.ndarray, sigma: np.ndarray, sqrt: bool = False) -> float:
    state = {'rho': rho, 'sigma': sigma}
    for i, mat in state.items():
        if mat.ndim == 2 and (mat.shape[0] == 1 or mat.shape[1] == 1):
            mat = mat.flatten()
            state[i] = mat
        if mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
            raise ValueError(f'Wrong {i} shape {mat.shape}')
        if mat.ndim != 1 and mat.ndim != 2:
            raise ValueError(f'Wrong {i} shape {mat.shape}')
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


def sym_ind(d: int, m: int) -> dict:
    if not isinstance(d, int):
        raise ValueError(f'Wrong dimension type {d} {type(d)}')
    if not isinstance(m, int):
        raise ValueError(f'Wrong multi type {m} {type(m)}')
    if m == 1:
        ind = {}
        for i in range(2**(d - 1)):
            num1 = bin(i).count('1')
            if num1 in ind:
                ind[num1].append(i)
            else:
                ind[num1] = [i]
    else:
        ind, ind_ = {}, {}
        for i in range(2**(d - 1)):
            num1 = bin(i).count('1')
            i_ = bin(i)[2::].zfill(d - 1)
            if num1 in ind_:
                ind_[num1].append(i_)
            else:
                ind_[num1] = [i_]
        for i in range(d**m):
            multi = ['']
            base = np.base_repr(i, d).zfill(m)
            for j in range(m):
                multi = [x + y for x in multi for y in ind_[int(base[j])]]
            ind[i] = [int(x, 2) for x in multi]
    return ind


def is_symmetric(mat: np.ndarray, m: int = 1) -> bool:
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
    d = nq // m + 1
    if nq % m == 0 and nq != m:
        ind = sym_ind(d, m)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape} or multi {m}')
    if mat.ndim == 1:
        for i in range(d**m):
            i_ = ind[i]
            if len(i_) != 1:
                a = mat[i_]
                is_sym = is_sym & np.allclose(a, a[0])
    elif mat.ndim == 2:
        for i in range(d**m):
            i_ = ind[i]
            for j in range(d**m):
                j_ = ind[j]
                if len(i_) != 1 or len(j_) != 1:
                    a = mat[np.ix_(i_, j_)]
                    is_sym = is_sym & np.allclose(a, a[0][0])
    return is_sym


def su2_decoding(qubit: np.ndarray, m: int = 1) -> np.ndarray:
    if qubit.ndim == 2 and (qubit.shape[0] == 1 or qubit.shape[1] == 1):
        qubit = qubit.flatten()
    if qubit.ndim == 2 and qubit.shape[0] != qubit.shape[1]:
        raise ValueError(f'Wrong qubit shape {qubit.shape}')
    if qubit.ndim != 1 and qubit.ndim != 2:
        raise ValueError(f'Wrong qubit shape {qubit.shape}')
    n = qubit.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f'Wrong matrix size {n} is not a power of 2')
    nq = int(np.log2(n))
    d = nq // m + 1
    if nq % m == 0 and nq != m:
        ind = sym_ind(d, m)
    else:
        raise ValueError(f'Wrong matrix shape {qubit.shape} or multi {m}')
    if qubit.ndim == 1:
        qudit = np.zeros(d**m, dtype=CDTYPE)
        for i in range(d**m):
            i_ = ind[i]
            qubit_i = qubit[i_]
            if np.allclose(qubit_i, qubit_i[0]):
                qudit[i] = qubit_i[0] * np.sqrt(len(i_))
            else:
                raise ValueError('Qubit matrix is not symmetric')
    elif qubit.ndim == 2:
        qudit = np.zeros([d**m, d**m], dtype=CDTYPE)
        for i in range(d**m):
            i_ = ind[i]
            for j in range(d**m):
                j_ = ind[j]
                qubit_ij = qubit[np.ix_(i_, j_)]
                if np.allclose(qubit_ij, qubit_ij[0][0]):
                    div = np.sqrt(len(i_)) * np.sqrt(len(j_))
                    qudit[i, j] = qubit_ij[0][0] * div
                else:
                    raise ValueError('Qubit matrix is not symmetric')
    return qudit


def su2_encoding(qudit: np.ndarray, m: int = 1, is_csr: bool = False) -> np.ndarray:
    if qudit.ndim == 2 and (qudit.shape[0] == 1 or qudit.shape[1] == 1):
        qudit = qudit.flatten()
    if qudit.ndim == 2 and qudit.shape[0] != qudit.shape[1]:
        raise ValueError(f'Wrong qudit shape {qudit.shape}')
    if qudit.ndim != 1 and qudit.ndim != 2:
        raise ValueError(f'Wrong qudit shape {qudit.shape}')
    d = round(qudit.shape[0]**(1 / m), 12)
    if d % 1 == 0:
        d = int(d)
        n = 2**((d - 1) * m)
        ind = sym_ind(d, m)
    else:
        raise ValueError(f'Wrong qudit shape {qudit.shape} or multi {m}')
    if qudit.ndim == 1:
        qubit = csr_matrix((n, 1), dtype=CDTYPE)
        for i in range(d**m):
            ind_i = ind[i]
            num_i = len(ind_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            i_ = (ind_i, np.zeros(num_i))
            qubit += csr_matrix((data, i_), shape=(n, 1))
        if not is_csr:
            qubit = qubit.toarray().flatten()
    elif qudit.ndim == 2:
        qubit = csr_matrix((n, n), dtype=CDTYPE)
        for i in range(d**m):
            ind_i = ind[i]
            num_i = len(ind_i)
            for j in range(d**m):
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