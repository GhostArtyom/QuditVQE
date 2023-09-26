import numpy as np
from math import log
from typing import List
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from numpy.linalg import det, eigh, svd
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RY, RZ, Rxx, Ryy, Rzz, U3, GlobalPhase, UnivMathGate

optional_basis = ['zyz', 'u3']
A = np.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]])
M = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / np.sqrt(2)


def is_power_of_two(num: int) -> bool:
    num = int(num)
    return (num & (num - 1) == 0) and num != 0


def is_unitary(mat: np.ndarray) -> bool:
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        d = mat.shape[0]
        i = np.eye(d)
        a = mat @ mat.conj().T
        return np.allclose(i, a)
    else:
        raise ValueError(f'Wrong matrix shape {mat.shape}')


def is_symmetric(mat: np.ndarray) -> bool:
    if mat.ndim == 2 and (mat.shape[0] == 1 or mat.shape[1] == 1):
        mat = mat.flatten()
    if mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
        raise ValueError(f'Wrong matrix shape {mat.shape}')
    b = True
    nq_bin = {}
    d = mat.shape[0]
    if not is_power_of_two(d):
        raise ValueError(f'{d} is not a power of 2')
    nq = int(np.log2(d)) + 1
    for i in range(d):
        num1 = bin(i).count('1')
        if num1 in nq_bin:
            nq_bin[num1].append(i)
        else:
            nq_bin[num1] = [i]
    if mat.ndim == 1:
        for i in range(nq):
            ii = nq_bin[i]
            if len(ii) != 1:
                a = mat[ii]
                b = b & np.allclose(a, a[0])
    elif mat.ndim == 2:
        for i in range(nq):
            ii = nq_bin[i]
            for j in range(nq):
                jj = nq_bin[j]
                if len(ii) != 1 or len(jj) != 1:
                    a = mat[np.ix_(ii, jj)]
                    b = b & np.allclose(a, a[0][0])
    else:
        raise ValueError('Wrong Input!')
    return b


def decompose_zyz(mat: np.array):
    phase = -np.angle(det(mat)) / 2
    matU = np.exp(1j * phase) * mat
    cos = np.sqrt(np.real(matU[0, 0] * matU[1, 1]))
    if abs(cos) > 1:
        cos = np.clip(cos, -1, 1)
    theta = 2 * np.arccos(cos)
    phi = np.angle(matU[1, 1]) + np.angle(matU[1, 0])
    lam = np.angle(matU[1, 1]) - np.angle(matU[1, 0])
    return phase, theta, phi, lam


def decompose_u3(mat: np.array):
    phase, theta, phi, lam = decompose_zyz(mat)
    phase += (phi + lam) / 2
    return phase, theta, phi, lam


def one_qubit_decompose(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = True) -> Circuit:
    name_phase = gate.name + '_phase'
    name_theta = gate.name + '_theta'
    name_phi = gate.name + '_phi'
    name_lam = gate.name + '_lam'
    obj = gate.obj_qubits
    mat = gate.matrix()
    circ = Circuit()
    if mat.shape != (2, 2):
        raise ValueError('Gate is not one qubit')
    if not np.allclose(np.eye(2), mat @ mat.conj().T):
        raise ValueError('Gate is not unitary')
    if basis == 'zyz':
        phase, theta, phi, lam = decompose_zyz(mat)
        circ += RZ(name_lam).on(obj)
        circ += RY(name_theta).on(obj)
        circ += RZ(name_phi).on(obj)
    elif basis == 'u3':
        circ += U3(name_theta, name_phi, name_lam).on(obj)
        phase, theta, phi, lam = decompose_u3(mat)
    else:
        raise ValueError(f'{basis} is not in {optional_basis}')
    if with_phase:
        circ += GlobalPhase(name_phase).on(obj)
        pr = {name_phase: phase, name_phi: phi, name_theta: theta, name_lam: lam}
    else:
        pr = {name_phi: phi, name_theta: theta, name_lam: lam}
    return circ, pr


def simult_svd(mat1: np.ndarray, mat2: np.ndarray):
    d = mat1.shape[0]
    u_a, d_a, v_a_h = svd(mat1)
    u_a_h = u_a.conj().T
    v_a = v_a_h.conj().T
    if np.count_nonzero(d_a) != d:
        raise ValueError('Not implemented yet for the situation that mat1 is not full-rank')
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
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
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


def two_qubit_decompose(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = True) -> Circuit:
    name_rxx = gate.name + '_Rxx'
    name_ryy = gate.name + '_Ryy'
    name_rzz = gate.name + '_Rzz'
    name_phase = gate.name + '_phase'
    obj0, obj1 = gate.obj_qubits
    mat = gate.matrix()
    circ = Circuit()
    circ_d = Circuit()
    if mat.shape != (4, 4):
        raise ValueError('Gate is not two qubit')
    if not np.allclose(np.eye(4), mat @ mat.conj().T):
        raise ValueError('Gate is not unitary')
    ur = np.real(M.conj().T @ mat @ M)
    ui = np.imag(M.conj().T @ mat @ M)
    QL, QR, D = simult_svd(ur, ui)
    A1, A0 = kron_factor_4x4_to_2x2s(M @ QL @ M.conj().T)
    B1, B0 = kron_factor_4x4_to_2x2s(M @ QR.T @ M.conj().T)
    k = (A.T / 4) @ np.angle(np.diag(D))
    pr = {name_rxx: -2 * k[1], name_ryy: -2 * k[2], name_rzz: -2 * k[3], name_phase: k[0] / -2}
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
                raise ValueError(f'{basis} is not in {optional_basis}')
            circ_d += gate_d
            pr.update(para)
        else:
            circ_d += g
    return circ_d, pr


def partial_trace(rho: np.ndarray, ind: int) -> np.ndarray:
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    elif rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
        rho = np.outer(rho, rho.conj())
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    d = rho.shape[0]
    if not is_power_of_two(d):
        raise ValueError(f'{d} is not a power of 2')
    n = d // 2
    nq = int(np.log2(n))
    if ind < 0 or ind > nq:
        raise ValueError(f'Index should be 0 to {nq}')
    pt = np.zeros([n, n], dtype=np.complex128)
    for i in range(n):
        i_ = bin(i)[2::].zfill(nq)
        i0 = int(i_[:ind] + '0' + i_[ind:], 2)
        i1 = int(i_[:ind] + '1' + i_[ind:], 2)
        for j in range(n):
            j_ = bin(j)[2::].zfill(nq)
            j0 = int(j_[:ind] + '0' + j_[ind:], 2)
            j1 = int(j_[:ind] + '1' + j_[ind:], 2)
            pt[i, j] = rho[i0, j0] + rho[i1, j1]
    return pt


def reduced_density_matrix(rho: np.ndarray, position: List[int]) -> np.ndarray:
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    elif rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
        rho = np.outer(rho, rho.conj())
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f'Wrong state shape {rho.shape}')
    d = rho.shape[0]
    if not is_power_of_two(d):
        raise ValueError(f'{d} is not a power of 2')
    nq = int(np.log2(d))
    p = [x for x in range(nq) if x not in position]
    for i in p[::-1]:
        rho = partial_trace(rho, i)
    return rho


def fidelity(rho: np.ndarray, sigma: np.ndarray, sqrt: bool = True) -> float:
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if sigma.ndim == 2 and (sigma.shape[0] == 1 or sigma.shape[1] == 1):
        sigma = sigma.flatten()
    if rho.ndim == 1 and sigma.ndim == 1:
        f = np.abs(rho.conj() @ sigma)
    elif rho.ndim == 1 and sigma.ndim == 2:
        f = np.sqrt(np.real(rho.conj().T @ sigma @ rho))
    elif rho.ndim == 2 and sigma.ndim == 1:
        f = np.sqrt(np.real(sigma.conj().T @ rho @ sigma))
    elif rho.ndim == 2 and sigma.ndim == 2:
        f = np.real(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))
    else:
        raise ValueError('Wrong Input!')
    if sqrt:
        return f
    else:
        return f**2


def su2_encoding(qudit: np.ndarray, m: int = 1) -> np.ndarray:
    if qudit.ndim == 2 and (qudit.shape[0] == 1 or qudit.shape[1] == 1):
        qudit = qudit.flatten()
    if qudit.ndim == 2 and qudit.shape[0] != qudit.shape[1]:
        raise ValueError(f'Wrong qudit shape {qudit.shape}')
    if qudit.ndim != 1 and qudit != 2:
        raise ValueError(f'Wrong qudit shape {qudit.shape}')
    if m == 1:
        d = qudit.shape[0]
        n = 2**(d - 1)
        nq_bin = {}
        for i in range(n):
            num1 = bin(i).count('1')
            if num1 in nq_bin:
                nq_bin[num1].append(i)
            else:
                nq_bin[num1] = [i]
    elif qudit.shape[0]**(1 / m) % 1 == 0:
        d = int(qudit.shape[0]**(1 / m))
        nq = (d - 1) * m
        n = 2**nq
        nq_b = {}
        nq_bin = {}
        for i in range(2**(d - 1)):
            num1 = bin(i).count('1')
            bin_i = bin(i)[2::].zfill(d - 1)
            if num1 in nq_b:
                nq_b[num1].append(bin_i)
            else:
                nq_b[num1] = [bin_i]
        for i in range(d**m):
            nq_bin[i] = [int(a + b, 2) for a in nq_b[i // d] for b in nq_b[i % d]]
    else:
        raise ValueError(f'Wrong qudit shape {qudit.shape} or num {m}')
    if qudit.ndim == 1:
        qubit = csr_matrix((1, n), dtype=np.complex128)
        for i in range(d**m):
            ind_i = nq_bin[i]
            num_i = len(ind_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            ind = (np.zeros(num_i), ind_i)
            qubit += csr_matrix((data, ind), shape=(1, n))
        qubit = qubit.toarray().flatten()
    elif qudit.ndim == 2:
        qubit = csr_matrix((n, n), dtype=np.complex128)
        for i in range(d**m):
            ind_i = nq_bin[i]
            num_i = len(ind_i)
            for j in range(d**m):
                ind_j = nq_bin[j]
                num_j = len(ind_j)
                ii = ind_i * num_j
                jj = np.repeat(ind_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubit += csr_matrix((data, (ii, jj)), shape=(n, n))
        qubit = qubit.toarray()
    return qubit
