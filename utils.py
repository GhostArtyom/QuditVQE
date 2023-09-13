import os
import sys
import pickle
import numpy as np
from typing import List
from numpy.linalg import det, norm
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, U3, GlobalPhase, UnivMathGate

optional_basis = ['zyz', 'u3']


def one_qubit_decompose(gate: UnivMathGate, basis: str = 'zyz', with_phase: bool = True) -> Circuit:
    str_phase = gate.name + '_phase'
    str_theta = gate.name + '_theta'
    str_phi = gate.name + '_phi'
    str_lam = gate.name + '_lam'
    obj = gate.obj_qubits
    mat = gate.matrix()
    d = mat.shape[0]
    circ = Circuit()
    if not np.allclose(np.eye(d), mat @ mat.conj().T):
        raise ValueError('The gate is not unitary')
    phase = -np.angle(det(mat)) / 2
    matU = mat * np.exp(1j * phase)
    cos = np.sqrt(np.real(matU[0, 0] * matU[1, 1]))
    theta = 2 * np.arccos(cos)
    phi = np.angle(matU[1, 1]) + np.angle(matU[1, 0])
    lam = np.angle(matU[1, 1]) - np.angle(matU[1, 0])
    if basis == 'zyz':
        circ += RZ(str_lam).on(obj)
        circ += RY(str_theta).on(obj)
        circ += RZ(str_phi).on(obj)
        if with_phase:
            circ += GlobalPhase(str_phase).on(obj)
    elif basis == 'u3':
        circ += U3(str_theta, str_phi, str_lam).on(obj)
        phase += (phi + lam) / 2
        if with_phase:
            circ += GlobalPhase(str_phase).on(obj)
    else:
        raise ValueError(f'{basis} is not a supported decomposition method of {optional_basis}')
    pr = {str_phase: phase, str_phi: phi, str_theta: theta, str_lam: lam}
    return circ, pr


def partial_trace(rho: np.ndarray, index: List[int]) -> np.ndarray:
    nq = int(rho.shape[0] / 2)
    d = int(np.log2(nq))
    pt = np.zeros([nq, nq], dtype=np.complex128)
    for i in range(nq):
        i_ = bin(i)[2::].zfill(d)
        i0 = int(i_[:index] + '0' + i_[index:], 2)
        i1 = int(i_[:index] + '1' + i_[index:], 2)
        for j in range(nq):
            j_ = bin(j)[2::].zfill(d)
            j0 = int(j_[:index] + '0' + j_[index:], 2)
            j1 = int(j_[:index] + '1' + j_[index:], 2)
            pt[i, j] = rho[i0, j0] + rho[i1, j1]
    return pt


def reduced_density_matrix(rho: np.ndarray, position: List[int]) -> np.ndarray:
    nq = int(np.log2(rho.shape[0]))
    p = [x for x in range(nq) if x not in position]
    for i in p[::-1]:
        rho = partial_trace(rho, i)
    return rho


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    if rho.ndim == 2 and (rho.shape[0] == 1 or rho.shape[1] == 1):
        rho = rho.flatten()
    if sigma.ndim == 2 and (sigma.shape[0] == 1 or sigma.shape[1] == 1):
        sigma = sigma.flatten()
    if rho.ndim == 1 and sigma.ndim == 1:
        f = np.abs(rho.conj() @ sigma)
    elif rho.ndim == 1 and sigma.ndim == 2:
        f = np.real(np.sqrt(rho.conj().T @ sigma @ rho))
    elif rho.ndim == 2 and sigma.ndim == 1:
        f = np.real(np.sqrt(sigma.conj().T @ rho @ sigma))
    elif rho.ndim == 2 and sigma.ndim == 2:
        f = np.real(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))
    else:
        raise ValueError('Wrong Input!')
    return f


def su2_encoding(qudit: np.ndarray) -> np.ndarray:
    d = np.shape(qudit)
    if len(d) == 2 and d[0] == d[1]:
        d = d[0]
        nq = d - 1
        qubits = csr_matrix(np.zeros([2**nq, 2**nq], dtype=np.complex128))
        if d < 15:
            nq_bin = {}
            for i in range(2**nq):
                num1 = bin(i).count('1')
                if num1 in nq_bin:
                    nq_bin[num1].append(i)
                else:
                    nq_bin[num1] = [i]
        elif d >= 15 and d <= 25:
            name = 'nq_bin_d=%d.pkl' % d
            path = os.path.join(sys.path[0], 'nq_bin', name)
            f_read = open(path, 'rb')
            nq_bin = pickle.load(f_read)
            f_read.close()
        else:
            raise ValueError('d is over 25!')
        for i in range(d):
            qubits_i = nq_bin[i]
            num_i = len(qubits_i)
            for j in range(d):
                qubits_j = nq_bin[j]
                num_j = len(qubits_j)
                ii = qubits_i * num_j
                jj = np.repeat(qubits_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubits += csr_matrix((data, (ii, jj)), shape=(2**nq, 2**nq))
        qubits = qubits.toarray()
    elif len(d) == 1 or (len(d) == 2 and (d[0] == 1 or d[1] == 1)):
        if len(d) == 2 and d[0] == 1:
            qudit = qudit.flatten()
            d = d[1]
        elif len(d) == 2 and d[1] == 1:
            qudit = qudit.flatten()
            d = d[0]
        else:
            d = d[0]
        nq = d - 1
        qubits = csr_matrix(np.zeros(2**nq, dtype=np.complex128))
        if d < 15:
            nq_bin = {}
            for i in range(2**nq):
                num1 = bin(i).count('1')
                if num1 in nq_bin:
                    nq_bin[num1].append(i)
                else:
                    nq_bin[num1] = [i]
        elif d >= 15 and d <= 25:
            name = 'nq_bin_d=%d.pkl' % d
            path = os.path.join(sys.path[0], 'nq_bin', name)
            f_read = open(path, 'rb')
            nq_bin = pickle.load(f_read)
            f_read.close()
        else:
            raise ValueError('d is over 25!')
        for i in range(d):
            qubits_i = nq_bin[i]
            num_i = len(qubits_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            ind = (np.zeros(num_i), qubits_i)
            qubits += csr_matrix((data, ind), shape=(1, 2**nq))
        qubits = qubits.toarray().flatten()
        qubits /= norm(qubits)
    else:
        raise ValueError('Wrong Input!')
    return qubits
