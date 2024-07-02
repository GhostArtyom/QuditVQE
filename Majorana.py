import time
import itertools
import numpy as np
from scipy.io import loadmat
from functools import reduce
from numpy.linalg import norm
from typing import List, Tuple
from sympy import solve, Symbol
from collections import defaultdict
from math import log, comb, factorial
from utils import updatemat, symmetric_decoding

DTYPE = np.float64
CDTYPE = np.complex128


def q(ket: str, dim: int = 2) -> np.ndarray:
    return np.eye(dim**len(ket), dtype=CDTYPE)[int(ket, dim)]


def num_qudits(state: np.ndarray, dim: int = 2) -> int:
    if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
        state = state.flatten()
    if state.ndim == 2 and state.shape[0] != state.shape[1]:
        raise ValueError(f'Wrong state shape {state.shape}')
    if state.ndim != 1 and state.ndim != 2:
        raise ValueError(f'Wrong state shape {state.shape}')
    if not isinstance(dim, (int, np.int64)):
        raise ValueError(f'Wrong dimension type {dim} {type(dim)}')
    n = state.shape[0]
    nq = round(log(n, dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong state size {n} is not a power of {dim}')
    return int(nq)


def state_site(state: np.ndarray, dim: int = 2, site: int = 0) -> np.ndarray:
    if not isinstance(dim, (int, np.int64)):
        raise ValueError(f'Wrong dimension type {dim} {type(dim)}')
    if not isinstance(site, (int, np.int64)):
        raise ValueError(f'Wrong dimension type {site} {type(site)}')
    nq = num_qudits(state, dim)
    if site < 0 or site > nq:
        raise ValueError(f'Wrong site index {site} is not in 0 to {nq}')
    ind = defaultdict(list)
    for i in range(dim**nq):
        base = np.base_repr(i, dim).zfill(nq)
        ind[int(base[site])].append(i)
    state_site = np.array([sum(state[ind[i]]) for i in range(dim)])
    state_site /= norm(state_site)
    return state_site


def stereographic_projection(z: complex) -> np.ndarray:
    abs_z2 = np.abs(z)**2
    coor = [2 * np.real(z), 2 * np.imag(z), abs_z2 - 1] / (abs_z2 + 1)
    return coor


def Majorana_representation(state: np.ndarray, dim: int = 2) -> List[np.ndarray]:
    if dim == 2:
        nq = num_qudits(state)
        qudit = symmetric_decoding(state)
    else:
        nq = dim - 1
        qudit = state
    z = Symbol('z')
    expr = sum([qudit[i] * np.sqrt(comb(nq, i)) * z**i for i in range(nq + 1)])
    root = solve(expr, z)
    coor_list = [stereographic_projection(complex(-r).conjugate()) for r in root]
    coor_q0 = [np.array([0, 0, 1], dtype=DTYPE)]
    coor_list.extend(coor_q0 * (nq - len(root)))
    return coor_list


def Majorana_points(points: List[np.ndarray], dim: int = 2) -> np.ndarray:
    nq_list = [num_qudits(i, dim) for i in points]
    if len(set(nq_list)) != 1:
        raise ValueError(f'Wrong number of qubits {nq_list} is not equal')
    num = len(points)
    points_list, K_list = [], []
    for ind in itertools.permutations(range(num)):
        points_perm = [points[i] for i in ind]
        points_list.append(reduce(np.kron, points_perm))
        K_perm = 1
        for i, j in enumerate(ind):
            K_perm *= np.inner(points[i].conj(), points[j])
        K_list.append(K_perm)
    K = factorial(num) * sum(K_list)
    state = sum(points_list) / np.sqrt(K)
    return state


def angle_to_coor(theta: float, phi: float) -> np.ndarray:
    coor = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    return np.array(coor)


def angle_to_state(theta: float, phi: float) -> np.ndarray:
    state = [np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)]
    return np.array(state)


def coor_to_angle(coor: np.ndarray) -> Tuple[float]:
    theta = np.arccos(coor[2])
    phi = np.arctan2(coor[1], coor[0])
    return theta, phi


def coor_to_state(coor: np.ndarray) -> np.ndarray:
    return angle_to_state(*coor_to_angle(coor))


def state_to_angle(state: np.ndarray) -> Tuple[float]:
    if state.shape != (2, ):
        raise ValueError(f'Wrong state shape {state.shape}')
    state /= norm(state)
    phase = np.angle(state[0])
    state /= np.exp(1j * phase)
    theta = 2 * np.arccos(state[0].real)
    phi = np.angle(state[1])
    return theta, phi


def state_to_coor(state: np.ndarray) -> np.ndarray:
    return angle_to_coor(*state_to_angle(state))


if __name__ == '__main__':
    dim = 3
    vec = 1
    savd_dict = {}
    start = time.perf_counter()
    for num in range(1, 6):
        for D in [5, 6, 7, 8, 9]:
            load = loadmat('./data_322/prepared_state_violation.mat')
            state = load[f'num{num}_D{D}_vec{vec}'][0]
            nq = num_qudits(state, dim)
            for s in range(nq):
                site = state_site(state, dim, s)
                coor = Majorana_representation(site, dim)
                savd_dict[f'num{num}_D{D}_vec{vec}_site{s+1}'] = coor
            t = time.perf_counter() - start
            print(f'num{num} D{D} vec{vec} {t:.2f}')
    # updatemat('./data_322/prepared_coor_violation.mat', savd_dict)