import time
import numpy as np
from typing import List
from math import log, comb
from scipy.io import loadmat
from numpy.linalg import norm
from sympy import solve, Symbol
from collections import defaultdict
from utils import updatemat, symmetric_decoding, DTYPE

np.set_printoptions(precision=15, linewidth=1000)


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
    z = complex(z).conjugate()  # Why conjugate?
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
    expr = sum([qudit[i] * np.sqrt(comb(nq, i)) * (-z)**i for i in range(nq + 1)])
    root = solve(expr, z)
    coor_list = [stereographic_projection(z) for z in root]
    coor_q0 = np.array([0, 0, 1], dtype=DTYPE)
    if len(coor_list) < nq:
        for _ in range(nq - len(coor_list)):
            coor_list.append(coor_q0)
    return coor_list


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
updatemat('./data_322/prepared_coor_violation.mat', savd_dict)