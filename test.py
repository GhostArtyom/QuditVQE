import re
import os
import sys
import time
import numpy as np
import scipy as sp
from utils import *
from math import log
from h5py import File
from numpy.linalg import *
from scipy.linalg import *
from scipy.io import loadmat
from scipy.optimize import minimize
from mindquantum.framework import *
from mindquantum.core.gates import *
from scipy.stats import unitary_group
from mindquantum.core.circuit import *
from IPython.display import display_svg
from mindquantum.core.operators import *
from mindquantum.algorithm.nisq import *
from mindquantum.simulator import Simulator
from mindquantum.algorithm.compiler import *
from scipy.sparse import csc_matrix, csr_matrix

np.set_printoptions(linewidth=1000)

# symmetric_encoding
d, m = 3, 2
# a = np.arange(d) + 1
# a = np.arange(d**2).reshape(d, d) + 1
# a = np.random.rand(d) + 1j * np.random.rand(d)
a = np.random.rand(d, d) + 1j * np.random.rand(d, d)
for i in range(m):
    qudit = a if i == 0 else np.kron(qudit, a)
qubit = symmetric_encoding(qudit, m)
# print(qudit, qudit.shape)
# print(qubit, qubit.shape)
print(qudit.shape, qubit.shape)
print(is_symmetric(qubit, m))
# symmetric_decoding
decode = symmetric_decoding(qubit, m)
print(np.allclose(qudit, decode))

# partial_trace
ndim = 2
d, m = 3, 7
a, b = {}, 1
np.random.seed(42)
for i in range(m):
    psi = np.random.rand(d) + 1j * np.random.rand(d)
    psi /= norm(psi)
    a[i] = psi if ndim == 1 else np.outer(psi, psi.conj())
    rho = a[i] if i == 0 else np.kron(rho, a[i])
    # print(a[i])
print(rho.shape)
ind = 2
t1 = time.perf_counter()
pt = partial_trace(rho, d, ind)
t2 = time.perf_counter()
for i in range(m):
    if i == ind:
        b *= 1
    elif i != ind and isinstance(b, int):
        b = a[i]
    else:
        b = np.kron(b, a[i])
b = np.outer(b, b.conj()) if ndim == 1 else b
print(np.allclose(b, pt), pt.shape, t2 - t1)

# reduced_density_matrix
position = [0, 1]
for ind, i in enumerate(position):
    b = a[i] if ind == 0 else np.kron(b, a[i])
t1 = time.perf_counter()
rdm = reduced_density_matrix(rho, d, position)
t2 = time.perf_counter()
b = np.outer(b, b.conj()) if ndim == 1 else b
print(np.allclose(b, rdm), rdm.shape, t2 - t1)

# one_qubit_decompose
d = 2
# gate = unitary_group.rvs(d)
gate = unitary_group.rvs(d, random_state=42)
print(gate)
gate_u = UnivMathGate('gate', gate).on(0)
gate_d, pr = one_qubit_decompose(gate_u, 'zyz')
print(gate_d)
gate_mat = gate_d.matrix(pr)
print(gate_mat)
print(gate_d.apply_value(pr))
print(norm(gate - gate_mat, 2))
print(fidelity(gate, gate_mat))
print(gate / gate_mat, np.allclose(gate, gate_mat))

# two_qubit_decompose
d = 4
# gate = unitary_group.rvs(d)
gate = unitary_group.rvs(d, random_state=42)
print(gate)
gate_u = UnivMathGate('', gate).on([0, 1])
gate_d, pr = two_qubit_decompose(gate_u, 'zyz')
gate_d.summary()
print(gate_d)
gate_mat = gate_d.matrix(pr)
print(gate_mat)
print(gate_d.apply_value(pr))
print(norm(gate - gate_mat, 2))
print(fidelity(gate, gate_mat))
print(gate / gate_mat, np.allclose(gate, gate_mat))
