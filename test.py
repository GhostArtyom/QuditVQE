import re
import os
import sys
import time
import numpy as np
import scipy as sp
from utils import *
from math import log
from h5py import File
import mindspore as ms
from numpy.linalg import *
from scipy.linalg import *
from mindquantum.simulator import *
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from mindquantum.framework import *
from mindquantum.core.gates import *
from scipy.stats import unitary_group
from mindquantum.core.circuit import *
from mindquantum.core.operators import *
from mindquantum.algorithm.nisq import *
from mindspore.common.initializer import *
from mindquantum.algorithm.compiler import *

np.set_printoptions(linewidth=200)

# su2_encoding
d, m = 3, 2
# a = np.arange(d) + 1
# a = np.arange(d**2).reshape(d, d) + 1
# a = np.random.rand(d) + 1j * np.random.rand(d)
a = np.random.rand(d, d) + 1j * np.random.rand(d, d)
for i in range(m):
    qudit = a if i == 0 else np.kron(qudit, a)
qubit = su2_encoding(qudit, m)
# print(qudit, qudit.shape)
# print(qubit, qubit.shape)
print(qudit.shape, qubit.shape)
print(is_symmetric(qubit, m))
# su2_decoding
decode = su2_decoding(qubit, m)
print(np.allclose(qudit, decode))

# partial_trace
d, m = 3, 3
a, b = {}, 1
# np.random.seed(42)
for i in range(m):
    psi = np.random.rand(d) + 1j * np.random.rand(d)
    psi /= norm(psi)
    a[i] = np.outer(psi, psi.conj())
    print(a[i])
    rho = a[i] if i == 0 else np.kron(rho, a[i])
print(rho.shape, np.trace(rho))
ind = 2
pt = partial_trace(rho, d, ind)
for i in range(m):
    if i == ind:
        b *= 1
    elif i != ind and isinstance(b, int):
        b = a[i]
    else:
        b = np.kron(b, a[i])
print(np.allclose(b, pt))
# reduced_density_matrix
position = [0, 1]
for ind, i in enumerate(position):
    b = a[i] if ind == 0 else np.kron(b, a[i])
rdm = reduced_density_matrix(rho, d, position)
print(np.allclose(b, rdm))

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