import re
import os
import sys
import h5py
import time
import numpy as np
import scipy as sp
from utils import *
import mindspore as ms
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from mindquantum.framework import *
from mindquantum.core.gates import *
from scipy.stats import unitary_group
from mindquantum.core.circuit import *
from mindquantum.core.operators import *
from mindquantum.algorithm.nisq import *
from mindspore.common.initializer import *
from mindquantum.simulator import Simulator
from mindquantum.algorithm.compiler import *
from numpy.linalg import det, svd, eigh, norm
from scipy.linalg import expm, sqrtm, block_diag

np.set_printoptions(linewidth=200)

d, m = 3, 2
a = np.arange(d) + 1
for i in range(m):
    qudit = a if i == 0 else np.kron(qudit, a)
qubit = su2_encoding(qudit, m)
print(qudit, qudit.shape)
print(qubit, qubit.shape)
print(is_symmetric(qubit))

# d = 2
# # gate = unitary_group.rvs(d)
# gate = unitary_group.rvs(d, random_state=42)
# print(gate)
# gate_u = UnivMathGate('gate', gate).on(0)
# gate_d, pr = one_qubit_decompose(gate_u, 'zyz')
# print(gate_d)
# print(gate_d.matrix(pr))
# print(gate_d.apply_value(pr))
# print(norm(gate - gate_d.matrix(pr)))

# d = 4
# # gate = unitary_group.rvs(d)
# gate = unitary_group.rvs(d, random_state=42)
# print(gate)
# gate_u = UnivMathGate('', gate).on([0, 1])
# gate_d, pr = two_qubit_decompose(gate_u, 'zyz')
# gate_d.summary()
# print(gate_d)
# gate_mat = gate_d.matrix(pr)
# print(gate_mat)
# print(gate_d.apply_value(pr))
# print(gate / gate_mat, np.allclose(gate, gate_mat))
# print(norm(gate - gate_mat, 2))

# d = 2**3
# np.random.seed(42)
# psi = np.random.rand(d) + 1j * np.random.rand(d)
# psi /= norm(psi)
# rho = np.outer(psi, psi.conj())
# print(psi)
# # print(rho)
# pt = partial_trace(rho, 0)
# rdm = reduced_density_matrix(rho, [1, 2])
# qubits = su2_encoding(psi)
# print(pt - rdm)