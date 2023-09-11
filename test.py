import re
import os
import sys
import h5py
import time
import numpy as np
import scipy as sp
from utils import *
from numpy.linalg import det, norm
from scipy.sparse import csr_matrix
from mindquantum.framework import *
from mindquantum.core.gates import *
from scipy.stats import unitary_group
from mindquantum.core.circuit import *
from IPython.display import display_svg
from mindquantum.core.operators import *
from mindquantum.algorithm.nisq import *
from mindquantum.simulator import Simulator
from mindquantum.algorithm.compiler import *
from mindquantum.algorithm.compiler.decompose.utils import *

np.set_printoptions(linewidth=200)

# d = 10
# qudit = np.random.rand(d) + 1j * np.random.rand(d)
# qudit /= norm(qudit)
# qudit = np.outer(qudit.conj(), qudit)
# print(qudit.shape, qudit.trace())
# qubits = su2_encoding(qudit)
# print(qubits.shape, qubits.trace())

# qudit = np.random.rand(d) + 1j * np.random.rand(d)
# qudit /= norm(qudit)
# qubits = su2_encoding(qudit)
# print(qudit.shape)
# print(qubits.shape)

d = 2
gate = unitary_group.rvs(d, random_state=42)
print(gate)
gate_u = UnivMathGate('gate', gate).on(0)
gate_d, pr = one_qubit_decompose(gate_u, 'zyz')
print(gate_d)
print(gate_d.matrix(pr))
print(gate_d.apply_value(pr))
print(norm(gate - gate_d.matrix(pr)))

gate_d = euler_decompose(gate_u, 'zyz')
print(gate_d)
print(gate_d.matrix())
print(norm(gate - gate_d.matrix()))

print(pr)
print(params_zyz(gate))
print(params_u3(gate))