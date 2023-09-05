import numpy as np
from utils import *
from numpy.linalg import norm

np.set_printoptions(linewidth=200)

d = 10
qudit = np.random.rand(d) + 1j * np.random.rand(d)
qudit /= norm(qudit)
qudit = np.outer(qudit.conj(), qudit)
print(qudit.shape, qudit.trace())
qubits = SU2_Encoding(qudit)
print(qubits.shape, qubits.trace())

qudit = np.random.rand(d) + 1j * np.random.rand(d)
qudit /= norm(qudit)
qubits = SU2_Encoding(qudit)
print(qudit.shape)
print(qubits.shape)