import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from QudiTop.gates import *
from numpy.linalg import norm
from QudiTop.circuit import Circuit
from QudiTop.global_var import DTYPE
from scipy.stats import unitary_group
from QudiTop.expectation import Expectation

np.set_printoptions(linewidth=250)
torch.set_printoptions(linewidth=250)


def ZYZ(d, name, obj, with_phase: bool = False):
    if d != 3:
        raise ValueError('Only works when d = 3')
    circ = Circuit(d, 1)
    index = [[1, 2], [0, 1], [1, 2]]
    for i, ind in enumerate(index):
        str_pr = f'{"".join(str(i) for i in ind)}_{i}'
        circ += RZ(d, ind, f'{name}RZ{str_pr}').on(obj)
        circ += RY(d, ind, f'{name}RY{str_pr}').on(obj)
        circ += RZ(d, ind, f'{name}Rz{str_pr}').on(obj)
        if with_phase:
            circ += GP(d, f'{name}phase_{i}').on(obj)
    return circ


def fidelity(rho: torch.Tensor, sigma: torch.Tensor, sqrt: bool = True) -> float:
    f = torch.abs(torch.trace(rho @ sigma) + 2 * torch.sqrt(torch.det(rho) * torch.det(sigma)))
    return f if sqrt else f**2


d, nq = 3, 1
ansatz = ZYZ(d, 'mat_', 0, True)
mat = np.random.rand(d**nq) + 1j * np.random.rand(d**nq)
mat /= norm(mat)
mat = np.outer(mat, mat.conj().T)
mat = torch.tensor(mat)

pr = ansatz.get_parameters()
g_num = len(ansatz.gates)
p_num = len(pr)

start = time.perf_counter()
p0 = np.random.uniform(-1, 1, p_num)
target = torch.tensor([1], dtype=DTYPE)
ansatz.assign_ansatz_parameters(dict(zip(pr, p0)))
optimizer = optim.Adam(ansatz.parameters(), lr=1e-2)
for i in range(500):
    out = ansatz.matrix(True)
    # out = ansatz.get_qs(grad_tensor=True)
    loss = 1 - fidelity(out, mat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        t = time.perf_counter() - start
        print('Loss: %.15f, Fidelity: %.15f, %3d, %.4f' % (loss, 1 - loss, i, t))
    if loss < 1e-8:
        break
t = time.perf_counter() - start
print('Loss: %.15f, Fidelity: %.15f, %3d, %.4f' % (loss, 1 - loss, i, t))

print(mat.numpy())
print(ansatz.matrix())

from utils import fidelity
print(fidelity(mat.numpy(), ansatz.matrix()))