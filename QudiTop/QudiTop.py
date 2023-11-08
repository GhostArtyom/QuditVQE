import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from utils import fidelity
from .circuit import Circuit
from numpy.linalg import norm
from .expectation import Expectation
from .gates import X, RY, RZ, GP, UMG
from scipy.stats import unitary_group

np.set_printoptions(linewidth=200)

d, nq = 3, 1
ansatz = Circuit(d, nq)
for i, ind in enumerate([[0, 2], [1, 2], [0, 2]]):
    str_ind = ''.join(str(i) for i in ind)
    ansatz += RZ(d, ind, f'RZ{str_ind}_{i}').on(0)
    ansatz += RY(d, ind, f'RY{str_ind}_{i}').on(0)
    ansatz += RZ(d, ind, f'Rz{str_ind}_{i}').on(0)
# ansatz += GP(d, f'phase').on(0)
p_num = len(ansatz.get_parameters())
g_num = len(ansatz.gates)
print('Number of qudits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

mat = unitary_group.rvs(d)
psi0 = np.zeros(d)
psi0[0] = 1
psi = mat @ psi0
rho = np.outer(psi, psi.conj())
ham = [(1.0, UMG(d, rho).on(0))]
expect = Expectation(ham)

loss_fn = nn.L1Loss()
start = time.perf_counter()
optimizer = optim.Adam(ansatz.parameters(), lr=1e-1)
p0 = np.random.uniform(-1, 1, p_num)
ansatz.assign_ansatz_parameters(p0)
for i in range(1000):
    out = expect(ansatz())
    loss = loss_fn(out, torch.Tensor([1]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        t = time.perf_counter() - start
        print('Loss: %.15f, Fidelity: %.15f, %2d, %.4f' % (loss, out, i, t))
    if loss < 1e-8:
        t = time.perf_counter() - start
        print('Loss: %.15f, Fidelity: %.15f, %2d, %.4f' % (loss, out, i, t))
        break
pr_res = ansatz.get_parameters()
psi_res = ansatz.get_qs(pr_res)
print('psi norm: %.20f' % norm(psi - psi_res, 2))
print('psi fidelity: %.20f' % fidelity(psi, psi_res))