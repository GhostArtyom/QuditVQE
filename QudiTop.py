import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from utils import fidelity
from numpy.linalg import norm
from QudiTop.circuit import Circuit
from QudiTop.global_var import DTYPE
from QudiTop.expectation import Expectation
from QudiTop.gates import X, RY, RZ, GP, UMG
from scipy.stats import unitary_group

np.set_printoptions(linewidth=200)


def one_qutrit_ansatz(gate: UMG, with_phase: bool = False):
    obj = gate.obj_qudits
    name = f'{gate.name}_'
    circ = Circuit(d, nq)
    for i, ind in enumerate([[0, 2], [1, 2], [0, 2]]):
        str_pr = f'{"".join(str(i) for i in ind)}_{i}'
        circ += RZ(d, ind, f'{name}RZ{str_pr}').on(obj)
        circ += RY(d, ind, f'{name}RY{str_pr}').on(obj)
        circ += RZ(d, ind, f'{name}Rz{str_pr}').on(obj)
    if with_phase:
        for i in obj:
            circ += GP(d, f'phase').on(i)
    return circ


d, nq = 3, 3
circ = Circuit(d, nq)
ansatz = Circuit(d, nq)
for i in range(nq):
    mat = unitary_group.rvs(d)
    print(mat)
    gate = UMG(d, mat, name=f'mat{i}').on(i)
    circ += gate
    ansatz += one_qutrit_ansatz(gate)

g_num = len(ansatz.gates)
p_num = len(ansatz.get_parameters())
print('Number of qudits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

psi = circ.get_qs()
rho = np.outer(psi, psi.conj())
obj = list(range(nq))[::-1]
Ham = [(1, UMG(d, rho).on(obj))]
expect = Expectation(Ham)
print('Hamiltonian Dimension:', rho.shape)

start = time.perf_counter()
target = torch.tensor([1], dtype=DTYPE)
p0 = np.random.uniform(-1, 1, p_num)
ansatz.assign_ansatz_parameters(p0)
optimizer = optim.Adam(ansatz.parameters(), lr=1e-1)
for i in range(1000):
    out = expect(ansatz())
    loss = nn.L1Loss()(out, target)
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

end = time.perf_counter()
print('Runtime: %f' % (end - start))