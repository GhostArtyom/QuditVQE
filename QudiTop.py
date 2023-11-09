import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from utils import fidelity
from numpy.linalg import norm
from QudiTop.circuit import Circuit
from QudiTop.global_var import DTYPE
from scipy.stats import unitary_group
from QudiTop.expectation import Expectation
from QudiTop.gates import X, RY, RZ, GP, UMG

np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)


def Cd(d, pr, state, obj, ctrl):
    if d != 3:
        raise ValueError('Only works when d = 3')
    if state < 0 or state >= d:
        raise ValueError(f'¦{state}⟩ control state should in 0 to {d-1}')
    circ = Circuit(d, nq)
    circ += RZ(d, [0, 1], f'{pr}RZ01').on(obj)
    circ += X(d, [0, 1]).on(obj, ctrl, state)
    circ += RZ(d, [0, 1], f'{pr}-RZ01').on(obj)
    circ += X(d, [0, 1]).on(obj, ctrl, state)
    circ += RZ(d, [0, 2], f'{pr}RZ02').on(obj)
    circ += X(d, [0, 2]).on(obj, ctrl, state)
    circ += RZ(d, [0, 2], f'{pr}-RZ02').on(obj)
    circ += X(d, [0, 2]).on(obj, ctrl, state)
    circ += GP(d, f'{pr}phase_obj').on(obj)
    circ += GP(d, f'{pr}phase_ctrl').on(ctrl)
    return circ


def qutrit_ansatz(gate: UMG, with_phase: bool = False):
    obj = gate.obj_qudits
    name = f'{gate.name}_'
    circ = Circuit(d, nq)
    index = [[0, 2], [1, 2], [0, 2]]
    if len(obj) == 1:
        for i, ind in enumerate(index):
            str_pr = f'{"".join(str(i) for i in ind)}_{i}'
            circ += RZ(d, ind, f'{name}RZ{str_pr}').on(obj[0])
            circ += RY(d, ind, f'{name}RY{str_pr}').on(obj[0])
            circ += RZ(d, ind, f'{name}Rz{str_pr}').on(obj[0])
    elif len(obj) == 2:
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U1').on(obj[1]))
        circ += Cd(d, f'{name}Cd1', 1, obj[1], obj[0])
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U2').on(obj[1]))
        circ += Cd(d, f'{name}Cd2', 2, obj[1], obj[0])
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U3').on(obj[1]))
        circ += RY(d, [1, 2], f'{name}RY1').on(obj[0], obj[1], 0)
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U4').on(obj[1]))
        circ += Cd(d, f'{name}Cd3', 2, obj[1], obj[0])
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U5').on(obj[1]))
        circ += RY(d, [0, 1], f'{name}RY2').on(obj[0], obj[1], 1)
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U6').on(obj[1]))
        circ += Cd(d, f'{name}Cd4', 0, obj[1], obj[0])
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U7').on(obj[1]))
        circ += RY(d, [1, 2], f'{name}RY3').on(obj[0], obj[1], 2)
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U8').on(obj[1]))
        circ += Cd(d, f'{name}Cd5', 2, obj[1], obj[0])
        circ += qutrit_ansatz(UMG(d, np.eye(d), name=f'{name}U9').on(obj[1]))
    else:
        raise ValueError('Only works when nq = 2')
    if with_phase:
        circ += [GP(d, 'phase').on(i) for i in obj]
    return circ


d, nq = 3, 2
circ = Circuit(d, nq)
ansatz = Circuit(d, nq)
mat = unitary_group.rvs(d**nq, random_state=42)
obj = list(range(nq))
gate = UMG(d, mat, name=f'mat').on(obj)
circ += gate
ansatz += qutrit_ansatz(gate)
ansatz += [GP(d, 'phase').on(i) for i in obj]
print(ansatz)

pr = ansatz.get_parameters()
g_num = len(ansatz.gates)
p_num = len(pr)
print('Number of qudits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

psi = circ.get_qs()
rho = np.outer(psi, psi.conj())
print('Hamiltonian Dimension:', rho.shape)
Ham = [(1, UMG(d, rho).on(obj[::-1]))]
expect = Expectation(Ham)

start = time.perf_counter()
p0 = np.random.uniform(-1, 1, p_num)
target = torch.tensor([1], dtype=DTYPE)
ansatz.assign_ansatz_parameters(dict(zip(pr, p0)))
optimizer = optim.Adam(ansatz.parameters(), lr=1e-1)
for i in range(1000):
    out = expect(ansatz())
    loss = nn.L1Loss()(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        t = time.perf_counter() - start
        print('Loss: %.15f, Fidelity: %.15f, %3d, %.4f' % (loss, out, i, t))
    if loss < 1e-8:
        break
t = time.perf_counter() - start
print('Loss: %.15f, Fidelity: %.15f, %3d, %.4f' % (loss, out, i, t))

pr_res = ansatz.get_parameters()
psi_res = ansatz.get_qs(pr_res)
print('psi norm: %.20f' % norm(psi - psi_res, 2))
print('psi fidelity: %.20f' % fidelity(psi, psi_res)**2)

end = time.perf_counter()
print('Runtime: %f' % (end - start))