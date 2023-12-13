import re
import time
import torch
import numpy as np
import torch.nn as nn
from h5py import File
from torch import optim
from QudiTop.gates import *
from scipy.io import loadmat
from numpy.linalg import norm
from QudiTop.circuit import Circuit
from QudiTop.global_var import DTYPE
from scipy.stats import unitary_group
from QudiTop.expectation import Expectation
from utils import fidelity, reduced_density_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Torch Device:', device)


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


def Cd(d, name, obj, ctrl, state):
    if d != 3:
        raise ValueError('Only works when d = 3')
    circ = Circuit(d, 2)
    circ += RZ(d, [0, 1], f'{name}RZ01').on(obj, ctrl, state)
    circ += RZ(d, [0, 2], f'{name}RZ02').on(obj, ctrl, state)
    circ += GP(d, f'{name}phase').on(obj, ctrl, state)
    return circ


def qutrit_ansatz(gate: UMG, with_phase: bool = False):
    d = gate.dim
    obj = gate.obj_qudits
    name = f'{gate.name}_'
    circ = Circuit(d, 2)
    if len(obj) == 1:
        circ += ZYZ(d, f'{name}', obj[0])
    elif len(obj) == 2:
        circ += ZYZ(d, f'{name}U1_', obj[0])
        circ += Cd(d, f'{name}C1_', obj[0], obj[1], 1)
        circ += ZYZ(d, f'{name}U2_', obj[0])
        circ += Cd(d, f'{name}C2_', obj[0], obj[1], 2)
        circ += ZYZ(d, f'{name}U3_', obj[0])
        circ += RY(d, [1, 2], f'{name}RY12').on(obj[1], obj[0], 2)
        circ += RY(d, [1, 2], f'{name}RY11').on(obj[1], obj[0], 1)
        circ += RY(d, [1, 2], f'{name}RY10').on(obj[1], obj[0], 0)
        circ += ZYZ(d, f'{name}U4_', obj[0])
        circ += Cd(d, f'{name}C3_', obj[0], obj[1], 2)
        circ += ZYZ(d, f'{name}U5_', obj[0])
        circ += RY(d, [0, 1], f'{name}RY22').on(obj[1], obj[0], 2)
        circ += RY(d, [0, 1], f'{name}RY21').on(obj[1], obj[0], 1)
        circ += RY(d, [0, 1], f'{name}RY20').on(obj[1], obj[0], 0)
        circ += ZYZ(d, f'{name}U6_', obj[0])
        circ += Cd(d, f'{name}C4_', obj[0], obj[1], 0)
        circ += ZYZ(d, f'{name}U7_', obj[0])
        circ += RY(d, [1, 2], f'{name}RY32').on(obj[1], obj[0], 2)
        circ += RY(d, [1, 2], f'{name}RY31').on(obj[1], obj[0], 1)
        circ += RY(d, [1, 2], f'{name}RY30').on(obj[1], obj[0], 0)
        circ += ZYZ(d, f'{name}U8_', obj[0])
        circ += Cd(d, f'{name}C5_', obj[0], obj[1], 2)
        circ += ZYZ(d, f'{name}U9_', obj[0])
    else:
        raise ValueError('Only works when nq <= 2')
    if with_phase:
        circ += [GP(d, f'{name}phase').on(i) for i in obj]
    return circ


rdm3 = loadmat('./mat/322_d3_num1_model957_RDM.mat')['RDM_3']
g = File('./mat/322_d3_num1_model957_RDM3_gates_L10_N7.mat', 'r')
position = g['RDM_site'][:] - 1  # subtract index of matlab to python
l = list(g.keys())  # list of HDF5 gates file keys
d = int(g['d'][0])  # dimension of qudit state
f = g['fidelity'][0][0]  # fidelity of gates
g_name = [x for x in l if 'gates' in x]  # list of Q_gates_?
key = lambda x: [int(s) if s.isdigit() else s for s in re.split('(\d+)', x)]
g_name = sorted(g_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
k = g[g_name[0]].shape[0]  # number of gates in one layer
gates = [[g[g[i][j]][:].view('complex').T for j in range(k)] for i in g_name]
g.close()

d, nq = 3, 7
circ = Circuit(d, nq)
ansatz = Circuit(d, nq)
for i in range(len(g_name)):
    for j in range(k):
        mat = gates[i][j]
        name = f'G{j + 1}_L{i + 1}'
        gate = UMG(d, mat, name=name).on([j, j + 1])
        circ += gate
        ansatz += qutrit_ansatz(gate, True)

pr = ansatz.get_parameters()
g_num = len(ansatz.gates)
p_num = len(pr)
print('Number of qudits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

psi = circ.get_qs()
rho = np.outer(psi, psi.conj())
print('Hamiltonian Dimension:', rho.shape)
Ham = [(1, UMG(d, rho).on(list(range(nq))))]
expect = Expectation(Ham)

rho_rdm = reduced_density_matrix(psi, d, position)
print('rho norm: %.20f' % norm(rdm3 - rho_rdm, 2))
print('rho fidelity: %.20f' % fidelity(rdm3, rho_rdm))

start = time.perf_counter()
p0 = np.random.uniform(-1, 1, p_num)
target = torch.tensor([1], dtype=DTYPE).to(device)
ansatz.assign_ansatz_parameters(dict(zip(pr, p0)))
optimizer = optim.Adam(ansatz.parameters(), lr=1e-2)
for i in range(10000):
    out = expect(ansatz()).to(device)
    loss = nn.L1Loss()(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 1 == 0:
        t = time.perf_counter() - start
        print('Loss: %.15f, Fidelity: %.15f, %3d, %.4f' % (loss, out, i, t))
    if loss < 1e-4:
        break
t = time.perf_counter() - start
print('Loss: %.15f, Fidelity: %.15f, %3d, %.4f' % (loss, out, i, t))

pr_res = ansatz.get_parameters()
psi_res = ansatz.get_qs()
print('psi norm: %.20f' % norm(psi - psi_res, 2))
print('psi fidelity: %.20f' % fidelity(psi, psi_res))

rho_res = reduced_density_matrix(psi_res, d, position)
print('rho fidelity: %.20f' % fidelity(rdm3, rho_res))

end = time.perf_counter()
print('Runtime: %f' % (end - start))