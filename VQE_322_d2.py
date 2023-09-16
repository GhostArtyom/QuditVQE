import re
import h5py
import time
import numpy as np
from utils import *
import mindspore as ms
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian

start = time.perf_counter()
np.set_printoptions(linewidth=200)
device_target = ms.get_context("device_target")


def fun(p0, sim_grad, iter_list=None):
    f, g = sim_grad(p0)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    if iter_list is not None:
        iter_list.append(f)
        i = len(iter_list)
        if i % 1 == 0:
            print('Optimal Gap: %.12f, %d' % (f, i))
    return f, g


g = h5py.File('./mat/322_d2_num1_model957_site3_gates_L39_N9_zhu(2).mat', 'r')
# position = g['position'][:] - 1  # subtract index of matlab to python
l = list(g.keys())  # list of HDF5 file keys
g_name = [x for x in l if 'gates' in x]  # list of Q_gates_?
key = lambda x: [int(s) if s.isdigit() else s for s in re.split('(\d+)', x)]
g_name = sorted(g_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
# print(position, g_name)
k = g[g_name[0]].shape[0]  # number of gates in one layer
gates = [[g[g[i][j]][:].view('complex').T for j in range(k)] for i in g_name]
g.close()

r = h5py.File('./mat/322_d2_num1_model957_RDM_v7.3.mat', 'r')
l = list(r.keys())
rdm = [r[i][:].view('complex').T for i in l]
rdm.insert(0, [])
r.close()

gate_pr = {}
ansatz = Circuit()
for i in range(len(g_name)):
    for j in range(k):
        name = 'G' + str(j + 1) + '_L' + str(i + 1)
        mat = gates[i][j]
        if j == k - 1:
            gate_u = UnivMathGate(name, mat).on(k - j - 1)
            gate_d, para = one_qubit_decompose(gate_u)
            gate_pr.update(para)
            ansatz += gate_d
        else:
            gate_u = UnivMathGate(name, mat).on([k - j - 2, k - j - 1])
            gate_d, para = two_qubit_decompose(gate_u)
            gate_pr.update(para)
            ansatz += gate_d

ansatz = ansatz.as_ansatz()
params_name = ansatz.ansatz_params_name
params_size = len(params_name)
print('Number of params: %d' % params_size)

ham = np.kron(np.kron(rdm[3], rdm[3]), rdm[3])
print('Hamiltonian Dimension:', ham.shape)
Ham = Hamiltonian(csr_matrix(ham))

if device_target == 'CPU':
    sim = Simulator('mqvector', ansatz.n_qubits)
elif device_target == 'GPU':
    sim = Simulator('mqvector_gpu', ansatz.n_qubits)
else:
    raise ValueError(f'{device_target} is not applicable')
sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
p0 = np.zeros(params_size)
f, g = sim_grad(p0)

fun(p0, sim_grad)
iter_list = []
res = minimize(fun, p0, args=(sim_grad, iter_list), method='bfgs', jac=True, tol=1e-6)

print(res.message)
print('Optimal Value: %.12f' % res.fun)

sim.reset()
res_pr = dict(zip(params_name, res.x))
sim.apply_circuit(ansatz.apply_value(res_pr))
psi_res = sim.get_qs()
sim.reset()
sim.apply_circuit(ansatz.apply_value(gate_pr))
psi = sim.get_qs()
rho = np.outer(psi_res.conj(), psi_res)
rho = reduced_density_matrix(rho, [3, 4, 5])
print('psi norm: %.20f' % norm(psi - psi_res, 2))
print('psi fidelity: %.20f' % fidelity(psi, psi_res))
print('rdm[3] norm: %.20f' % norm(rdm[3] - rho, 2))
print('rdm[3] fidelity: %.20f' % fidelity(rdm[3], rho))

end = time.perf_counter()
print('Runtime: %f' % (end - start))