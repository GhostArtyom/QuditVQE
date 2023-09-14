import re
import h5py
import time
import numpy as np
from utils import *
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian

start = time.perf_counter()
np.set_printoptions(linewidth=200)


def fun(p0, sim_grad, energy_list=None):
    f, g = sim_grad(p0)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    if energy_list is not None:
        energy_list.append(f)
        i = len(energy_list)
        if i % 10 == 0:
            energy_gap = abs(min_eigval - f)
            print('Energy Gap: %.12f, %d' % (energy_gap, i))
    return f, g


g = h5py.File('mat/322_d2_num1_model957_site3_gates_L39_N9_zhu(2).mat', 'r')
# position = g['position'][:] - 1  # subtract index of matlab to python
l = list(g.keys())  # list of HDF5 file keys
g_name = [x for x in l if 'gates' in x]  # list of Q_gates_?
key = lambda x: [int(s) if s.isdigit() else s for s in re.split('(\d+)', x)]
g_name = sorted(g_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
# print(position, g_name)
k = g[g_name[0]].shape[0]  # number of gates in one layer
gates = [[g[g[i][j]][:].view('complex').T for j in range(k)] for i in g_name]
g.close()

r = h5py.File('mat/322_d2_num1_model957_RDM_v7.3.mat', 'r')
l = list(r.keys())
rdm = [r[i][:].view('complex').T for i in l]
rdm.insert(0, [])
r.close()

pr = {}
circ_u = Circuit()
circ_d = Circuit()
for i in range(len(g_name)):
    for j in range(k):
        gate_name = 'G' + str(j + 1) + '_L' + str(i + 1)
        gate_mat = gates[i][j]
        if j == k - 1:
            gate_u = UnivMathGate(gate_name, gate_mat).on(k - j - 1)
            gate_d, para = one_qubit_decompose(gate_u)
            pr.update(para)
            circ_u += gate_u
            circ_d += gate_d
        else:
            gate_u = UnivMathGate(gate_name, gate_mat).on([k - j - 2, k - j - 1])
            gate_d, para = two_qubit_decompose(gate_u)
            pr.update(para)
            circ_u += gate_u
            circ_d += gate_d

ansatz = circ_d.as_ansatz()
params_name = ansatz.ansatz_params_name
params_size = len(params_name)
print('Number of params: %d' % params_size)

ham = np.kron(np.kron(rdm[3], rdm[3]), rdm[3])
print('Hamiltonian Dimension:', ham.shape)
Ham = Hamiltonian(csr_matrix(ham))

eigval, eigvec = np.linalg.eig(ham)
eigvec = np.transpose(eigvec)
min_eigval = np.min(eigval)
min_eigvec = eigvec[np.argmin(eigval)]
print('Minimum Eigenvalue:', min_eigval)

sim = Simulator('mqvector', ansatz.n_qubits)
sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
p0 = np.zeros(params_size)
f, g = sim_grad(p0)

fun(p0, sim_grad)
energy_list = []
res = minimize(fun, p0, args=(sim_grad, energy_list), method='bfgs', jac=True)

print('Optimized Energy: %.12f' % res.fun)
print('Energy Gap: %.12f' % abs(min_eigval - res.fun))

end = time.perf_counter()
print('Runtime: %f' % (end - start))
