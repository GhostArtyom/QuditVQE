import re
import time
import numpy as np
from utils import *
from h5py import File
from scipy.io import loadmat
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator, get_supported_simulator


def fun(p0, sim_grad, args=None):
    f, g = sim_grad(p0)
    f = 1 - np.real(f)[0][0]
    g = -np.real(g)[0][0]
    if args is not None:
        args.append(f)
        i = len(args)
        if i % 10 == 0:
            global start, num, layers
            t = time.perf_counter() - start
            print('num%s, Layers: %d, ' % (num, layers), end='')
            print('Loss: %.15f, Fidelity: %.15f, %d, %.4f' % (f, 1 - f, i, t))
    return f, g


mat_gates = {
    '1a': '322_d3_num1_model957_RDM3_gates_L10_N7_r0.9_nsweep20',
    '1b': '322_d3_num1_model957_RDM3_gates_L10_N7_r0.9_contextual_level3',
    '1c': '322_d3_num1_model957_RDM3_gates_L10_N7_r0.9_contextual_level0',
    '2': '322_d3_num2_model394_RDM3_gates_L10_N7_r0.8',
    '4': '322_d3_num4_model123_RDM3_gates_L10_N7_r0.8',
    '5': '322_d3_num5_model523_RDM3_gates_L10_N7_r0.8',
    '7': '322_d3_num7_model164_RDM3_gates_L10_N9_r0.8',
    '8': '322_d3_num8_model138_RDM3_gates_L10_N9_r0.8',
    '9': '322_d3_num9_model36_RDM3_gates_L10_N9_r0.8',
    '10': '322_d3_num10_model317_RDM3_gates_L10_N9_r0.8'
}
mat_rdm = {
    '1a': '322_d3_num1_model957_RDM_new',
    '1b': '322_d3_num1_model957_RDM_contextual_level3',
    '1c': '322_d3_num1_model957_RDM_contextual_level0_new',
    '2': '322_d3_num2_model394_RDM',
    '3': '322_d3_num3_model371_RDM',
    '4': '322_d3_num4_model123_RDM',
    '5': '322_d3_num5_model523_RDM',
    '6': '322_d3_num6_model165_RDM',
    '7': '322_d3_num7_model164_RDM',
    '8': '322_d3_num8_model138_RDM',
    '9': '322_d3_num9_model36_RDM',
    '10': '322_d3_num10_model317_RDM'
}

num = input('File name: num')
rdm3 = loadmat(f'./mat/{mat_rdm[num]}.mat')['RDM_3']

g = File(f'./mat/{mat_gates[num]}.mat')
position = g['RDM_site'][:] - 1  # subtract index of matlab to python
l = list(g.keys())  # list of HDF5 gates file keys
d = int(g['d'][0])  # dimension of qudit state
f = g['fidelity'][0][0]  # fidelity of gates
print('gates fidelity: %.20f' % f)
g_name = [x for x in l if 'gates' in x]  # list of Q_gates_?
key = lambda x: [int(s) if s.isdigit() else s for s in re.split('(\d+)', x)]
g_name = sorted(g_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
k = g[g_name[0]].shape[0]  # number of gates in one layer
gates = [[g[g[i][j]][:].view('complex').T for j in range(k)] for i in g_name]
g.close()

circ = Circuit()
ansatz = Circuit()
nq = (k + 1) * (d - 1)
Id = symmetric_encoding(np.eye(d**2), 2)
p = np.eye(Id.shape[0]) - Id
for i in range(len(g_name)):
    for j in range(k):
        name = f'G{j + 1}_L{i + 1}'
        mat = symmetric_encoding(gates[i][j], 2) + p
        obj = list(range(nq - (d - 1) * (j + 2), nq - (d - 1) * j))
        circ += UnivMathGate(name, mat).on(obj)

layers = int(input('Number of layers: '))
for i in range(layers):
    for j in range(k):
        name = f'G{j + 1}_L{i + 1}'
        mat = symmetric_encoding(gates[i][j], 2) + p
        obj = list(range(nq - (d - 1) * (j + 2), nq - (d - 1) * j))
        gate_u = UnivMathGate(name, mat).on(obj)
        ansatz += qutrit_symmetric_ansatz(gate_u)

p_name = ansatz.ansatz_params_name
p_num = len(p_name)
g_num = sum(1 for _ in ansatz)
print('Number of qubits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

sim = Simulator('mqvector', nq)
sim.apply_circuit(circ)
psi = sim.get_qs()
csc = csc_matrix(psi)
rho = csc.T.dot(csc.conj())
Ham = Hamiltonian(rho)
print('Hamiltonian Dimension:', rho.shape)

psi = symmetric_decoding(psi, k + 1)
rho_rdm = reduced_density_matrix(psi, d, position)
print('rdm3 & rho norm L2:  %.20f' % norm(rdm3 - rho_rdm, 2))
print('rdm3 & rho fidelity: %.20f' % fidelity(rdm3, rho_rdm))

sim_list = set([i[0] for i in get_supported_simulator()])
if 'mqvector_gpu' in sim_list and nq > 12:
    sim = Simulator('mqvector_gpu', nq)
    method = 'BFGS'
    print(f'Simulator: mqvector_gpu, Method: {method}')
else:
    sim = Simulator('mqvector', nq)
    method = 'BFGS'  # TNC CG
    print(f'Simulator: mqvector, Method: {method}')
sim_grad = sim.get_expectation_with_grad(Ham, ansatz)

start = time.perf_counter()
p0 = np.random.uniform(-np.pi, np.pi, p_num)
res = minimize(fun, p0, args=(sim_grad, []), method=method, jac=True, options={'gtol': 1e-8, 'maxiter': 10000})
print(res.message)
print('Optimal: %.20f, %s' % (res.fun, res.fun))
print(f'Number of layers: {layers}')

sim.reset()
pr_res = dict(zip(p_name, res.x))
sim.apply_circuit(ansatz.apply_value(pr_res))
psi_res = sim.get_qs()
psi_res = symmetric_decoding(psi_res, k + 1)
rho_res_rdm = reduced_density_matrix(psi_res, d, position)

print('psi & psi_res norm L2:  %.20f' % norm(psi - psi_res, 2))
print('psi & psi_res fidelity: %.20f' % fidelity(psi, psi_res))
print('rdm3 & rho_res norm L2:  %.20f' % norm(rdm3 - rho_res_rdm, 2))
print('rdm3 & rho_res fidelity: %.20f' % fidelity(rdm3, rho_res_rdm))

total = time.perf_counter() - start
print(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h')
