import re
import time
import numpy as np
from utils import *
from h5py import File
from scipy.io import loadmat
from numpy.linalg import norm
from scipy.sparse import csr_matrix
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
            global start
            t = time.perf_counter() - start
            print('Loss: %.15f, Fidelity: %.15f, %d, %.4f' % (f, 1 - f, i, t))
    return f, g


rdm3 = loadmat('./mat/322_d2_num1_model957_RDM.mat')['RDM_3']
g = File('./mat/322_d2_num1_model957_RDM3_gates_L10_N9.mat')
position = g['RDM_site'][:] - 1  # subtract index of matlab to python
l = list(g.keys())  # list of HDF5 gates file keys
d = int(g['d'][0])  # dimension of qudit state
g_name = [x for x in l if 'gates' in x]  # list of Q_gates_?
key = lambda x: [int(s) if s.isdigit() else s for s in re.split('(\d+)', x)]
g_name = sorted(g_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
k = g[g_name[0]].shape[0]  # number of gates in one layer
gates = [[g[g[i][j]][:].view('complex').T for j in range(k)] for i in g_name]
g.close()

pr = {}
circ = Circuit()
ansatz = Circuit()
for i in range(len(g_name)):
    for j in range(k):
        name = f'G{j + 1}_L{i + 1}'
        mat = gates[i][j]
        gate_u = UnivMathGate(name, mat).on([k - j - 1, k - j])
        gate_d, para = two_qubit_decompose(gate_u)
        pr.update(para)
        circ += gate_u
        ansatz += gate_d

nq = ansatz.n_qubits
p_name = ansatz.ansatz_params_name
pr = {i: pr[i] for i in p_name}
p_num = len(p_name)
g_num = sum(1 for _ in ansatz)
print('Number of qubits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

sim = Simulator('mqvector', nq)
sim.apply_circuit(circ)
psi = sim.get_qs()

rho = np.outer(psi, psi.conj())
Ham = Hamiltonian(csr_matrix(rho))
print('Hamiltonian Dimension:', rho.shape)

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
    method = 'TNC'
    print(f'Simulator: mqvector, Method: {method}')
sim_grad = sim.get_expectation_with_grad(Ham, ansatz)

start = time.perf_counter()
# p0 = np.array(list(pr.values()))
p0 = np.random.uniform(-1, 1, p_num)
res = minimize(fun, p0, args=(sim_grad, []), method=method, jac=True, options={'gtol': 1e-8, 'maxiter': 10000})
print(res.message)
print('Optimal: %.20f, %s' % (res.fun, res.fun))

sim.reset()
pr_res = dict(zip(p_name, res.x))
sim.apply_circuit(ansatz.apply_value(pr_res))
psi_res = sim.get_qs()
rho_res_rdm = reduced_density_matrix(psi_res, d, position)

print('psi & psi_res norm L2:  %.20f' % norm(psi - psi_res, 2))
print('psi & psi_res fidelity: %.20f' % fidelity(psi, psi_res))
print('rdm3 & rho_res norm L2:  %.20f' % norm(rdm3 - rho_res_rdm, 2))
print('rdm3 & rho_res fidelity: %.20f' % fidelity(rdm3, rho_res_rdm))

end = time.perf_counter()
print('Runtime: %f' % (end - start))