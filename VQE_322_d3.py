import re
import time
import numpy as np
from utils import *
from h5py import File
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator, get_supported_simulator

start = time.perf_counter()
np.set_printoptions(linewidth=200)


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
            print('Optimal Gap: %.20f, %d, %.4f' % (f, i, t))
    return f, g


g = File('./mat/322_d3_num1_model957_RDM3_gates_L10_N7_variational.mat', 'r')
position = g['RDM_site'][:] - 1  # subtract index of matlab to python
l = list(g.keys())  # list of HDF5 gates file keys
d = int(g['d'][0])  # dimension of qudit state
g_name = [x for x in l if 'gates' in x]  # list of Q_gates_?
key = lambda x: [int(s) if s.isdigit() else s for s in re.split('(\d+)', x)]
g_name = sorted(g_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
# print(position, g_name)
k = g[g_name[0]].shape[0]  # number of gates in one layer
gates = [[g[g[i][j]][:].view('complex').T for j in range(k)] for i in g_name]
g.close()

r = File('./mat/322_d3_num1_model957_RDM_v7.3.mat', 'r')
l = list(r.keys())  # list of HDF5 rdm file keys
rdm = [r[i][:].view('complex').T for i in l]
rdm.insert(0, [])
r.close()

pr = {}
circ = Circuit()
ansatz = Circuit()
nq = (k + 1) * (d - 1)
c = np.eye(2**(2 * (d - 1))) - su2_encoding(np.eye(d**2), 2)
for i in range(len(g_name)):
    for j in range(k):
        name = f'G{j + 1}_L{i + 1}'
        mat = su2_encoding(gates[i][j], 2) + c
        obj = list(range(nq - (d - 1) * (j + 2), nq - (d - 1) * j))
        gate_u = UnivMathGate(name, mat).on(obj)
        # gate_d, para = two_qubit_decompose(gate_u)
        # pr.update(para)
        circ += gate_u
        # ansatz += gate_d

# ansatz =
p_name = ansatz.ansatz_params_name
# pr = {i: pr[i] for i in p_name}
p_num = len(p_name)
g_num = sum(1 for _ in ansatz)
print('Number of qubits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

sim = Simulator('mqvector', nq)
sim.apply_circuit(circ)
psi = sim.get_qs()

ham = np.outer(psi, psi.conj())
print('Hamiltonian Dimension:', ham.shape)
Ham = Hamiltonian(csr_matrix(ham))

psi = su2_decoding(psi, k + 1)
rho_rdm = reduced_density_matrix(psi, d, position)
print('rho norm: %.20f' % norm(rdm[3] - rho_rdm, 2))
print('rho fidelity: %.20f' % fidelity(rdm[3], rho_rdm))

sim_list = set([i[0] for i in get_supported_simulator()])
if 'mqvector_gpu' in sim_list and nq > 10:
    sim = Simulator('mqvector_gpu', nq)
    method = 'BFGS'
    print(f'Simulator: mqvector_gpu, Method: {method}')
else:
    sim = Simulator('mqvector', nq)
    method = 'TNC'
    print(f'Simulator: mqvector, Method: {method}')
sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
# p0 = np.array(list(pr.values()))
p0 = np.random.uniform(-1, 1, p_num)
fun(p0, sim_grad)
res = minimize(fun, p0, args=(sim_grad, []), method=method, jac=True, tol=1e-8)

print(res.message)
print('Optimal Value: %.20f' % res.fun)

sim.reset()
res_pr = dict(zip(p_name, res.x))
sim.apply_circuit(ansatz.apply_value(res_pr))
psi_res = sim.get_qs()
psi_res = su2_decoding(psi_res, k + 1)
rho_res_rdm = reduced_density_matrix(psi_res, d, position)

print('psi norm: %.20f' % norm(psi - psi_res, 2))
print('psi fidelity: %.20f' % fidelity(psi, psi_res))
print('rho norm: %.20f' % norm(rdm[3] - rho_res_rdm, 2))
print('rho fidelity: %.20f' % fidelity(rdm[3], rho_res_rdm))

end = time.perf_counter()
print('Runtime: %f' % (end - start))