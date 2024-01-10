import re
import time
import numpy as np
from utils import *
from h5py import File
from scipy.io import loadmat
from scipy.optimize import minimize
from numpy.linalg import norm, matrix_rank
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
            print('num%s, %s, Layers: %d, ' % (num, model, layers), end='')
            print('Loss: %.15f, Fidelity: %.15f, %d, %.4f' % (f, 1 - f, i, t))
    return f, g


folder_dict = {
    1: 'type1_no_violation',
    2: 'type2_Q3_Q4_different_violation',
    3: 'type3_Q3_Q4_same_violation',
    4: 'type4_Q4_violation'
}
path = f'./data_232/{folder_dict[1]}'
mat_dict = file_dict(path)
num = input('File name: num')
RDM_name = mat_dict[f'RDM_{num}']
model = re.search('model\d+', RDM_name).group(0)

rdm2 = loadmat(f'{path}/RDM/{RDM_name}')['RDM_2']
print('RDM2 Rank:', matrix_rank(rdm2))
s = File(f'{path}/target_state/{mat_dict[f"target_state_{num}"]}', 'r')
state = s['target_state_vec'][:].view('complex').conj()  # bra -> ket
s.close()

d = 3  # dimension of qudit state
k = 5  # number of gates in one layer
position = np.array([2, 3])  # position of rdm2

layers = int(input('Number of layers: '))
nq = (k + 1) * (d - 1)
ansatz = Circuit()
for i in range(layers):
    for j in range(k):
        name = f'G{j + 1}_L{i + 1}'
        mat = np.eye(2**(2 * (d - 1)))
        obj = list(range(nq - (d - 1) * (j + 2), nq - (d - 1) * j))
        gate_u = UnivMathGate(name, mat).on(obj)
        ansatz += qutrit_symmetric_ansatz(gate_u)

p_name = ansatz.ansatz_params_name
p_num = len(p_name)
g_num = sum(1 for _ in ansatz)
print('Number of qubits: %d' % nq)
print('Number of params: %d' % p_num)
print('Number of gates: %d' % g_num)

psi = su2_encoding(state, k + 1, is_csr=True)
rho = psi.dot(psi.conj().T)
Ham = Hamiltonian(rho)
print('Hamiltonian Dimension:', rho.shape)

rho_rdm = reduced_density_matrix(state, d, position)
print('rdm2 & rho norm L2:  %.20f' % norm(rdm2 - rho_rdm, 2))
print('rdm2 & rho fidelity: %.20f' % fidelity(rdm2, rho_rdm))

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
psi_res = su2_decoding(psi_res, k + 1)
rho_res_rdm = reduced_density_matrix(psi_res, d, position)

print('state & psi_res norm L2:  %.20f' % norm(state - psi_res, 2))
print('state & psi_res fidelity: %.20f' % fidelity(state, psi_res))
print('rdm2 & rho_res norm L2:  %.20f' % norm(rdm2 - rho_res_rdm, 2))
print('rdm2 & rho_res fidelity: %.20f' % fidelity(rdm2, rho_res_rdm))

total = time.perf_counter() - start
print(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h')