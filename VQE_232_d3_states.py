import re
import time
import numpy as np
from utils import *
from h5py import File
from scipy.io import loadmat
from scipy.optimize import minimize
from numpy.linalg import norm, matrix_rank
from logging import info, INFO, basicConfig
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator, get_supported_simulator


def fun(p0, sim_grad, loss_list=None):
    '''Optimize function of fidelity
    p0: initial parameters
    sim_grad: simulator forward with gradient
    loss_list: list of loss values for loss function
    '''
    f, g = sim_grad(p0)
    f = 1 - np.real(f)[0][0]
    g = -np.real(g)[0][0]
    if loss_list is not None:
        loss_list.append(f)
        i = len(loss_list)
        if i % 10 == 0:
            global start, num, layers
            t = time.perf_counter() - start
            print(f'num{num}, {model}, Layers: {layers}, ', end='')
            print(f'Loss: {f:.15f}, Fidelity: {1-f:.15f}, {i}, {t:.2f}')
            info(f'Loss: {f:.15f}, Fidelity: {1-f:.15f}, {i}, {t:.2f}')
    return f, g


def callback(xk):
    '''Callback when loss < tol
    xk: current parameter vector
    '''
    f, _ = sim_grad(xk)
    loss = 1 - np.real(f)[0][0]
    if loss < 1e-8: # tolerance
        raise StopIteration


# dict of folders
dict_folder = {
    1: 'type1_no_violation',
    2: 'type2_Q3_Q4_different_violation',
    3: 'type3_Q3_Q4_same_violation',
    4: 'type4_Q4_violation'
}
t = 1  # which type of folder
path = f'./data_232/{dict_folder[t]}'  # path of folder
dict_mat = dict_file(path)  # dict of mat files
num = input('File name: num')  # input num of file index
RDM_name = dict_mat[f'RDM_{num}']  # RDM mat file name
model = re.search('model\d+', RDM_name).group(0)  # model number
layers = int(input('Number of layers: '))  # input number of layers

log = f'./data_232/Logs/type{t}_num{num}_{model}_L{layers}.log'
basicConfig(filename=log, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)

rdm2 = loadmat(f'{path}/RDM/{RDM_name}')['RDM_2']
info(f'RDM2 Rank: {matrix_rank(rdm2)}')
s = File(f'{path}/target_state/{dict_mat[f"target_state_{num}"]}', 'r')
state = s['target_state_vec'][:].view('complex').conj()  # bra -> ket
s.close()

d = 3  # dimension of qudit state
k = 5  # number of gates in one layer
nq = (k + 1) * (d - 1)  # number of qubits
position = np.array([2, 3])  # position of rdm2
ansatz = Circuit()  # qutrit symmetric ansatz
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
info(f'Number of qubits: {nq}')
info(f'Number of params: {p_num}')
info(f'Number of gates: {g_num}')

psi = su2_encoding(state, k + 1, is_csr=True)  # encode qutrit target state to qubit
rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
Ham = Hamiltonian(rho)  # set target state as Hamiltonian
info(f'Hamiltonian Dimension: {rho.shape}')

rho_rdm = reduced_density_matrix(state, d, position)
info(f'rdm2 & rho norm L2:  {norm(rdm2 - rho_rdm, 2):.20f}')
info(f'rdm2 & rho fidelity: {fidelity(rdm2, rho_rdm):.20f}')

sim_list = set([i[0] for i in get_supported_simulator()])
if 'mqvector_gpu' in sim_list and nq >= 12:
    sim = Simulator('mqvector_gpu', nq)
    method = 'BFGS'
    info(f'Simulator: mqvector_gpu, Method: {method}')
else:
    sim = Simulator('mqvector', nq)
    method = 'BFGS'  # TNC CG
    info(f'Simulator: mqvector, Method: {method}')
sim_grad = sim.get_expectation_with_grad(Ham, ansatz)

start = time.perf_counter()
options = {'gtol': 1e-8, 'maxiter': 1e6}  # solver options
p0 = np.random.uniform(-np.pi, np.pi, p_num)  # initial parameters
res = minimize(fun, p0, args=(sim_grad, []), method=method, jac=True, callback=callback, options=options)
info(res.message)
info(f'Number of layers: {layers}')
info(f'Optimal: {res.fun:.20f}, {res.fun}')
print(f'Optimal: {res.fun:.20f}, {res.fun}')

sim.reset()  # reset simulator to zero state
pr_res = dict(zip(p_name, res.x))  # optimal result parameters
sim.apply_circuit(ansatz.apply_value(pr_res))  # apply result params to circuit
psi_res = sim.get_qs()  # get result pure state
psi_res = su2_decoding(psi_res, k + 1)  # decode qubit result state to qutrit
rho_res_rdm = reduced_density_matrix(psi_res, d, position)

info(f'state & psi_res norm L2:  {norm(state - psi_res, 2):.20f}')
info(f'state & psi_res fidelity: {fidelity(state, psi_res):.20f}')
info(f'rdm2 & rho_res norm L2:  {norm(rdm2 - rho_res_rdm, 2):.20f}')
info(f'rdm2 & rho_res fidelity: {fidelity(rdm2, rho_res_rdm):.20f}')

total = time.perf_counter() - start
info(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h, Iter: {res.nfev}')
print(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h, Iter: {res.nfev}')