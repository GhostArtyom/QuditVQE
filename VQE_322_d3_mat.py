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
            print(f'num{num}, Layers: {layers}, ', end='')
            print(f'Loss: {f:.15f}, Fidelity: {1-f:.15f}, {i}, {t:.2f}')
    return f, g


def callback(xk):
    '''Callback when loss < tol
    xk: current parameter vector
    '''
    f, _ = sim_grad(xk)
    loss = 1 - np.real(f)[0][0]
    if loss < 1e-12:  # tolerance
        raise StopIteration


mat_states = {
    1: '322_d3_num1_model957_RDM3_target_state_vector',
    2: '322_d3_num1_model957_RDM3_target_state_vector_contextual_level3',
    3: '322_d3_num1_model957_RDM3_target_state_vector_contextual_level0_new'
}
mat_rdm = {
    1: '322_d3_num1_model957_RDM_new',
    2: '322_d3_num1_model957_RDM_contextual_level3',
    3: '322_d3_num1_model957_RDM_contextual_level0_new',
}

num = int(input('File name: num'))  # input num of file index
layers = int(input('Number of layers: '))  # input number of layers

rdm3 = loadmat(f'./mat/{mat_rdm[num]}.mat')['RDM_3']
print(f'RDM3 Rank: {matrix_rank(rdm3)}')
s = File(f'./mat/{mat_states[num]}.mat', 'r')
state = s['target_state_vec'][:].view('complex').conj()  # bra -> ket
s.close()

d = 3  # dimension of qudit state
k = 6  # number of gates in one layer
nq = (k + 1) * (d - 1)  # number of qubits
position = np.array([2, 3, 4])  # position of rdm3
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
print(f'Number of qubits: {nq}')
print(f'Number of params: {p_num}')
print(f'Number of gates: {g_num}')

psi = su2_encoding(state, k + 1, is_csr=True)  # encode qutrit state to qubit
rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
Ham = Hamiltonian(rho)  # set target state as Hamiltonian
print(f'Hamiltonian Dimension: {rho.shape}')

rho_rdm = reduced_density_matrix(state, d, position)
print(f'rdm3 & rho norm L2:  {norm(rdm3 - rho_rdm, 2):.20f}')
print(f'rdm3 & rho fidelity: {fidelity(rdm3, rho_rdm):.20f}')

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
options = {'gtol': 1e-12, 'maxiter': 500}  # solver options
p0 = np.random.uniform(-np.pi, np.pi, p_num)  # initial parameters
res = minimize(fun, p0, args=(sim_grad, []), method=method, jac=True, callback=callback, options=options)
print(res.message)
print(f'Number of layers: {layers}')
print(f'Optimal: {res.fun:.20f}, {res.fun}')

sim.reset()
pr_res = dict(zip(p_name, res.x))  # optimal result parameters
sim.apply_circuit(ansatz.apply_value(pr_res))  # apply result params to circuit
psi_res = sim.get_qs()  # get result pure state
psi_res = su2_decoding(psi_res, k + 1)  # decode qubit result state to qutrit
rho_res_rdm = reduced_density_matrix(psi_res, d, position)

print(f'state & psi_res norm L2:  {norm(state - psi_res, 2):.20f}')
print(f'state & psi_res fidelity: {fidelity(state, psi_res):.20f}')
print(f'rdm3 & rho_res norm L2:  {norm(rdm3 - rho_res_rdm, 2):.20f}')
print(f'rdm3 & rho_res fidelity: {fidelity(rdm3, rho_res_rdm):.20f}')

total = time.perf_counter() - start
print(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h, Iter: {res.nfev}')