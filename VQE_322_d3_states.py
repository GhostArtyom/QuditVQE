import re
import os
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
            print(f'num{num}, {model}, vec{vec}, L{layers}, ', end='')
            print(f'Loss: {f:.15f}, Fidelity: {1-f:.15f}, {i}, {t:.2f}')
            info(f'vec{vec}, Loss: {f:.15f}, Fidelity: {1-f:.15f}, {i}, {t:.2f}')
    return f, g


def callback(xk):
    '''Callback when loss < tol
    xk: current parameter vector
    '''
    minima = 0.25
    f, _ = sim_grad(xk)
    loss = 1 - np.real(f)[0][0]
    if 0 < loss - minima < 2e-3:
        local_minima.append(loss - minima)
    if len(local_minima) >= 20:
        info(f'vec{vec}: {local_minima}')
        info(f'Reach local minima, restart optimization')
        print(f'Reach local minima, restart optimization')
        raise StopAsyncIteration
    if loss < 1e-12:  # tolerance
        raise StopIteration


layers = 2  # number of layers
num = input('File name: num')  # input num of file index
sub = sorted(os.listdir('./data_322'))[int(num)]
path = f'./data_322/{sub}'  # path of subfolder
dict_mat = dict_file(path)  # dict of mat files
name = dict_mat[f'target_state_1']  # state file name
model = re.search('model\d+', name).group(0)  # model number

log = f'./data_322/Logs/num{num}_{model}_L{layers}.log'
basicConfig(filename=log, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)

s = File(f'{path}/target_state/{name}', 'r')
s_name = [x for x in s.keys() if 'state' in x]  # list of target_state_vec_?
key = lambda x: [int(y) if y.isdigit() else y for y in re.split('(\d+)', x)]
s_name = sorted(s_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
state = {i + 1: s[j][:].view('complex') for i, j in enumerate(s_name)}
vec_num = len(s_name)  # number of target_state_vec in mat file
uMPS_name = [i for i in dict_file(f'{path}/uMPS').values() if f'num{num}' in i]  # uMPS file name
uMPS_name = sorted(uMPS_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
energy_list = [loadmat(f'{path}/uMPS/{uMPS_name[i]}')['energy'][0][0] for i in range(vec_num)]
info(f'Energy list: {energy_list}')
s.close()

d = 3  # dimension of qudit state
k = 6  # number of gates in one layer
nq = (k + 1) * (d - 1)  # number of qubits
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

sim_list = set([i[0] for i in get_supported_simulator()])
if 'mqvector_gpu' in sim_list and nq > 14:
    sim = Simulator('mqvector_gpu', nq)
    method = 'BFGS'
    info(f'Simulator: mqvector_gpu, Method: {method}')
else:
    sim = Simulator('mqvector', nq)
    method = 'BFGS'  # TNC CG
    info(f'Simulator: mqvector, Method: {method}')

iter_list = []
fidelity_list = []
for vec in range(1, vec_num + 1):  # index start from 1
    psi = su2_encoding(state[vec], k + 1, is_csr=True)  # encode qutrit state to qubit
    rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
    Ham = Hamiltonian(rho)  # set target state as Hamiltonian

    start = time.perf_counter()
    sim.reset()  # reset simulator to zero state
    sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
    while True:
        try:
            local_minima = []
            options = {'gtol': 1e-12, 'maxiter': 500}  # solver options
            p0 = np.random.uniform(-np.pi, np.pi, p_num)  # initial parameters
            res = minimize(fun, p0, args=(sim_grad, []), method=method, jac=True, callback=callback, options=options)
            break
        except StopIteration:
            break
        except StopAsyncIteration:
            continue
    info(res.message)
    print(res.message)
    info(f'Optimal: {res.fun}, Fidelity: {1 - res.fun:.20f}')
    print(f'Optimal: {res.fun}, Fidelity: {1 - res.fun:.20f}')
    iter_list.append(res.nfev)
    fidelity_list.append(1 - res.fun)
    info(f'vec{vec}: {iter_list}')
    print(f'vec{vec}: {iter_list}')
    info(f'vec{vec}: {fidelity_list}')
    print(f'vec{vec}: {fidelity_list}')

    total = time.perf_counter() - start
    info(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h, Iter: {res.nfev}')
    print(f'Runtime: {total:.4f}s, {total/60:.4f}m, {total/3600:.4f}h, Iter: {res.nfev}')