import re
import os
import time
import numpy as np
from h5py import File
from typing import List
from scipy.optimize import minimize
from logging import info, INFO, basicConfig
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator.utils import GradOpsWrapper
from mindquantum.simulator import Simulator, get_supported_simulator
from utils import circuit_depth, su2_encoding, qutrit_symmetric_ansatz


def running(num: int, layers: int, maxiter: int, repetitions: int):

    def optimization(init_params: np.ndarray, sim_grad: GradOpsWrapper, loss_list: List[float] = None):
        '''Optimization function of fidelity.
        Args:
            init_params (np.ndarray): initial parameters.
            sim_grad (GradOpsWrapper): simulator forward with gradient.
            loss_list (List[float]): list of loss values for loss function.
        Returns:
            loss (float): loss value of optimization.
            grad (np.ndarray): gradients of parameters.
        '''
        f, g = sim_grad(init_params)
        loss = 1 - np.real(f)[0][0]
        grad = -np.real(g)[0][0]
        if loss_list is not None:
            loss_list.append(loss)
            i = len(loss_list)
            if i <= 50 and i % 10 == 0 or i > 50 and i % 50 == 0:
                t = time.perf_counter() - start
                info(f'num{num}, D{D}, Repeat: {r}, Loss: {loss:.15f}, Fidelity: {1-loss:.15f}, {i}, {t:.2f}')
        return loss, grad

    def callback(curr_params: np.ndarray, tol: float = 1e-15):
        '''Callback when reach local minima or loss < tol.
        Args:
            curr_params (np.ndarray): current parameters.
            tol (float): tolerance of loss function.
        '''
        f, _ = sim_grad(curr_params)
        loss = 1 - np.real(f)[0][0]
        minima1, minima2 = 0.5, 0.25
        if 0 < loss - minima1 < 2e-3:
            local_minima1.append(loss - minima1)
        if 0 < loss - minima2 < 2e-3:
            local_minima2.append(loss - minima2)
        if len(local_minima1) >= 30:
            info(f'num{num}, D{D}: reach local minima1, restart optimization')
            raise StopAsyncIteration
        if len(local_minima2) >= 30:
            info(f'num{num}, D{D}: reach local minima2, restart optimization')
            raise StopAsyncIteration
        if loss < tol:
            raise StopIteration

    path = f'./data_322/classical'
    sub = sorted(os.listdir(path))[num - 1]
    model = re.search('model\d+', sub).group(0)

    s = File(f'{path}/322_classical_d3_num{num}_{model}_target_state_vector.mat')
    state = s['target_state_vec'][:].view('complex')
    D = s['D'][0]  # bond dimension
    N = s['N'][0]  # number of qudits
    s.close()

    log = f'./data_322/Logs/num1~5_classical_L{layers}.log'
    basicConfig(filename=log, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)

    d = 3  # dimension of qudit state
    nq = N * (d - 1)  # number of qubits
    ansatz = Circuit()  # qutrit symmetric ansatz
    for i in range(layers):
        for j in range(N - 1):
            name = f'G{j + 1}_L{i + 1}'
            mat = np.eye(2**(2 * (d - 1)))
            obj = list(range(nq - (d - 1) * (j + 2), nq - (d - 1) * j))
            gate_u = UnivMathGate(name, mat).on(obj)
            ansatz += qutrit_symmetric_ansatz(gate_u)

    p_name = ansatz.ansatz_params_name
    p_num = len(p_name)
    g_num = sum(1 for _ in ansatz)
    depth = circuit_depth(ansatz)
    info(f'Number of qubits: {nq}')
    info(f'Number of gates: {g_num}')
    info(f'Number of params: {p_num}')
    info(f'Depth of circuit: {depth}')

    sim_list = set([i[0] for i in get_supported_simulator()])
    if 'mqvector_gpu' in sim_list and nq >= 14:
        sim = Simulator('mqvector_gpu', nq)
        method = 'BFGS'
        info(f'Simulator: mqvector_gpu, Method: {method}')
    else:
        sim = Simulator('mqvector', nq)
        method = 'BFGS'  # TNC CG
        info(f'Simulator: mqvector, Method: {method}')

    num_str = f'num{num}'
    eval_dict[num_str], time_dict[num_str], fidelity_dict[num_str] = [], [], []
    for r in range(1, repetitions + 1):
        psi = su2_encoding(state, N, is_csr=True)  # encode qutrit state to qubit
        rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
        Ham = Hamiltonian(rho)  # set target state as Hamiltonian

        start = time.perf_counter()
        sim.reset()  # reset simulator to zero state
        sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
        while True:
            try:
                local_minima1, local_minima2 = [], []
                solver_options = {'gtol': 1e-15, 'maxiter': maxiter}
                init_params = np.random.uniform(-np.pi, np.pi, p_num)
                res = minimize(optimization, init_params, (sim_grad, []), method, \
                                jac=True, callback=callback, options=solver_options)
                break
            except StopIteration:
                break  # reach loss tolerance
            except StopAsyncIteration:
                continue  # reach local minima
        fidelity = 1 - res.fun
        end = time.perf_counter()
        seconds = round(end - start, 2)
        time_dict[num_str].append(seconds)
        eval_dict[num_str].append(res.nfev)
        fidelity_dict[num_str].append(fidelity)

        info(f'Optimal: {res.fun}, Fidelity: {fidelity:.20f}')
        info(f'{res.message}\n{eval_dict}\n{time_dict}\n{fidelity_dict}')
    print(f'num{num} D={D} L={layers} finish\n{fidelity_dict}')
    info(f'num{num} D={D} L={layers} finish')


eval_dict, time_dict, fidelity_dict = {}, {}, {}
for num in range(1, 6):
    running(num, layers=2, maxiter=500, repetitions=10)