import re
import os
import time
import numpy as np
from h5py import File
from typing import List, Union
from scipy.optimize import minimize
from logging import info, INFO, basicConfig
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator.utils import GradOpsWrapper
from mindquantum.simulator import Simulator, get_supported_simulator
from utils import updatemat, circuit_depth, symmetric_encoding, qutrit_symmetric_ansatz


def running(num: int, D: int, vec: Union[int, List[int]], repeat: Union[int, range, List[int]], layers: int = 2):

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
            fidelity_iter.append(1 - loss)
            loss_list.append(loss)
            i = len(loss_list)
            if i % 10 == 0:
                t = time.perf_counter() - start
                info(f'num{num} D{D} L{layers} vec{v} r{r} Loss: {loss:.15f} Fidelity: {1-loss:.15f} {i} {t:.2f}')
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
        if layers != 1:
            if len(local_minima1) >= 30:
                info(f'num{num} D{D} vec{v} r{r}: reach local minima1, restart optimization')
                raise StopAsyncIteration
            if len(local_minima2) >= 30:
                info(f'num{num} D{D} vec{v} r{r}: reach local minima2, restart optimization')
                raise StopAsyncIteration
        if loss < tol:
            raise StopIteration

    sub = [i for i in sorted(os.listdir('./data_322')) if f'num{num}' in i][0]
    path = f'./data_322/{sub}'  # path of subfolder
    log = f'./data_322/Logs/num1~5_violation_D5_L{layers}.log'
    basicConfig(filename=log, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)

    s = File(f'{path}/target_state/322_violation_d3_D{D}_{sub}_target_state_vector.mat')
    s_name = [x for x in s.keys() if 'state' in x]  # list of target_state_vec_?
    key = lambda x: [int(y) if y.isdigit() else y for y in re.split('(\d+)', x)]
    s_name = sorted(s_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
    state = {i + 1: s[j][:].view('complex') for i, j in enumerate(s_name)}
    vec_num = len(s_name)  # number of target_state_vec in mat file
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

    time_dict, eval_dict, fidelity_dict = {}, {}, {}
    vec = [vec] if isinstance(vec, int) else vec
    for v in vec:
        vec_str = f'vec{v}'
        time_dict[vec_str], eval_dict[vec_str], fidelity_dict[vec_str] = [], [], []
        repetitions = range(1, repeat + 1) if isinstance(repeat, int) else repeat

        psi = symmetric_encoding(state[v], k + 1, is_csr=True)  # encode qutrit state to qubit
        rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
        Ham = Hamiltonian(rho)  # set target state as Hamiltonian
        for r in repetitions:
            sim.reset()  # reset simulator to zero state
            sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
            solver_options = {'gtol': 1e-15, 'maxiter': 500}

            start = time.perf_counter()
            while True:
                try:
                    fidelity_iter, local_minima1, local_minima2 = [], [], []
                    init_params = np.random.uniform(-np.pi, np.pi, p_num)
                    res = minimize(optimization, init_params, (sim_grad, []), method, \
                                   jac=True, callback=callback, options=solver_options)
                    break
                except StopIteration:
                    break  # reach loss tolerance
                except StopAsyncIteration:
                    continue  # reach local minima
            end = time.perf_counter()

            fidelity = 1 - res.fun
            minute = round(((end - start) / 60), 2)
            time_dict[vec_str].append(minute)
            eval_dict[vec_str].append(res.nfev)
            fidelity_dict[vec_str].append(fidelity)

            mat_name = f'./data_322/fidelity_violation_L{layers}.mat'
            save = {f'num{num}_D{D}_vec{v}_r{r}_fidelity': fidelity_iter}
            updatemat(mat_name, save)

            info(f'Optimal: {res.fun}, Fidelity: {fidelity:.20f}, Repeat: {r}')
            info(f'{res.message}\n{eval_dict}\n{time_dict}\n{fidelity_dict}')
        print(f'num{num} D{D} vec{v} finish')
    info(f'num{num} D{D} finish')
    print(f'num{num} D{D} finish')


for num in range(1, 6):
    running(num, D=5, vec=40, repeat=1, layers=2)
