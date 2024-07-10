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


def running(model: int, num: int, repeat: Union[int, range, List[int]], layers: int = 2):

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
                info(f'model{model} D{D} iter{num} r{repeat} Loss: {loss:.15f} Fidelity: {1-loss:.15f} {i} {t:.2f}')
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
        if 0 < loss - minima1 < 1e-4:
            local_minima1.append(loss - minima1)
        if 0 < loss - minima2 < 1e-4:
            local_minima2.append(loss - minima2)
        if len(local_minima1) >= 100:
            info(f'model{model} D{D} iter{num} r{repeat}: reach local minima1, restart optimization')
            raise StopAsyncIteration
        if len(local_minima2) >= 100:
            info(f'model{model} D{D} iter{num} r{repeat}: reach local minima2, restart optimization')
            raise StopAsyncIteration
        if loss < tol:
            raise StopIteration

    path = f'./data_232/from_classical_to_violation_iter30'  # path of folder
    name = f'{path}/232_d3_D9_model{model}_RDM2_iter{num}_target_state_vector.mat'
    s = File(name)
    state = s['target_state_vec'][:].view('complex')
    D = s['D'][0]  # bond dimension
    N = s['N'][0]  # number of qudits
    s.close()

    log = f'./data_232/Logs/from_classical_to_violation_iter30_L{layers}.log'
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

    psi = symmetric_encoding(state, N, is_csr=True)  # encode qutrit state to qubit
    rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
    Ham = Hamiltonian(rho)  # set target state as Hamiltonian

    sim = Simulator('mqvector', nq)
    if num == 1 and repeat == 1:
        info(f'Number of qubits: {nq}')
        info(f'Number of gates: {g_num}')
        info(f'Number of params: {p_num}')
        info(f'Depth of circuit: {depth}')
        info(f'Simulator: mqvector, Method: BFGS')
    solver_options = {'gtol': 1e-15, 'maxiter': 500}
    sim_grad = sim.get_expectation_with_grad(Ham, ansatz)

    start = time.perf_counter()
    while True:
        try:
            local_minima1, local_minima2 = [], []
            init_params = np.random.uniform(-np.pi, np.pi, p_num)
            res = minimize(optimization, init_params, (sim_grad, []), method='BFGS', jac=True, callback=callback, options=solver_options)
            break
        except StopIteration:
            break  # reach loss tolerance
        except StopAsyncIteration:
            continue  # reach local minima
    end = time.perf_counter()

    fidelity = 1 - res.fun
    iter_str = f'iter{num}'
    seconds = round(end - start, 2)
    time_dict[iter_str].append(seconds)
    eval_dict[iter_str].append(res.nfev)
    fidelity_dict[iter_str].append(fidelity)

    sim.reset()  # reset simulator to zero state
    pr_res = dict(zip(p_name, res.x))
    sim.apply_circuit(ansatz.apply_value(pr_res))
    psi_res = sim.get_qs()
    mat_name = f'{path}/fidelity_state_pr_model{model}_L{layers}.mat'
    save = {f'{iter_str}_r{repeat}_fidelity': fidelity, f'{iter_str}_r{repeat}_state': psi_res, f'{iter_str}_r{repeat}_pr': res.x}
    updatemat(mat_name, save)

    info(f'Optimal: {res.fun:.8e} Fidelity: {fidelity:.15f} {res.message}')
    info(f'Eval List: {eval_dict[iter_str]}')
    info(f'Time List: {time_dict[iter_str]}')
    info(f'Fidelity List: {fidelity_dict[iter_str]}')


def iter_dict(num_iter):
    return {f'iter{i}': [] for i in range(1, num_iter + 1)}


layers = int(input('Number of layers: '))
for model in [1216, 1705]:
    num_iter, repetitions = 30, 20
    eval_dict = iter_dict(num_iter)
    time_dict = iter_dict(num_iter)
    fidelity_dict = iter_dict(num_iter)
    for repeat in range(1, repetitions + 1):
        for num in range(1, num_iter + 1):
            running(model, num, repeat, layers)
        info(f'model{model} D9 repeat{repeat} finish\n{eval_dict}\n{time_dict}\n{fidelity_dict}')
        print(f'model{model} D9 repeat{repeat} finish')