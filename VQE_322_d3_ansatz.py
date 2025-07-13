import re
import os
import time
import numpy as np
from h5py import File
from logging import info
from logger import Logger
from scipy.io import loadmat
from functools import reduce
from typing import List, Union
from scipy.optimize import minimize
from mindquantum.core.circuit import Circuit
from scipy.sparse import eye, kron, csr_matrix
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator.utils import GradOpsWrapper
from mindquantum.core.gates import X, Z, H, RY, RZ, UnivMathGate
from mindquantum.simulator import Simulator, get_supported_simulator
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz, SGAnsatz
from utils import fidelity, updatemat, circuit_depth, symmetric_encoding, qutrit_symmetric_ansatz, CDTYPE


def running(num: int, D: int, vec: Union[int, List[int]], repeat: Union[int, range, List[int]], layers: int):

    def optimization(params: np.ndarray, sim_grad: GradOpsWrapper, loss_list: List[float] = None):
        '''Optimization function of fidelity.
        Args:
            params (np.ndarray): circuit parameters.
            sim_grad (GradOpsWrapper): simulator forward with gradient.
            loss_list (List[float]): list of loss values for loss function.
        Returns:
            loss (float): loss value of optimization.
            grad (np.ndarray): gradients of parameters.
        '''
        f, g = sim_grad(params)
        loss = np.real(f)[0][0]
        grad = np.real(g)[0][0]
        if loss_list is not None:
            loss_list.append(loss)
            i = len(loss_list)
            if is_penalty and i % 10 == 0:
                sim_.reset()
                pr = dict(zip(p_name, params))
                sim_.apply_circuit(ansatz.apply_value(pr))
                penalty_ = p_coeff * fidelity(sim_.get_qs(), P)
                t = time.perf_counter() - start
                info(
                    f'num{num} D{D} vec{v} r{repeat} L{layers}, Loss: {loss:.12f}, Fideltiy: {-loss+penalty_:.12f}, {i}/{maxiter}, {t:.2f}, Penalty: {penalty_:.8f}'
                )
            else:
                t = time.perf_counter() - start
                info(f'num{num} D{D} vec{v} r{repeat} L{layers}, Loss: {loss:.12f}, Fideltiy: {-loss:.12f}, {i}/{maxiter}, {t:.2f}')
            if i >= 100 and loss >= 0:
                info(f'num{num} D{D} vec{v} r{r}: poor params initialization, restart optimization')
                raise StopAsyncIteration
        return loss, grad

    def callback(params: np.ndarray, tol: float = 1e-8):
        '''Callback when reach local minima or loss < tol.
        Args:
            params (np.ndarray): circuit parameters.
            tol (float): tolerance of loss function.
        '''
        f, _ = sim_grad(params)
        loss = np.real(f)[0][0]
        minimal1, minimal2, minimal3 = -0.25, -0.5, -0.75
        if 0 < loss - minimal1 < 1e-3:
            local_minima1.append(loss)
        if 0 < loss - minimal2 < 1e-3:
            local_minima2.append(loss)
        if 0 < loss - minimal3 < 1e-3:
            local_minima3.append(loss)
        if len(local_minima1) >= 50:
            info(f'num{num} D{D} vec{v} r{r}: reach local minima1 {-minimal1:.2f}, restart optimization')
            raise StopAsyncIteration
        if len(local_minima2) >= 50:
            info(f'num{num} D{D} vec{v} r{r}: reach local minima2 {-minimal2:.2f}, restart optimization')
            raise StopAsyncIteration
        if len(local_minima3) >= 50:
            info(f'num{num} D{D} vec{v} r{r}: reach local minima3 {-minimal3:.2f}, restart optimization')
            raise StopAsyncIteration
        if 1 + loss < tol:
            raise StopIteration

    def penalty(n_qudits: int) -> csr_matrix:
        p = csr_matrix(([1, -1, -1, 1], ([1, 1, 2, 2], [1, 2, 1, 2])), shape=(4, 4), dtype=CDTYPE) / 2
        P = csr_matrix((4**n_qudits, 4**n_qudits), dtype=CDTYPE)
        for i in range(n_qudits):
            p_list = [eye(4, dtype=CDTYPE)] * (n_qudits - 1)
            p_list.insert(i, p)
            P += reduce(kron, p_list)
        return P

    def N_block(pr_str: str, wires: List[int]) -> Circuit:
        circ = Circuit()
        circ += RZ(-np.pi / 2).on(wires[1])
        circ += X.on(wires[0], wires[1])
        circ += RZ({f'{pr_str}_z': -2}).on(wires[0])
        circ += RZ(np.pi / 2).on(wires[0])
        circ += RY({f'{pr_str}_x': 2}).on(wires[1])
        circ += RY(-np.pi / 2).on(wires[1])
        circ += X.on(wires[1], wires[0])
        circ += RY({f'{pr_str}_y': -2}).on(wires[1])
        circ += RY(np.pi / 2).on(wires[1])
        circ += X.on(wires[0], wires[1])
        circ += RZ(np.pi / 2).on(wires[0])
        return circ

    def spin_conserving_ansatz(n_qubits: int, layers: int) -> Circuit:
        ind = 0
        circ = Circuit()
        # for i in range(n_qubits):
        #     circ += X.on(i)
        # for i in range(0, n_qubits, 2):
        #     circ += H.on(i)
        #     circ += X.on(i + 1, i)
        for _ in range(layers):
            for i in range(0, n_qubits // 2, 2):
                circ += N_block(f'p{ind}', [i, i + 1])
                if i != n_qubits - i - 2:
                    circ += N_block(f'p{ind}', [n_qubits - i - 2, n_qubits - i - 1])
                ind += 1
            for i in range(1, n_qubits // 2, 2):
                circ += N_block(f'p{ind}', [i, i + 1])
                if i != n_qubits - i - 2:
                    circ += N_block(f'p{ind}', [n_qubits - i - 2, n_qubits - i - 1])
                ind += 1
        return circ

    def symmetry_preserving_ansatz(n_qubits: int, layers: int) -> Circuit:
        circ = Circuit()
        for i in range(layers):
            for j in range(n_qudits - 1):
                name = f'G{j + 1}_L{i + 1}'
                mat = np.eye(2**(2 * (d - 1)))
                obj = list(range(n_qubits - (d - 1) * (j + 2), n_qubits - (d - 1) * j))
                gate_u = UnivMathGate(name, mat).on(obj)
                circ += qutrit_symmetric_ansatz(gate_u)
        return circ

    sub = [i for i in sorted(os.listdir('./data_322')) if f'num{num}' in i][0]
    path = f'./data_322/{sub}'  # path of subfolder

    s = File(f'{path}/target_state/322_violation_d3_D{D}_{sub}_target_state_vector.mat')
    s_name = [x for x in s.keys() if 'state' in x]  # list of target_state_vec_?
    key = lambda x: [int(y) if y.isdigit() else y for y in re.split('(\d+)', x)]
    s_name = sorted(s_name, key=key)  # sort 1,10,11,...,2 into 1,2,...,10,11
    state = {i + 1: s[j][:].view('complex') for i, j in enumerate(s_name)}
    s.close()

    d = 3  # dimension of qudits
    n_qudits = 7  # number of qudits
    n_qubits = n_qudits * (d - 1)  # number of qubits
    if which_ansatz == 'HEA':
        entangle_mapping = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        ansatz = HardwareEfficientAnsatz(n_qubits, [RZ, RY, RZ], Z, entangle_mapping, depth=layers).circuit
    elif which_ansatz == 'SGA':
        k = int(np.ceil(np.log2(D) + 1))
        ansatz = SGAnsatz(n_qubits, k, layers).circuit
    elif which_ansatz == 'SCA':
        ansatz = spin_conserving_ansatz(n_qubits, layers)
    elif which_ansatz == 'SPA':
        ansatz = symmetry_preserving_ansatz(n_qubits, layers)

    p_name = ansatz.ansatz_params_name
    p_num = len(p_name)
    g_num = sum(1 for _ in ansatz)
    depth = circuit_depth(ansatz)
    info(f'Number of qubits: {n_qubits}')
    info(f'Number of gates: {g_num}')
    info(f'Number of params: {p_num}')
    info(f'Depth of circuit: {depth}')

    sim_list = set([i[0] for i in get_supported_simulator()])
    if 'mqvector_gpu' in sim_list and n_qubits >= 14:
        sim = Simulator('mqvector_gpu', n_qubits)
        sim_ = Simulator('mqvector_gpu', n_qubits)
        method = 'BFGS'
        info(f'Simulator: mqvector_gpu, Method: {method}')
    else:
        sim = Simulator('mqvector', n_qubits)
        method = 'BFGS'  # TNC CG
        info(f'Simulator: mqvector, Method: {method}')

    if is_penalty:
        P = penalty(n_qudits)  # penalty term of anti-symmetric state

    time_dict, eval_dict, fidelity_dict = {}, {}, {}
    vec = [vec] if isinstance(vec, int) else vec
    for v in vec:
        time_dict[f'vec{v}'], eval_dict[f'vec{v}'], fidelity_dict[f'vec{v}'] = [], [], []
        repetitions = range(1, repeat + 1) if isinstance(repeat, int) else repeat

        psi = symmetric_encoding(state[v], n_qudits, is_csr=True)  # encode qutrit state to qubit
        rho = psi.dot(psi.conj().T)  # rho & psi are both csr_matrix
        if is_penalty:
            Ham = Hamiltonian(-rho + p_coeff * P)  # plus penalty term
        else:
            Ham = Hamiltonian(-rho)  # set target state as Hamiltonian
        for r in repetitions:
            D_vec_r = f'D{D}_vec{v}_r{r}'
            mat_name = f'{path}/{which_ansatz}_num{num}_L{layers}.mat'
            fidelity_res = loadmat(mat_name)[f'{D_vec_r}_fidelity'].item()
            if fidelity_res >= 0.9999:
                info(f'Fidelity: {fidelity_res} >= 0.9999, no need to rerun')
                break

            sim.reset()  # reset simulator to zero state
            sim_grad = sim.get_expectation_with_grad(Ham, ansatz)
            solver_options = {'gtol': 1e-15, 'maxiter': maxiter}

            start = time.perf_counter()
            while True:
                try:
                    local_minima1, local_minima2, local_minima3 = [], [], []
                    init_params = np.random.uniform(-np.pi, np.pi, p_num)
                    res = minimize(optimization, init_params, (sim_grad, []), method, jac=True, callback=callback, options=solver_options)
                    break  # callback=callback,
                except StopIteration:
                    break  # reach loss tolerance
                except StopAsyncIteration:
                    continue  # reach local minima
            end = time.perf_counter()

            sim.reset()  # reset simulator to zero state
            pr_res = dict(zip(p_name, res.x))
            sim.apply_circuit(ansatz.apply_value(pr_res))
            psi_res = sim.get_qs()

            fidelity_res = -res.fun
            minute = round(((end - start) / 60), 2)
            time_dict[f'vec{v}'].append(minute)
            eval_dict[f'vec{v}'].append(res.nfev)
            fidelity_dict[f'vec{v}'].append(fidelity_res)

            save = {f'{D_vec_r}_fidelity': fidelity_res, f'{D_vec_r}_state': psi_res, f'{D_vec_r}_pr': res.x}
            updatemat(mat_name, save)

            info(f'{res.message}\n{eval_dict}\n{time_dict}\n{fidelity_dict}')
        info(f'num{num} D{D} vec{v} finished')
    info(f'num{num} D{D} vec{vec} finished')


which_ansatz = 'SGA'
p_coeff, is_penalty = 2, False
D, repeat, maxiter = 5, 1, 500
for layers in [15]:  #, 20
    for num in [5]:  #1, 2, 3, 4, 5
        sub = [i for i in sorted(os.listdir('./data_322')) if f'num{num}' in i][0]
        log = f'./data_322/Logs/{sub}_{which_ansatz}_D{D}_L{layers}.log'
        logger = Logger(log)
        logger.add_handler()
        vec = [20, 30]
        # vec = [1, 10, 20, 30, 40]
        # vec = [5, 15, 25, 35, 40]
        running(num, D, vec, repeat, layers)
        logger.remove_handler()
