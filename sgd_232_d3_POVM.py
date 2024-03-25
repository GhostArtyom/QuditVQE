import os
import copy
import time
import random
import numpy as np
import cvxpy as cp
from sympy import Symbol
from sqlalchemy import true
from numpy.linalg import norm
from scipy.linalg import expm
from multiprocessing import Pool
import evoMPS.tdvp_uniform as mps
from sympy.utilities import lambdify
from logging import info, INFO, basicConfig
'''
python == 3.8.2 or 3.9.13
numpy == 1.19.2 or 1.19.3
scipy == 1.5.2 or 1.5.4
sqlalchemy == 2.0.29
Mosek == 10.1.28
evoMPS == 2.1.0
cvxpy == 1.4.2
sympy == 1.12
'''


def t2observable(t, Diag_list):
    D1 = Diag_list[0]
    D2 = Diag_list[1]
    D3 = Diag_list[2]

    H0 = np.array([[0, t[0], t[1]], [-t[0], 0, t[2]], [-t[1], -t[2], 0]])

    H1 = np.array([[0, t[3], t[4]], [-t[3], 0, t[5]], [-t[4], -t[5], 0]])

    H2 = np.array([[0, t[6], t[7]], [-t[6], 0, t[8]], [-t[7], -t[8], 0]])

    U0 = expm(H0)
    U1 = expm(H1)
    U2 = expm(H2)
    A0 = U0.dot(D1).dot(U0.conj().T)
    A1 = U1.dot(D2).dot(U1.conj().T)
    A2 = U2.dot(D3).dot(U2.conj().T)
    return A0, A1, A2


def obs2POVM(d, A0, A1, A2):
    Id = np.eye(d)
    # POVM for A0
    M00 = cp.Variable((d, d), hermitian=True)
    M01 = cp.Variable((d, d), hermitian=True)
    POVM0 = M00 - M01
    #POVM for A1
    M10 = cp.Variable((d, d), hermitian=True)
    M11 = cp.Variable((d, d), hermitian=True)
    POVM1 = M10 - M11
    #POVM for A2
    M20 = cp.Variable((d, d), hermitian=True)
    M21 = cp.Variable((d, d), hermitian=True)
    POVM2 = M20 - M21

    constraints = [M00 >> 0, M01 >> 0, M00 + M01 == Id, M10 >> 0, M11 >> 0, M10 + M11 == Id, M20 >> 0, M21 >> 0, M20 + M21 == Id]
    objective = cp.Minimize(cp.norm(A0 - POVM0) + cp.norm(A1 - POVM1) + cp.norm(A2 - POVM2))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)
    P0 = POVM0.value
    P1 = POVM1.value
    P2 = POVM2.value
    t_POVM = []
    t_POVM.extend(P0.flatten())
    t_POVM.extend(P1.flatten())
    t_POVM.extend(P2.flatten())
    # print("difference of A0 and P0 = ", norm(A0-P0))
    # print("difference of A1 and P1 = ", norm(A1-P1))
    # print("difference of A2 and P2 = ", norm(A2-P2))
    return t_POVM, P0, P1, P2


def hamiltonian(d, coef, A0, A1, A2):
    Id = np.eye(d)
    h0 = -coef[0] * np.kron(A0, Id)
    h1 = -coef[1] * np.kron(A1, Id)
    h2 = -coef[2] * np.kron(A2, Id)
    h3 = -coef[3] * np.kron(A0, A0)
    h4 = -coef[4] * np.kron(A0, A1)
    h5 = -coef[5] * np.kron(A0, A2)
    h6 = -coef[6] * np.kron(A1, A0)
    h7 = -coef[7] * np.kron(A1, A1)
    h8 = -coef[8] * np.kron(A1, A2)
    h9 = -coef[9] * np.kron(A2, A0)
    h10 = -coef[10] * np.kron(A2, A1)
    h11 = -coef[11] * np.kron(A2, A2)
    h_matrix = h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10 + h11
    ham = np.reshape(h_matrix, (d, d, d, d))
    return ham


def A_lfp_rfp(D, d, coef, evo_step, evo_threshold, evo_iter_max, A0, A1, A2):
    ham = hamiltonian(d, coef, A0, A1, A2)
    sim = mps.EvoMPS_TDVP_Uniform(D, d, ham)
    for i in range(evo_iter_max):
        sim.update()
        energy = sim.h_expect.real
        sim.take_step(evo_step)
        eta = sim.eta.real
        # print(i, energy, eta)
        if (i + 1) % 300 == 0 and eta > 0.1:
            evo_step = max(evo_step * 0.85, 0.0002)
        if (i + 1) % 500 == 0 and eta > 0.01:
            evo_step = max(evo_step * 0.85, 0.0001)
        if eta < evo_threshold or (i + 1) == evo_iter_max:
            return sim.A[0], sim.l[0], sim.r[0], eta, energy


def cal_gradient(d, t_value, coef, A, lfp, rfp):
    t = [Symbol('t' + str(i), complex=true) for i in range(3 * d**2)]
    obs = []
    for i in range(3):
        ele = t[i * d**2:(i + 1) * d**2]
        mat = np.reshape(ele, (d, d))
        obs.append(mat)
    A0 = obs[0]
    A1 = obs[1]
    A2 = obs[2]
    Id = np.eye(d)
    h0 = -coef[0] * np.kron(A0, Id)
    h1 = -coef[1] * np.kron(A1, Id)
    h2 = -coef[2] * np.kron(A2, Id)
    h3 = -coef[3] * np.kron(A0, A0)
    h4 = -coef[4] * np.kron(A0, A1)
    h5 = -coef[5] * np.kron(A0, A2)
    h6 = -coef[6] * np.kron(A1, A0)
    h7 = -coef[7] * np.kron(A1, A1)
    h8 = -coef[8] * np.kron(A1, A2)
    h9 = -coef[9] * np.kron(A2, A0)
    h10 = -coef[10] * np.kron(A2, A1)
    h11 = -coef[11] * np.kron(A2, A2)
    h_matrix = h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10 + h11
    ham = np.reshape(h_matrix, (d, d, d, d))
    row1, row2, col1, col2 = np.nonzero(ham)
    e = 0 + 0j
    for i in range(len(row1)):
        k, s, u, v = row1[i], row2[i], col1[i], col2[i]
        h_ele = ham[k, s, u, v]
        m1 = A[u].dot(A[v])
        m2 = (A[k].dot(A[s])).conj().T
        mat1 = lfp.dot(m1)
        mat2 = rfp.dot(m2)
        e += h_ele * np.trace(mat1.dot(mat2))
    fun_gradient = [e.diff(t) for t in t]
    g = lambdify([t], fun_gradient)
    grad = g(t_value)
    grad = np.array(np.real(grad))
    return grad


def run_one_model(model, coef, local, D, d, t_type, n_try, Diag_list):
    t = initial_t(9, t_type)

    # print("\n model = ", model)
    # print("\n coef = ", coef)
    # print("\n local = ", local)

    log = os.path.join(os.getcwd(), f'sgd_232/sgd_232_d{d}_POVM_model{model}_{t_type}.log')
    basicConfig(filename=log, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)

    i = 0
    gd_iter_max = 80
    gd_threshold = 1.0e-2
    evo_step = 0.012
    evo_threshold = 1.0e-3
    evo_iter_max = 50000
    momentum = 0
    # print("initial t = \n", t)
    while i <= gd_iter_max:
        i += 1
        # print("\n iteration {0}".format(i))
        time_iter_start = time.perf_counter()
        learning_rate = max(0.12 * 0.9**(i - 1), 1.0e-4)
        '''
        covert parameters "t" to observables "obs" and project into POVM "A0, A1, A2"
        '''
        A0, A1, A2 = t2observable(t, Diag_list)
        t, A0, A1, A2 = obs2POVM(d, A0, A1, A2)
        '''
        excute once TDVP method to obtain d*D*D tensor "A" and D*D left/right fixed point "lfp"/"rfp"
        '''
        attempts = 0
        success = False
        while attempts < 3 and not success:
            try:
                A, lfp, rfp, eta, energy = A_lfp_rfp(D, d, coef, evo_step, evo_threshold, evo_iter_max, A0, A1, A2)
                success = True
            except:
                attempts += 1
                evo_step = max(evo_step * 0.65, 0.001)
                evo_iter_max += 2000
                if attempts == 3:
                    break
        '''
        excute once gradient descent method to update parameters "t"
        '''
        time_grad_start = time.perf_counter()
        grad = cal_gradient(d, t, coef, A, lfp, rfp)
        time_grad_end = time.perf_counter()
        time_grad = time_grad_start - time_grad_end
        # print("once gradient time cost = ", time_grad)
        t_old = copy.deepcopy(t)
        t = t - learning_rate * grad + momentum * np.array(t)
        # print("gradident =", grad)
        grad_norm = norm(grad)
        time_iter_end = time.perf_counter()
        time_iter = time_iter_end - time_iter_start
        # print("iteration time cost = ", time_iter)
        # print("\n", [int(model), i, int(local), energy, grad_norm, eta, learning_rate])
        # print("\n", t_old)

        e_gap = energy - local
        # if t_type == "complex":
        #     file_e_name = "POVM_complex_232_" + "d{0}_model{1}_sgd_energy.txt".format(d, model)
        #     file_t_name = "POVM_complex_232_" + "d{0}_model{1}_sgd_observables.txt".format(d, model)
        # elif t_type == "real":
        #     file_e_name = "POVM_real_232_" + "d{0}_model{1}_sgd_energy.txt".format(d, model)
        #     file_t_name = "POVM_real_232_" + "d{0}_model{1}_sgd_observables.txt".format(d, model)

        # f_e_iterative = open(os.path.join(os.getcwd(), file_e_name), mode="a")
        # f_e_iterative.write("{0}\n".format(str([n_try, i, d, D, int(local), energy, grad_norm, eta, learning_rate])))

        # f_t_iterative = open(os.path.join(os.getcwd(), file_t_name), mode="a")
        # f_t_iterative.write("{0}\n".format(str([n_try, i, t_old])))

        # if grad_norm < gd_threshold or i == gd_iter_max or e_gap > 0 and e_gap < 1.0e-3:
        if e_gap < 0:
            info(f'n_try: {n_try}, i: {i}, D{D}, local: {local}, energy: {energy}, grad_norm: {grad_norm}, eta: {eta}, learning_rate: {learning_rate}, t_old: {t_old}')
            break


def initial_t(num, t_type):
    t = []
    for i in range(num):
        if t_type == "complex":
            t.append(random.uniform(-np.pi, np.pi) + 1j * random.uniform(-np.pi, np.pi))
        elif t_type == "real":
            t.append(random.uniform(-np.pi, np.pi))
    return t


def running(n_try, i_model):
    d = 3
    D = 9

    all_model = [1216, 1410, 1705, 45]
    all_coef = np.array([[3, -2, 1, 2, -2, 5, -1, 1, 3, 1, -2, -2], [2, 1, 1, 2, 2, 1, 0, -1, 0, -1, 1, 0],
                         [2, -1, 1, 1, -1, 2, -2, 0, 2, -1, 0, 0], [8, 2, 2, 8, -4, 8, 3, 4, 3, -5, 8, -5]])
    all_local = [-9, -4, -5, -20]
    model = all_model[i_model]
    coef = -all_coef[i_model, :]
    local = all_local[i_model]

    D1 = np.diag((1, 1, -1))
    D2 = np.diag((1, -1, -1))
    Diag_list = [D2, D1, D2]

    t_type = "complex"

    run_one_model(model, coef, local, D, d, t_type, n_try, Diag_list)
    # efficient_times = 0
    # for n_try in range(1, int(1e5)):
    #     success = False
    #     while not success:
    #         try:
    #             run_one_model(model, coef, local, D, d, t_type, n_try, Diag_list)
    #             success = True
    #             efficient_times += 1
    #         except:
    #             break
    # print(f'model{model} efficient trails = {efficient_times}')


def parallel_run(run_func, input_model, pool_size):
    pool = Pool(pool_size)
    for initial_num in range(int(1e5)):
        pool.apply_async(func=run_func, args=(initial_num, input_model))
    # for i_model in [2, 1]:  # num0:1216, num1:1410, num2:1705, num3:45
    #     pool.apply_async(func=run_func, args=(i_model, ))
    pool.close()
    pool.join()


if __name__ == '__main__':
    # num1:1410, num2:1705
    input_model = int(input('input model num'))
    parallel_run(running, input_model, 12)
    # for initial_num in range(20):
    #     running(initial_num, 1)