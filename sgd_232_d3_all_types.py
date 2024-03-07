import os
import copy
import math
import random
import numpy as np
import numdifftools as nd
from numpy.linalg import norm
from scipy.linalg import expm
import evoMPS.tdvp_uniform as mps
from logging import info, INFO, basicConfig
# python == 3.8.2 or 3.9.13
# numpy == 1.19.2 or 1.19.3
# scipy == 1.5.2 or 1.5.4
# numdifftools == 0.9.41
# evoMPS == 2.1.0

# the dimension of local observable is d=3
# I0 = np.eye(3)
# I1 = -np.eye(3)
# D1 = np.diag((1, 1, -1))
# D2 = np.diag((1, -1, -1))


def param_A0A1(t, Diag_list):
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
    # commute01 = norm(A0.dot(A1) - A1.dot(A0))
    # commute02 = norm(A0.dot(A2) - A2.dot(A0))
    # commute12 = norm(A1.dot(A2) - A2.dot(A1))
    # print(commute12)
    # if commute01 > 1.0e-6 and commute02 > 1.0e-6 and commute12 > 1.0e-6:
    return A0, A1, A2


def param_hamiltonian(d, t, coef, Diag_list):
    A0, A1, A2 = param_A0A1(t, Diag_list)
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


def cal_e(d, t, coef, A, lfp, rfp, Diag_list):
    ham = param_hamiltonian(d, t, coef, Diag_list)
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
    return e.real


def A_lfp_rfp(D, d, t, coef, evo_step, eta_bound, iter_max, Diag_list):
    ham = param_hamiltonian(d, t, coef, Diag_list)
    sim = mps.EvoMPS_TDVP_Uniform(D, d, ham)
    for i in range(iter_max):
        sim.update()
        energy = sim.h_expect.real
        sim.take_step(evo_step)
        eta = sim.eta.real
        # print(eta)
        if (i + 1) % 300 == 0 and eta > 0.1:
            evo_step = max(evo_step * 0.85, 0.0002)
        if (i + 1) % 500 == 0 and eta > 0.01:
            evo_step = max(evo_step * 0.85, 0.0001)
        if eta < eta_bound or (i + 1) == iter_max:
            return sim.A[0], sim.l[0], sim.r[0], eta, energy


def once_update_t(D, d, t, coef, learning_rate, evo_step, eta_bound, iter_max, Diag_list):
    attempts = 0
    success = False
    while attempts < 3 and not success:
        try:
            A, lfp, rfp, eta, energy = A_lfp_rfp(D, d, t, coef, evo_step, eta_bound, iter_max, Diag_list)
            success = True
        except:
            attempts += 1
            evo_step = max(evo_step * 0.75, 0.001)
            iter_max += 5000
            if attempts == 3:
                break
    fun = lambda t: cal_e(d, t, coef, A, lfp, rfp, Diag_list)
    grad = nd.Gradient(fun)(t)
    t_old = copy.deepcopy(t)
    t += -learning_rate * grad
    return t_old, t, eta, energy, grad


def run_one_model(i_model, D, d, initial_t, type, Diag_list):
    all_model = [1216, 1410, 1705, 45]
    all_coef = np.array([[3, -2, 1, 2, -2, 5, -1, 1, 3, 1, -2, -2], [2, 1, 1, 2, 2, 1, 0, -1, 0, -1, 1, 0],
                         [2, -1, 1, 1, -1, 2, -2, 0, 2, -1, 0, 0], [8, 2, 2, 8, -4, 8, 3, 4, 3, -5, 8, -5]])
    all_local = [-9, -4, -5, -20]
    model = all_model[i_model]
    coef = -all_coef[i_model, :]
    local = all_local[i_model]

    log = os.path.join(os.getcwd(), f'sgd_232/sgd_232_d{d}_all_types_model{model}.log')
    basicConfig(filename=log, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=INFO)

    # print("\n i_model = ", i_model)
    # print("\n model = ", model)
    # print("\n coef = ", coef)
    # print("\n local = ", local)

    evo_step = 0.012
    evo_threshold = 1.0E-3
    evo_iter_max = 20000
    gd_iter_max = 80
    gd_threshold = 1.0E-3
    i = 0
    t = copy.deepcopy(initial_t)
    # print("initial t = ", t)
    # print("type = ", type)
    while i <= gd_iter_max:
        i += 1
        # print("\n iteration = ", i)
        learning_rate = max(0.12 * 0.98**(i - 1), 0.01)
        t_old, t, eta, energy, grad = once_update_t(D, d, t, coef, learning_rate, evo_step, evo_threshold, evo_iter_max, Diag_list)
        grad_norm = norm(grad)
        # print([int(model), int(local), i, energy, t_old, eta, grad_norm, learning_rate])
        # if grad_norm < gd_threshold or i == gd_iter_max or energy - local < 0.001 and energy - local > 0:
        if energy < local:
            info(f'model{model}, local: {local}, energy: {energy}, iter: {i}, eta: {eta}, grad_norm: {grad_norm}, t_old: {list(t_old)}')


def initialize_t(num):
    initial_t = []
    for i in range(num):
        # initial_t.append(random.uniform(-1,1))
        initial_t.append(random.uniform(-math.pi / 2, math.pi / 2))
    # initial_t /= norm(initial_t)
    return initial_t


def generate_obs():
    D1 = np.diag((1, 1, -1))
    D2 = np.diag((1, -1, -1))
    I0 = np.eye(3)
    I1 = -np.eye(3)
    Diag_list = [D1, D2]

    all_Diag = []
    all_type = []
    for i in range(1, 3):
        for j in range(1, 3):
            for k in range(1, 3):
                type = [i, j, k]
                Diag1 = Diag_list[type[0] - 1]
                Diag2 = Diag_list[type[1] - 1]
                Diag3 = Diag_list[type[2] - 1]
                ele = [Diag1, Diag2, Diag3]
                all_Diag.append(ele)
                all_type.append(type)

                for p in range(3):
                    ele0 = copy.deepcopy(ele)
                    type0 = copy.deepcopy(type)
                    ele0[p] = I0
                    type0[p] = 0
                    all_Diag.append(ele0)
                    all_type.append(type0)

                    ele1 = copy.deepcopy(ele)
                    type1 = copy.deepcopy(type)
                    ele1[p] = I1
                    type1[p] = -1
                    all_Diag.append(ele1)
                    all_type.append(type1)
    return all_Diag, all_type


def running(initial_num, type_num, input_model):
    d = 3
    D = 10
    all_Diag, all_type = generate_obs()
    for i_model in [input_model]:
        Diag_list = all_Diag[type_num]
        type = all_type[type_num]
        # print("type = ", type)
        initial_t = initialize_t(9)
        run_one_model(i_model, D, d, initial_t, type, Diag_list)


def parallel_run(run_func, input_model, pool_size=2, callback=None):
    from multiprocessing import Pool
    pool = Pool(pool_size)
    for initial_num in range(10000):
        for type_num in range(56):  # N_type = len(all_Diag) = 56
            pool.apply_async(func=run_func, args=(initial_num, type_num, input_model), callback=callback)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # num0:1216, num1:1410, num2:1705, num3:45
    input_model = int(input('input model num'))
    parallel_run(running, input_model, 8)
    # for initial_num in range(20):
    #     for type_num in range(56):
    #         running(initial_num, type_num, 0)