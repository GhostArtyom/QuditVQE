import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix


def partial_trace(rho, index):
    nq = int(rho.shape[0] / 2)
    d = int(np.log2(nq))
    pt = np.zeros([nq, nq], dtype=complex)
    for i in range(nq):
        i_ = bin(i)[2::].zfill(d)
        i0 = int(i_[:index] + '0' + i_[index:], 2)
        i1 = int(i_[:index] + '1' + i_[index:], 2)
        for j in range(nq):
            j_ = bin(j)[2::].zfill(d)
            j0 = int(j_[:index] + '0' + j_[index:], 2)
            j1 = int(j_[:index] + '1' + j_[index:], 2)
            pt[i, j] = rho[i0, j0] + rho[i1, j1]
    return pt


def reduced_density_matrix(rho, position):
    nq = int(np.log2(rho.shape[0]))
    p = [x for x in range(nq) if x not in position]
    for i in p[::-1]:
        rho = partial_trace(rho, i)
    return rho


def SU2_Encoding(qudit):
    d = np.shape(qudit)
    if np.iscomplex(qudit).any():
        dtype = 'complex'
    else:
        dtype = 'float64'
    if len(d) == 2 and d[0] == d[1]:
        d = d[0]
        nq = d - 1
        qubits = csr_matrix(np.zeros([2**nq, 2**nq], dtype))
        nq_bin = {}
        for i in range(2**nq):
            num1 = bin(i).count('1')
            if num1 in nq_bin:
                nq_bin[num1].append(i)
            else:
                nq_bin[num1] = [i]
        for i in range(d):
            qubits_i = nq_bin[i]
            num_i = len(qubits_i)
            for j in range(d):
                qubits_j = nq_bin[j]
                num_j = len(qubits_j)
                ii = qubits_i * num_j
                jj = np.repeat(qubits_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubits += csr_matrix((data, (ii, jj)), shape=(2**nq, 2**nq))
        qubits = qubits.toarray()
    elif len(d) == 1 or (len(d) == 2 and (d[0] == 1 or d[1] == 1)):
        if len(d) == 2 and d[0] == 1:
            qudit = qudit.flatten()
            d = d[1]
        elif len(d) == 2 and d[1] == 1:
            qudit = qudit.flatten()
            d = d[0]
        else:
            d = d[0]
        nq = d - 1
        qubits = csr_matrix(np.zeros(2**nq, dtype))
        nq_bin = {}
        for i in range(2**nq):
            num1 = bin(i).count('1')
            if num1 in nq_bin:
                nq_bin[num1].append(i)
            else:
                nq_bin[num1] = [i]
        for i in range(d):
            qubits_i = nq_bin[i]
            num_i = len(qubits_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            ind = (np.zeros(num_i), qubits_i)
            qubits += csr_matrix((data, ind), shape=(1, 2**nq))
        qubits = qubits.toarray().flatten()
        qubits /= norm(qubits)
    else:
        raise ValueError('Wrong Input!')
    return qubits