import numpy as np
from math import log
from numpy.linalg import norm
from scipy.linalg import expm

np.set_printoptions(linewidth=200)
qubit_gates = {
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'I': np.array([[1, 0], [0, 1]], dtype=np.complex128)
}


def str_special(str_pr):
    special = {'': 1, 'π': np.pi, '√2': np.sqrt(2), '√3': np.sqrt(3), '√5': np.sqrt(5)}
    if isinstance(str_pr, (int, str)):
        return str(str_pr)
    elif str_pr % 1 == 0:
        return str(int(str_pr))
    div = -1 if str_pr < 0 else 1
    str_pr *= -1 if str_pr < 0 else 1
    for key, val in special.items():
        if isinstance(str_pr, str):
            break
        if np.isclose(str_pr / val % 1, 0):
            div *= int(str_pr / val)
            str_pr = key if div == 1 else f'-{key}' if div == -1 else f'{div}{key}'
        elif np.isclose(val / str_pr % 1, 0):
            div *= int(val / str_pr)
            key = 1 if val == 1 else key
            str_pr = f'{key}/{div}' if div > 0 else f'-{key}/{-div}'
    if isinstance(str_pr, str):
        return str_pr
    return str(round(str_pr * div, 4))


def str_ket(state: np.ndarray, dim: int) -> str:
    '''Get ket format of the qudit state'''
    if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
        state = state.flatten()
    if state.ndim != 1:
        raise ValueError(f'State requires a 1-D ndarray, but get {state.shape}')
    nq = round(log(len(state), dim), 12)
    if nq % 1 != 0:
        raise ValueError(f'Wrong state shape {state.shape} is not a power of {dim}')
    nq = int(nq)
    tol = 1e-8
    string = []
    for ind, val in enumerate(state):
        base = np.base_repr(ind, dim).zfill(nq)
        real = np.real(val)
        imag = np.imag(val)
        str_real = str_special(real)
        str_imag = str_special(imag)
        if np.abs(val) < tol:
            continue
        if np.abs(real) < tol:
            string.append(f'{str_imag}j¦{base}⟩')
            continue
        if np.abs(imag) < tol:
            string.append(f'{str_real}¦{base}⟩')
            continue
        if str_imag.startswith('-'):
            string.append(f'{str_real}{str_imag}j¦{base}⟩')
        else:
            string.append(f'{str_real}+{str_imag}j¦{base}⟩')
    return '\n'.join(string)


class QuditGate:
    '''Base class for qudit gates'''

    def __init__(self, name, n_qudits):
        '''Initialize a QuditGate.'''
        if not isinstance(name, str):
            raise TypeError(f'Excepted string for gate name, get {type(name)}')
        self.name = name
        self.n_qudits = n_qudits

    def on(self, obj_qudits, ctrl_qudits=None):
        '''Define which qudits the gate act on and control qudits'''
        if isinstance(obj_qudits, int):
            obj_qudits = [obj_qudits]
        if isinstance(ctrl_qudits, int):
            ctrl_qudits = [ctrl_qudits]
        if ctrl_qudits is None:
            ctrl_qudits = []
        if set(obj_qudits) & set(ctrl_qudits):
            raise ValueError(f'{self.name} obj_qudits {obj_qudits} and ctrl_qudits {ctrl_qudits} cannot be same')
        if len(set(obj_qudits)) != len(obj_qudits):
            raise ValueError(f'{self.name} gate obj_qudits {obj_qudits} cannot be repeated')
        if len(set(ctrl_qudits)) != len(ctrl_qudits):
            raise ValueError(f'{self.name} gate ctrl_qudits {ctrl_qudits} cannot be repeated')
        if len(obj_qudits) != self.n_qudits:
            raise ValueError(f'{self.name} gate requires {self.n_qudits} qudit, but gets {len(obj_qudits)}')
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits
        return self


class PauliGate(QuditGate):
    '''Pauli qudit gate'''

    def __init__(self, name, n_qudits, dim, ind, obj_qudits=None, ctrl_qudits=None):
        '''Initialize an PauliGate'''
        super().__init__(name, n_qudits)
        if len(ind) != 2:
            raise ValueError(f'{name} index length {len(ind)} should be 2')
        if len(set(ind)) != len(ind):
            raise ValueError(f'{name} index {ind} cannot be repeated')
        if min(ind) < 0 or max(ind) >= dim:
            raise ValueError(f'{name} index {ind} should in 0 to {dim-1}')
        self.dim = dim
        self.ind = ind
        self.name = name
        self.n_qudits = n_qudits
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits

    def __repr__(self):
        '''Return a string representation of the object.'''
        str_obj = ' '.join(str(i) for i in self.obj_qudits)
        str_ctrl = ' '.join(str(i) for i in self.ctrl_qudits)
        str_ind = ''.join(str(i) for i in self.ind)
        if len(str_ctrl):
            return f'{self.name}{str_ind}({str_obj} <-: {str_ctrl})'
        else:
            return f'{self.name}{str_ind}({str_obj})'

    def matrix(self):
        '''Get matrix of the gate'''
        ind = self.ind
        pauli = qubit_gates[self.name]
        mat = np.eye(self.dim, dtype=np.complex128)
        mat[np.ix_(ind, ind)] = pauli
        return mat


class RotationGate(QuditGate):
    '''Rotation qudit gate'''

    def __init__(self, name, n_qudits, dim, pr, ind, obj_qudits=None, ctrl_qudits=None):
        '''Initialize an RotationGate'''
        super().__init__(name, n_qudits)
        if len(ind) != 2:
            raise ValueError(f'{name} index length {len(ind)} should be 2')
        if len(set(ind)) != len(ind):
            raise ValueError(f'{name} index {ind} cannot be repeated')
        if min(ind) < 0 or max(ind) >= dim:
            raise ValueError(f'{name} index {ind} should in 0 to {dim-1}')
        self.pr = pr
        self.dim = dim
        self.ind = ind
        self.name = name
        self.n_qudits = n_qudits
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits

    def __repr__(self):
        '''Return a string representation of the object.'''
        str_obj = ' '.join(str(i) for i in self.obj_qudits)
        str_ctrl = ' '.join(str(i) for i in self.ctrl_qudits)
        str_ind = ''.join(str(i) for i in self.ind)
        str_pr = str_special(self.pr)
        if len(str_ctrl):
            return f'{self.name}{str_ind}({str_pr}|{str_obj} <-: {str_ctrl})'
        else:
            return f'{self.name}{str_ind}({str_pr}|{str_obj})'

    def matrix(self):
        '''Get matrix of the gate'''
        ind = self.ind
        pauli = qubit_gates[self.name[-1]]
        mat = np.eye(self.dim, dtype=np.complex128)
        mat[np.ix_(ind, ind)] = expm(-0.5j * self.pr * pauli)
        return mat


class X(PauliGate):
    '''Pauli-X gate'''

    def __init__(self, dim, ind):
        '''Initialize an X gate'''
        super().__init__('X', 1, dim, ind)


class Y(PauliGate):
    '''Pauli-Y gate'''

    def __init__(self, dim, ind):
        '''Initialize an Y gate'''
        super().__init__('Y', 1, dim, ind)


class Z(PauliGate):
    '''Pauli-Z gate'''

    def __init__(self, dim, ind):
        '''Initialize an Z gate'''
        super().__init__('Z', 1, dim, ind)


class RX(RotationGate):
    '''Rotation qudit gate around x-axis'''

    def __init__(self, dim, pr, ind):
        '''Initialize an RX gate'''
        super().__init__('RX', 1, dim, pr, ind)


class RY(RotationGate):
    '''Rotation qudit gate around y-axis'''

    def __init__(self, dim, pr, ind):
        '''Initialize an RY gate'''
        super().__init__('RY', 1, dim, pr, ind)


class RZ(RotationGate):
    '''Rotation qudit gate around z-axis'''

    def __init__(self, dim, pr, ind):
        '''Initialize an RZ gate'''
        super().__init__('RZ', 1, dim, pr, ind)


class Circuit(list):
    '''The qudit circuit module'''

    def __init__(self, dim):
        '''Initialize a Circuit'''
        list.__init__([])
        self.dim = dim

    def __add__(self, gate):
        '''Addition operator'''
        if self.dim != gate.dim:
            raise ValueError(f'{gate.name} gate requires dimension of {self.dim}, but gets {gate.dim}')
        if not gate.obj_qudits:
            raise ValueError(f'{gate.name} gate should act on some qudits first')
        if isinstance(gate, QuditGate):
            self.append(gate)
        else:
            self.extend(gate)
        return self

    def __iadd__(self, gate):
        '''In-place addition operator'''
        if self.dim != gate.dim:
            raise ValueError(f'{gate.name} gate requires dimension of {self.dim}, but gets {gate.dim}')
        if not gate.obj_qudits:
            raise ValueError(f'{gate.name} gate should act on some qudits first')
        if isinstance(gate, QuditGate):
            self.append(gate)
        else:
            self.extend(gate)
        return self

    def extend(self, gate):
        '''Extend a circuit'''
        super().extend(gate)

    @property
    def n_qudits(self):
        '''Number of qudit circuit'''
        site = []
        for g in circ:
            site += g.obj_qudits
            site += g.ctrl_qudits
        return len(set(site))


# Simulator
class Simulator:
    '''Qudit simulator which simulate qudit circuit'''

    def __init__(self, dim, n_qudits):
        '''Initialize a Simulator object'''
        state = np.zeros(dim**n_qudits, dtype=np.complex128)
        state[0] = 1
        self.dim = dim
        self.sim = state
        self.n_qudits = n_qudits

    def __repr__(self):
        '''Return a string representation of the object'''
        if self.n_qudits < 4:
            return self.get_qs(True)
        return str(self.get_qs())

    def reset(self):
        '''Reset simulator to qudit zero state'''
        state = np.zeros(self.dim**self.n_qudits, dtype=np.complex128)
        state[0] = 1
        self.sim = state

    def get_qs(self, ket: bool = False) -> np.ndarray:
        '''Get qudit state of the simluator'''
        if not isinstance(ket, bool):
            raise TypeError(f'ket requires a bool, but get {type(ket)}')
        if ket:
            return str_ket(self.sim, self.dim)
        return self.sim

    def set_qs(self, state: np.ndarray):
        '''Set qudit state of the simluator'''
        if state.ndim == 2 and (state.shape[0] == 1 or state.shape[1] == 1):
            state = state.flatten()
        if state.ndim != 1:
            raise ValueError(f'State requires a 1-D ndarray, but get {state.shape}')
        nq = round(log(len(state), self.dim), 12)
        if nq % 1 != 0:
            raise ValueError(f'Wrong state shape {state.shape} is not a power of {self.dim}')
        nq = int(nq)
        if self.n_qudits != nq:
            raise ValueError(f'Mismatch number of qudits: state {nq}, simulator {self.n_qudits}')
        div = norm(state, 2)
        if np.isclose(div, 0):
            raise ValueError('Norm of state is equal to 0')
        self.sim = state / div

    def apply_circuit(self, circuit):
        '''Apply a circuit on the simulator'''
        if self.dim != circuit.dim:
            raise ValueError(f'Mismatch dimension: circuit {circuit.dim}, simulator {self.dim}')
        if self.n_qudits < circuit.n_qudits:
            raise ValueError(f'Mismatch number of qudits: circuit {circuit.n_qudits}, simulator {self.n_qudits}')
        d = self.dim
        state = self.sim
        nq = self.n_qudits
        n = self.sim.shape[0]
        for g in circuit:
            idx = nq - g.obj_qudits[0] - 1
            if nq == g.n_qudits:
                state = g.matrix() @ state
            else:
                for i in range(n // d):
                    ii = np.base_repr(i, d).zfill(nq - 1)
                    i_ = [int(ii[:idx] + str(k) + ii[idx:], d) for k in range(d)]
                    state[i_] = g.matrix() @ state[i_]
        self.sim = state / norm(state)


d = 3
t = np.pi / 2
circ = Circuit(d) + X(d, [0, 1]).on(0) + Y(d, [0, 2]).on(1)
circ += Z(d, [1, 2]).on(2)
# circ = Circuit(d) + RX(d, t, [0, 1]).on(0) + RY(d, t, [0, 2]).on(1)
# circ += RZ(d, t, [1, 2]).on(2)
nq = circ.n_qudits
for g in circ:
    print(g.matrix(), g)
print(circ)

sim = Simulator(d, nq)
np.random.seed(42)
state = np.random.rand(d**nq) + 1j * np.random.rand(d**nq)
state /= norm(state)
sim.set_qs(state)
print(sim.get_qs())
sim.apply_circuit(circ)
print(sim.get_qs())

q = {i: np.eye(d)[i] for i in range(d)}
state = np.kron(q[0], q[1])
state /= norm(state)
print(str_ket(state, d))
SWAP = np.zeros([d**2, d**2], dtype=int)
for i in range(d**2):
    base = np.base_repr(i, d).zfill(2)
    j = int(base[::-1], d)
    SWAP[i, j] = 1
state = SWAP @ state
print(str_ket(state, d))
print(SWAP)