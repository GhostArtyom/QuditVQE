import numpy as np
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
    if str_pr == 0 or abs(str_pr) == 1:
        return str_pr
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
    else:
        str_pr = round(str_pr * div, 4)
    return str_pr


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


class RotationGate(QuditGate):
    '''Rotation qudit gate'''

    def __init__(self, name, n_qudits, dim, pr, ind, obj_qudits=None, ctrl_qudits=None):
        '''Initialize an RotationGate'''
        super().__init__(name, n_qudits)
        if len(ind) != 2:
            raise ValueError(f'{name} ind length {len(ind)} should be 2')
        if len(set(ind)) != len(ind):
            raise ValueError(f'{name} ind {ind} cannot be repeated')
        if min(ind) < 0 or max(ind) >= dim:
            raise ValueError(f'{name} ind {ind} should in 0 to {dim-1}')
        self.pr = pr
        self.dim = dim
        self.ind = ind
        self.name = name
        self.n_qudits = n_qudits
        self.obj_qudits = obj_qudits
        self.ctrl_qudits = ctrl_qudits

    def __str__(self):
        """Return a string representation of the object."""
        str_obj = ' '.join([str(i) for i in self.obj_qudits])
        str_ctrl = ' '.join([str(i) for i in self.ctrl_qudits])
        str_ind = ''.join([str(i) for i in self.ind])
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

    def matrix(self):
        '''Get matrix of the circuit'''


d = 3
t = np.pi
circ = Circuit(d) + RX(d, t, [0, 2]).on(0, 1) + RY(d, t / 2, [0, 1]).on(1)
circ += RZ(d, t / 3, [0, 2]).on(2)
for g in circ:
    print(g)
print(circ)