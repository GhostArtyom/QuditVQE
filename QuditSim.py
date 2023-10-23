import numpy as np
from scipy.linalg import expm

np.set_printoptions(linewidth=200)
qubit_gates = {
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'I': np.array([[1, 0], [0, 1]], dtype=np.complex128)
}


class QuantumGate:
    '''Base class for quantum gates'''

    def __init__(self, name, n_qudits):
        '''Initialize a QuantumGate.'''
        if not isinstance(name, str):
            raise TypeError(f'Excepted string for gate name, get {type(name)}')
        self.name = name
        self.n_qudits = n_qudits

    def __iter__(self):
        '''Iterate quantum circuit'''
        yield from super().__iter__()

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


class RotationGate(QuantumGate):
    '''Rotation qudit gate'''

    def __init__(self, name, n_qudits, dim, ind):
        '''Initialize an RotationGate'''
        super().__init__(name, n_qudits)
        if len(ind) != 2:
            raise ValueError(f'{name} ind length {len(ind)} should be 2')
        if len(set(ind)) != len(ind):
            raise ValueError(f'{name} ind {ind} cannot be repeated')
        if min(ind) < 0 or max(ind) >= dim:
            raise ValueError(f'{name} ind {ind} should in 0 to {dim-1}')
        self.name = name
        self.n_qudits = n_qudits

    def matrix(self):
        '''Get matrix of the gate'''
        ind = self.ind
        pauli = qubit_gates[self.name[-1]]
        mat = np.eye(self.dim, dtype=np.complex128)
        mat[np.ix_(ind, ind)] = expm(-.5j * self.theta * pauli)
        return mat


class RX(RotationGate):
    '''Rotation qudit gate around x-axis'''

    def __init__(self, dim, theta, ind):
        '''Initialize an RX gate'''
        super().__init__('RX', 1, dim, ind)
        self.dim = dim
        self.ind = ind
        self.theta = theta


class Circuit(list):
    '''The qudit circuit module'''

    def __init__(self, dim):
        '''Initialize a Circuit'''
        list.__init__([])
        self.dim = dim

    def extend(self, gate):
        '''Extend a circuit'''
        super().extend(gate)

    def __add__(self, gate):
        '''Addition operator'''
        if self.dim != gate.dim:
            raise ValueError(f'{gate.name} gate requires dimension of {self.dim}, but gets {gate.dim}')
        if isinstance(gate, QuantumGate):
            self.append(gate)
        else:
            self.extend(gate)
        return self

    def __iadd__(self, gate):
        '''In-place addition operator'''
        if isinstance(gate, QuantumGate):
            self.append(gate)
        else:
            self.extend(gate)
        return self

    def matrix(self):
        '''Get matrix of the circuit'''
        for g in self:
            print(g)


d = 3
t = np.pi / 2
circ = Circuit(d) + RX(d, t, [0, 2]).on(0, 1) + RX(d, t, [0, 1]).on(1)
circ += RX(d, t, [0, 2]).on(2)
for i in circ:
    print(i.matrix())
circ.matrix()