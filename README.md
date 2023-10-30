# QuditVQE

Qudit variational quantum eigensolver

- [x] MPS -> Gates
  - [x] Reduced density matrix
  - [x] State endian (little -> big)
  - [x] Using bra instead of ket
- [x] $SU(2)$ Encoding
  - [x] one-qudit vector
  - [x] one-qudit matrix
  - [x] two-qudit vector
  - [x] two-qudit matrix
  - [x] muti-qudit vector & matrix
- [ ] UnivMathGate -> Parameter Gates
  - [x] one-qubit decompose with params (zyz/u3+GP)
  - [x] two-qubit decompose with params (zyz/u3+Rxyz+GP)
  - [ ] multi-qubit decompose with params (preserve symmetry)
- [ ] Qudit gates generate by qubits
  - [x] Qubit gates which preserve symmetry
  - [x] `p = np.eye(2^nq) - su2_encoding(np.eye(d))` 
  - [x] one-qutrit unitary -> symmetric $2$-qubit unitary
  - [ ] two-qutrit unitary -> symmetric $4$-qubit unitary
  - [ ] one-qudit unitary -> symmetric $(d-1)$-qubit unitary
  - [ ] two-qudit unitary -> symmetric $(d-1)^2$-qubit unitary
- [x] Feature `sym_ind()` & `su2_decoding()` 
- [x] Feature `partial_trace()` for qudit
- [x] Feature `partial_trace()` for `psi` 
- [ ] Loss function with gradient
    - [x] Now: using `ham=rho` instead of `rdm[3]` 
    - [ ] Only using `rdm[3]` for Hamiltonian

## LaTeX

- [ ] Prove: projector preserves both operation and unitary
- [ ] Symmetry state encoding for qudit state and unitary gate
- [ ] Decomposition of multi-qubit gate that preserve symmetry

## Read

- [ ] Unitary 2-design / t-design
- [x] Kochen-Specker Contextuality
- [ ] Semidefinite programming relaxations for quantum correlations

Universal Qudit Gates

- QR: Orthogonal-triangular Decomposition
- CSD: Cosine-Sine Decomposition
- QSD: Quantum Shannon Decomposition
- CINC: Controlled-Increment gate
- GCX: Generalized Controlled-X gate
- CDNOT: Controlled-Double-NOT gate
- [x] Muthukrishnan & Stroud Jr - Multivalued logic gates for quantum computation, 2000 PRA, $\Gamma_2[Y_d]$ 
- [x] JL & R Brylinski - Universal Quantum Gates, 2002 Mathematics of Quantum Computation & arXiv
- [ ] Brennen, O'Leary & Bullock - Criteria for exact qudit universality, 2005 PRA, CINC
- [ ] Bullock, O'Leary & Brennen - Asymptotically Optimal Quantum Circuits for d-Level Systems, 2005 PRL & 2004 arXiv, ancilla QR & CINC
- [ ] Brennen, Bullock & O'Leary - Efficient Circuits for Exact-Universal Computation With Qudits, 2006 QIC & 2005 arXiv, QR & CINC
- [ ] Khan & Perkowski - Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition, 2006 Theor Comput Sci, CSD & uniformly controlled Givens rotation
- [x] Sawicki & Karnas - Universality of Single-Qudit Gates, 2017 Annales Henri Poincaré
- [ ] Di & Wei - Synthesis of multivalued quantum logic circuits by elementary gates, 2013 PRA, GCX
- [x] Li, Gu, et al. - Efficient universal quantum computation with auxiliary Hilbert space, 2013 PRA, ququart CDNOT
- [x] Luo, Chen, Yang & Wang - Geometry of Quantum Computation with Qudits, 2014 Sci Rep
- [ ] Luo & Wang - Universal quantum computation with qudits, 2014 Sci China Phys Mech Astron, $C_2[R_d]$ 
- [ ] Wang, Hu, Sanders & Kais - Qudits and High-Dimensional Quantum Computing, 2020 Frontiers in Physics, $C_2[R_d]$ 
- [ ] Jiang, Wei, Song & Hua - Synthesis and upper bound of Schmidt rank of the bipartite controlled-unitary gates, 2022 arXiv, $\C^M\otimes\C^N$ & GCX
- [ ] Jiang, Liu & Wei - Optimal synthesis of general multi-qutrit quantum computation, 2023 arXiv, GCX & CINC
- [x] Zi, Li & Sun - Optimal Synthesis of Multi-Controlled Qudit Gate, 2023 arXiv, ancilla $\ket{0}\text{-}U$ 
- [ ] Fischer, Tavernelli, et al. - Universal Qudit Gate Synthesis for Transmons, 2023 PRX Quantum, $C^m[U]$ 

Martin Aulbach - Classification of Entanglement in Symmetric States

- [x] Chapter 1.3.1: Majorana Representation
- [ ] Chapter 3: Majorana Representation
- [ ] Chapter 5: Classification of Symmetric State

Matthew Robinson - Symmetry and the Standard Model꞉ Mathematics and Particle Physics

- [ ] Chapter 3.2: Introduction to Lie Groups

Chaichian & Hagedorn - Symmetries in Quantum Mechanics꞉ From Angular Momentum to Supersymmetry

- [ ] Chapter 6: Representations of the Rotation Group

## MindQuantum

- [x] Fix `NaN` error for `np.sqrt(eigvals)` 
- [x] Improve precision of `params_zyz()` 
- [x] Fix wrong index of return values of `kron_factor_4x4_to_2x2s()` 

## QuditSim

QuditGate
- [x] PauliGate
- [x] RotationGate
- [ ] Control gate
- [ ] Parameter gate
- [ ] Multi-qudit gate

Circuit
- [x] add +
- [x] iadd +=
- [x] extend
- [x] n_qudits

Simulator
- [x] reset
- [x] get_qs
- [x] set_qs
- [ ] apply_circuit
    - [x] on obj_qudits
    - [ ] on ctrl_qudits