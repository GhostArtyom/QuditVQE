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
  - [x] one-qutrit unitary $\to$ symmetric $2$-qubit unitary
  - [x] two-qutrit unitary $\to$ symmetric $4$-qubit unitary
  - [ ] one-qudit unitary $\to$ symmetric $(d-1)$-qubit unitary
  - [ ] two-qudit unitary $\to$ symmetric $(d-1)^2$-qubit unitary
- [ ] Qudit Simulator
  - [x] Universal qutrit decompose / synthesis
  - [ ] Universal qudit decompose / synthesis
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
- TCX/Z: Ternary Controlled-X/Z gate
- CDNOT: Controlled-Double-NOT gate
- [x] Muthukrishnan & Stroud Jr - Multivalued logic gates for quantum computation, 2000 PRA, $\Gamma_2[Y_d]$ 
- [x] JL & R Brylinski - Universal Quantum Gates, 2002 Mathematics of Quantum Computation & arXiv
- [x] Brennen, O'Leary & Bullock - Criteria for exact qudit universality, 2005 PRA, CINC
- [x] Bullock, O'Leary & Brennen - Asymptotically Optimal Quantum Circuits for d-Level Systems, 2005 PRL & 2004 arXiv, ancilla QR & CINC
- [x] Brennen, Bullock & O'Leary - Efficient Circuits for Exact-Universal Computation With Qudits, 2006 QIC & 2005 arXiv, QR & CINC
- [ ] Khan & Perkowski - Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition, 2006 Theor Comput Sci, CSD & uniformly controlled $R_d$
- [x] Sawicki & Karnas - Universality of Single-Qudit Gates, 2017 Annales Henri Poincaré
- [x] Di, Jie & Wei - Cartan decomposition of a two-qutrit gate, 2008 Sci China Ser G, KAK & $\mathrm{SWAP}_3$
- [x] Di & Wei - Elementary gates of ternary quantum logic circuit, 2012 arXiv, TCX & TCZ
- [ ] Di & Wei - Synthesis of multivalued quantum logic circuits by elementary gates, 2013 PRA, GCX & QSD
- [ ] Di & Wei - Optimal synthesis of multivalued quantum circuits, 2015 PRA, GCX & QSD_optimal
- [x] Li, Gu, et al. - Efficient universal quantum computation with auxiliary Hilbert space, 2013 PRA, ququart CDNOT
- [x] Luo, Chen, Yang & Wang - Geometry of Quantum Computation with Qudits, 2014 Sci Rep
- [ ] Luo & Wang - Universal quantum computation with qudits, 2014 Sci China Phys Mech Astron, $C_2[R_d]$ 
- [ ] Wang, Hu, Sanders & Kais - Qudits and High-Dimensional Quantum Computing, 2020 Front Phys, $C_2[R_d]$ 
- [x] Jiang, Wei, Song & Hua - Synthesis and upper bound of Schmidt rank of the bipartite controlled-unitary gates, 2022 arXiv, $\mathbb{C}^M\otimes\mathbb{C}^N$ & GCX
- [ ] Jiang, Liu & Wei - Optimal synthesis of general multi-qutrit quantum computation, 2023 arXiv, GCX & CINC
- [x] Zi, Li & Sun - Optimal Synthesis of Multi-Controlled Qudit Gate, 2023 arXiv, ancilla $\ket{0}\text{-}U$ 
- [ ] Fischer, Tavernelli, et al. - Universal Qudit Gate Synthesis for Transmons, 2023 PRX Quantum, $C^m[U]$ 

Martin Aulbach - Classification of Entanglement in Symmetric States

- [x] Chapter 1.3.1: Majorana Representation
- [x] Chapter 3: Majorana Representation
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
`./QudiTop`
https://gitee.com/forcekeng/quditop
https://github.com/forcekeng/QudiTop

Gates
- [x] PauliGate
- [x] RotationGate
- [x] Control gate
- [x] Parameter gate
- [x] Multi-qudit gate

Circuit
- [x] add +
- [x] iadd +=
- [x] extend
- [x] n_qudits
- [x] reset()
- [x] get_qs()
- [x] matrix()
- [x] on obj_qudits
- [x] on ctrl_qudits
- [x] on ctrl_states