# Qudit_VQE

Qudit variational quantum eigensolver

## Qudit VQE

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
- [ ] Optimize efficiency of `partial_trace()`
- [ ] Feature `su2_decoding()` for `rho`
- [ ] Loss function with gradient
    - [x] Now: using `ham=rho` instead of `rdm[3]` 
    - [ ] Only using `rdm[3]` for Hamiltonian

## LaTeX

- [ ] Prove: projector preserves both operation and unitary
- [ ] Symmetry state encoding for qudit state and unitary gate
- [ ] Decomposition of multi-qubit gate that preserve symmetry

## Read

- [ ] Unitary 2-design / t-design

Martin Aulbach - Classification of Entanglement in Symmetric States

- [ ] Chapter 3: Majorana Representation

Matthew Robinson - Symmetry and the Standard Model꞉ Mathematics and Particle Physics

- [ ] Chapter 3.2.7: $SU(2)$ Groups

Chaichian & Hagedorn - Symmetries in Quantum Mechanics꞉ From Angular Momentum to Supersymmetry

- [ ] Chapter 6: Representations of the Rotation Group

## mindquantum

- [x] Fix `NaN` error for `np.sqrt(eigvals)` 
- [x] Improve precision of `params_zyz()` 
- [x] Fix wrong index of return values of `kron_factor_4x4_to_2x2s()` 

## Extra Work

基于SciPy的qutrit模拟模块
- [ ] qutrit模拟
- [ ] 通用量子门
- [ ] 通用量子电路
