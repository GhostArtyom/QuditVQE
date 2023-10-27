# Qudit Simulator

已实现

- 常用 1-qudit 门
    - 泡利门 $X^{(j,k)},Y^{(j,k)},Z^{(j,k)}$ 
    - 旋转门 $RX^{(j,k)},RY^{(j,k)},RZ^{(j,k)}$ 
- Qudit 电路 `class Circuit` 
    - 加号运算 `+` 
    - 增量赋值运算 `+=` 
    - 电路扩展运算 `extend` 
    - 电路 qudit 数量运算 `n_qudits` 
- Qudit 模拟器 `class Simulator` 
    - 将模拟器重置为零态 `reset` 
    - 获取模拟器的当前量子态 `get_qs` 
    - 设置模拟器的量子态 `set_qs` 
    - 在模拟器上应用量子电路 `apply_circuit` 




待实现

> 有些函数可以套用 MindQuantum 的同名函数，但是需将判断是否为 2 次幂 `is_power_of_two` 换成判断是否为 d 次幂

- 多 Qudit 门，受控 Qudit 门
- 含参 Qudit 门的参数解析器 `ParameterResolver` 
- 求给定hamiltonian的期望 `get_expectation` 
- 求期望及梯度 `get_expectation_with_grad` 
- 用生成梯度算子的信息包装梯度算子 `GradOpsWrapper` 
- 计算当前密度矩阵的偏迹 `get_partial_trace` 
- 计算两个量子态的保真度 `fidelity` 



泡利门、旋转门矩阵形式

- $j,k$ 为子空间位置 index，需满足 $0\le j<k<d$ 

$$
\begin{align}
X^{(j,k)}&=\ket{j}\bra{k}+\ket{k}\bra{j},\quad
&RY^{(j,k)}=\exp{-\mathrm{i}\theta X^{(j,k)}/2} \\[.5ex]
Y^{(j,k)}&=-\mathrm{i}\ket{j}\bra{k}+\mathrm{i}\ket{k}\bra{j},\quad
&RY^{(j,k)}=\exp{-\mathrm{i}\theta Y^{(j,k)}/2} \\[.5ex]
Z^{(j,k)}&=\ket{j}\bra{j}-\ket{k}\bra{k},\quad
&RZ^{(j,k)}=\exp{-\mathrm{i}\theta Z^{(j,k)}/2}
\end{align}
$$

- 举例说明，此处设 $d=3$ 即 qutrit 体系

$$
X^{(0,1)}=\begin{pmatrix}
0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1
\end{pmatrix},\quad
Y^{(0,2)}=\begin{pmatrix}
0 & 0 & -\mathrm{i} \\ 0 & 1 & 0 \\ \mathrm{i} & 0 & 0
\end{pmatrix},\quad
Z^{(1,2)}=\begin{pmatrix}
1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1
\end{pmatrix} \\
$$

$$
RX^{(0,1)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & -\mathrm{i}\sin\frac{\theta}{2} & 0 \\
-\mathrm{i}\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
0 & 0 & 1
\end{pmatrix},\quad
RY^{(0,2)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & 0 & -\sin\frac{\theta}{2} \\ 0 & 1 & 0 \\ \sin\frac{\theta}{2} & 0 & \cos\frac{\theta}{2}
\end{pmatrix},\quad
RZ^{(1,2)}(\theta)=\begin{pmatrix}
1 & 0 & 0 \\ 0 & e^{-i\theta/2} & 0 \\ 0 & 0 & e^{i\theta/2}
\end{pmatrix} \\
$$



多 Qudit 门



受控 Qudit 门



参考文献

- JL & R Brylinski - Universal Quantum Gates
- Muthukrishnan & Stroud Jr - Multivalued logic gates for quantum computation
- Brennen, O'Leary & Bullock - Criteria for exact qudit universality
- Bullock, O'Leary & Brennen - Asymptotically Optimal Quantum Circuits for d-Level Systems
- Brennen, Bullock & O'Leary - Efficient Circuits for Exact-Universal Computation With Qudits
- Khan & Perkowski - Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition
- Sawicki & Karnas - Universality of Single-Qudit Gates
- Di & Wei - Synthesis of multivalued quantum logic circuits by elementary gates
- Li, Gu, et al. - Efficient universal quantum computation with auxiliary Hilbert space
- Luo & Wang - Universal quantum computation with qudits
- Wang, Hu, Sanders & Kais - Qudits and High-Dimensional Quantum Computing
- Zi, Li & Sun - Optimal Synthesis of Multi-Controlled Qudit Gate
- Shende, Bullock & Markov - Synthesis of quantum logic circuits
- Nielsen & Chuang - Quantum Computation and Quantum Information
