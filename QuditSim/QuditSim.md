# Qudit Simulator

已实现

- 常用 1-qudit 门
    - 泡利门 $X_d^{(i,j)},Y_d^{(i,j)},Z_d^{(i,j)}$ 
    - 旋转门 $RX_d^{(i,j)},RY_d^{(i,j)},RZ_d^{(i,j)}$ 
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

- 通用数学门 `UnivMathGate` 
- 单 qudit 门 $H_d,P_d^{(i,j)},\mathrm{INC}_d$ 
- 多 qudit 门 $\rm SWAP,MVCG$ 
- 受控 qudit 门 $\mathrm{CINC}_d,C_2[R_d],C_k[R_d],\mathrm{GCX}_d,C^m[U_d]$ 
- 含参 qudit 门的参数解析器 `ParameterResolver` 
- 求给定 Hamiltonian 的期望 `get_expectation` 
- 求 Hamiltonian 期望和梯度 `get_expectation_with_grad` 
- 用生成梯度算子的信息包装梯度算子 `GradOpsWrapper` 



泡利门、旋转门矩阵形式 [5]

- $i,j$ 为子空间位置 index，需满足 $0\le i\lt j\lt d$ 
```math
\begin{align}
X_d^{(i,j)}&=\ket{i}\bra{j}+\ket{j}\bra{i},\quad
&RY_d^{(i,j)}=\exp\{-\mathrm{i}\theta X_d^{(i,j)}/2\} \\[.5ex]
Y_d^{(i,j)}&=-\mathrm{i}\ket{i}\bra{j}+\mathrm{i}\ket{j}\bra{i},\quad
&RY_d^{(i,j)}=\exp\{-\mathrm{i}\theta Y_d^{(i,j)}/2\} \\[.5ex]
Z_d^{(i,j)}&=\ket{i}\bra{i}-\ket{j}\bra{j},\quad
&RZ_d^{(i,j)}=\exp\{-\mathrm{i}\theta Z_d^{(i,j)}/2\}
\end{align}
```
- 举例说明，此处设 $d=3$ 即 qutrit 体系
```math
X_3^{(0,1)}=\begin{pmatrix}
0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1
\end{pmatrix},\quad
Y_3^{(0,2)}=\begin{pmatrix}
0 & 0 & -\mathrm{i} \\ 0 & 1 & 0 \\ \mathrm{i} & 0 & 0
\end{pmatrix},\quad
Z_3^{(1,2)}=\begin{pmatrix}
1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1
\end{pmatrix} \\
```
```math
RX_3^{(0,1)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & -\mathrm{i}\sin\frac{\theta}{2} & 0 \\
-\mathrm{i}\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
0 & 0 & 1
\end{pmatrix},\;\;
RY_3^{(0,2)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & 0 & -\sin\frac{\theta}{2} \\ 0 & 1 & 0 \\ \sin\frac{\theta}{2} & 0 & \cos\frac{\theta}{2}
\end{pmatrix},\;\;
RZ_3^{(1,2)}(\theta)=\begin{pmatrix}
1 & 0 & 0 \\ 0 & \mathrm{e}^{-i\theta/2} & 0 \\ 0 & 0 & \mathrm{e}^{i\theta/2}
\end{pmatrix} \\
```



通用数学门 UnivMathGate

- 输入单/多 qudit 酉矩阵，生成单/多 qudit 门
- 判断输入矩阵值是否为酉矩阵，维度是否为 d 次幂

单 qudit 门

- 哈达玛门 Hadamard Gate
```math
H_d\ket{j}=\frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}\omega^{ij}\ket{i},\quad \omega=\mathrm{e}^{2\pi\mathrm{i}/d}
```

- 置换门 Permutation Gate
```math
P_d^{(i,j)}=\ket{i}\bra{j}+\ket{j}\bra{i}+\sum_{k\ne i,j}\ket{k}\bra{k}
```

- 增量门 Increment Gate
```math
\mathrm{INC}_d\ket{j}=\ket{(j+1)\bmod d},\quad
\mathrm{INC}_3=\begin{pmatrix}
0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0
\end{pmatrix}
```

多 qudit 门

- 交换门 Swap Gate [7]
```math
\mathrm{SWAP}\ket{i,j}=\ket{j,i}
```

- 多值受控门 Multi-Value-Controlled Gate [7]，其中 $U_i$ 为单 qudit 门
```math
\mathrm{MVCG}=\bigoplus_{i=0}^{d-1}U_i
=\begin{pmatrix}
U_0 \\ & U_1 \\ && \ddots \\ &&& U_{d-1}
\end{pmatrix}
```

受控 qudit 门

- 受控增量门 Controlled-Increment Gate，qudit 版本的 $\rm CNOT$ 门 [1,2,3]
```math
\mathrm{CINC}_d\ket{i,j}=\left\{\begin{array}{c}
\ket{i,(j+1)\bmod d} & i=d-1 \\
\ket{i,j} & i\ne d-1
\end{array}\right.
```
```math
\mathrm{CINC}_d=\mathbb{I}_{d^2-d}\oplus\mathrm{INC}_d
=\begin{pmatrix}
\mathbb{I}_{d^2-d} & \\ & \mathrm{INC}_d
\end{pmatrix}
```

- 当控制位为 $\ket{d-1}$ 态时，单 qudit 受控门 [6,7]
```math
C_2[U_d]=\mathbb{I}_{d^2-d}\oplus R_d
=\begin{pmatrix}
\mathbb{I}_{d^2-d} & \\ & R_d
\end{pmatrix}
```

- 当控制位为 $\ket{d-1}$ 态时，多 qudit 受控门 [6,7]
```math
C_k[R_d]=\mathbb{I}_{d^k-d}\oplus R_d
=\begin{pmatrix}
\mathbb{I}_{d^k-d} & \\ & R_d
\end{pmatrix}
```

- 通用受控置换门 General Controlled X，当控制位为 $\ket{m}$ 态时，作用置换门 $P_d$ 到目标位上
  
    文献 [5] 所用符号为 $X^{(ij)}$，与本文档的置换门 $P_d^{(i,j)}$ 等价
```math
\mathrm{GCX}_d\ket{i,j}=\left\{\begin{array}{c}
\ket{i}\otimes P_d\ket{j} & i=m \\
\ket{i,j} & i\ne m
\end{array}\right.
```

- 通用受控门，当控制位为 $\ket{m}$ 态时，作用 $U_d$ 到目标位上，其中 $U_d$ 为单 qudit 门
```math
C^m[U_d]\ket{i,j}=\left\{\begin{array}{c}
\ket{i}\otimes U_d\ket{j} & i=m \\
\ket{i,j} & i\ne m
\end{array}\right.
```
```math
C^m[U_d]=\ket{m}\bra{m}\otimes U_d+\sum_{i\ne m}\ket{i}\bra{i}\otimes\mathbb{I}_{d^2-d}
=\begin{pmatrix}
\mathbb{I}_{dm} & \\ & U_d \\ && \mathbb{I}_{d(d-m-1)}
\end{pmatrix}
```




参考文献

1. Brennen, O'Leary & Bullock - Criteria for exact qudit universality
2. Bullock, O'Leary & Brennen - Asymptotically Optimal Quantum Circuits for d-Level Systems
3. Brennen, Bullock & O'Leary - Efficient Circuits for Exact-Universal Computation With Qudits
4. Khan & Perkowski - Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition
5. Di & Wei - Synthesis of multivalued quantum logic circuits by elementary gates
6. Luo & Wang - Universal quantum computation with qudits
7. Wang, Hu, Sanders & Kais - Qudits and High-Dimensional Quantum Computing
8. Shende, Bullock & Markov - Synthesis of quantum logic circuits
9. Nielsen & Chuang - Quantum Computation and Quantum Information