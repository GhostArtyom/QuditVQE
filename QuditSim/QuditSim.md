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
- 单 qudit 门 $H_d,\,\mathrm{INC}_d,\,\mathrm{GP}_d$ 
- 多 qudit 门 $\mathrm{SWAP}_d,\,\mathrm{MVCG}_d$ 
- 受控 qudit 门 $\mathrm{CINC}_d,\,\mathrm{GCX}_d$ 
- 已有 qudit 门的控制位与受控态
- 含参 qudit 门的参数解析器 `ParameterResolver` 
- 求给定 Hamiltonian 的期望 `get_expectation` 
- 求 Hamiltonian 期望和梯度 `get_expectation_with_grad` 
- 用生成梯度算子的信息包装梯度算子 `GradOpsWrapper` 



泡利门、旋转门 [5,8,9]

- $i,j$ 为子空间的位置 index，需满足 $0\le i\le d-1,\,0\le j\le d-1$ 
  - `X(dim, ind).on(obj_qudits, ctrl_qudits, ctrl_states)` 
  - `RX(dim, pr, ind).on(obj_qudits, ctrl_qudits, ctrl_states)` 
```math
\begin{aligned}
\sigma_x^{(i,j)}&=\ket{i}\bra{j}+\ket{j}\bra{i},
&X_d^{(i,j)}=\sigma_x^{(i,j)}+\sum_{k\ne i,j}\ket{k}\bra{k},\qquad
&RX_d^{(i,j)}=\exp\{-\mathrm{i}\theta \sigma_x^{(i,j)}/2\} \\
\sigma_y^{(i,j)}&=-\mathrm{i}\ket{i}\bra{j}+\mathrm{i}\ket{j}\bra{i},
&Y_d^{(i,j)}=\sigma_y^{(i,j)}+\sum_{k\ne i,j}\ket{k}\bra{k},\qquad
&RY_d^{(i,j)}=\exp\{-\mathrm{i}\theta \sigma_y^{(i,j)}/2\} \\
\sigma_z^{(i,j)}&=\ket{i}\bra{i}-\ket{j}\bra{j},
&Z_d^{(i,j)}=\sigma_z^{(i,j)}+\sum_{k\ne i,j}\ket{k}\bra{k},\qquad
&RZ_d^{(i,j)}=\exp\{-\mathrm{i}\theta \sigma_z^{(i,j)}/2\}
\end{aligned}
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
\end{pmatrix}
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



通用数学门 Universal Math Gate

- 输入单/多 qudit 酉矩阵，生成单/多 qudit 门
- 判断输入矩阵值是否为酉矩阵，维度是否为 d 次幂
- `UnivMathGate(dim, name, mat).on(obj_qudits, ctrl_qudits, ctrl_states)`

单 qudit 门

- 哈达玛门 Hadamard Gate [7]
  - `H(dim).on(obj_qudits, ctrl_qudits, ctrl_states)` 
```math
H_d\ket{j}=\frac{1}{\sqrt{d}}\sum_{i=0}^{d-1}\omega^{ij}\ket{i},\;\; \omega=\mathrm{e}^{2\pi\mathrm{i}/d},\;\;
(H_d)_{i,j}=\omega^{ij},\;\;
H_3=\frac{1}{\sqrt{3}}
\begin{pmatrix}
1 & 1 & 1 \\
1 & \omega^1 & \omega^{2} \\
1 & \omega^2 & \omega^{4}
\end{pmatrix}
```

- 增量门 Increment Gate [1-3,9]
  - `INC(dim).on(obj_qudits, ctrl_qudits, ctrl_states)` 
```math
\mathrm{INC}_d\ket{j}=\ket{(j+1)\bmod d}
=\begin{pmatrix}
& 1 \\ \mathbb{I}_{d-1}
\end{pmatrix},\quad
\mathrm{INC}_3=\begin{pmatrix}
0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0
\end{pmatrix}
```

- 全局相位门 Global Phase Gate
  - `GP(dim).on(obj_qudits, ctrl_qudits, ctrl_states)` 
```math
\mathrm{GP}_d=\mathrm{diag}\big\{\mathrm{e}^{-i\theta},\mathrm{e}^{-i\theta},\dots,\mathrm{e}^{-i\theta}\big\},\quad
\mathrm{GP}_3=
\begin{pmatrix}
\mathrm{e}^{-i\theta}\! & 0 & 0 \\
0 & \!\mathrm{e}^{-i\theta}\! & 0 \\
0 & 0 & \!\mathrm{e}^{-i\theta}\!
\end{pmatrix}
```

多 qudit 门

- 交换门 Swap Gate [7]
  - `SWAP(dim, obj_qudits=[i, j], ctrl_qudits, ctrl_states)` 
  - 等价于 `SWAP(dim).on(obj_qudits=[i, j], ctrl_qudits, ctrl_states)` 
```math
\mathrm{SWAP}_d\ket{i,j}=\ket{j,i},\quad
(\mathrm{SWAP}_d)_{i,j}=\left\{\begin{array}{c}
1, & j=(i\times d+\lfloor i/d\rfloor)\bmod d^2 \\
0, & \text{others}
\end{array}\right.
```
```math
\mathrm{SWAP}_3=\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
```
```python
SWAP = np.zeros([d**2, d**2], dtype=np.complex128)
for i in range(d**2):
    j = np.mod(i * d + i // d, d**2)
    SWAP[i, j] = 1
```

- 多值受控门 Multi-Value-Controlled Gate，其中 $U_i$ 为单 qudit 门 [7]
  - `MVCG(dim, mat).on(obj_qudits, ctrl_qudits, ctrl_states)` 
  - `mat` 为一组 d 个酉矩阵
```math
\mathrm{MVCG}_d=\bigoplus_{i=0}^{d-1}U_i
=\begin{pmatrix}
U_0 \\ & U_1 \\ && \ddots \\ &&& U_{d-1}
\end{pmatrix}
```

受控 qudit 门

- 受控增量门 Controlled-Increment Gate，qudit 版本的 $\rm CNOT$ 门 [1-3,9]
  - `CINC(dim, obj_qudits=i, ctrl_qudits=j, ctrl_states=d-1)` 
  - 等价于 `INC(dim).on(obj_qudits=i, ctrl_qudits=j, ctrl_states=d-1)` 
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

- 单受控 qudit 门，受控位为 $\ket{d-1}$ 态 [6,7]
```math
C_2[U_d]=\mathbb{I}_{d^2-d}\oplus U_d
=\begin{pmatrix}
\mathbb{I}_{d^2-d} & \\ & U_d
\end{pmatrix}
```

- 多受控 qudit 门，受控位为 $\ket{d-1}$ 态 [6,7]
```math
C_k[U_d]=\mathbb{I}_{d^k-d}\oplus U_d
=\begin{pmatrix}
\mathbb{I}_{d^k-d} & \\ & U_d
\end{pmatrix}
```

- 通用受控 X 门 General Controlled X：当控制位为 $\ket{m}$ 态时，作用泡利 X 门(置换门) $X_d^{(a,b)}$ 到目标位上，受控态需满足 $0\le m\le d-1$ [5,8,9]
  - `GCX(dim, ind, obj_qudits=i, ctrl_qudits=j, ctrl_states=m)`
  - 等价于 `X(dim, ind).on(obj_qudits=i, ctrl_qudits=j, ctrl_states=m)` 
```math
\mathrm{GCX}_d\ket{i,j}=\left\{\begin{array}{c}
\ket{i}\otimes X_d^{(a,b)}\ket{j} & i=m \\
\ket{i,j} & i\ne m
\end{array}\right.
```

- 通用受控门：当控制位为 $\ket{m}$ 态时，作用 $U_d$ 到目标位上，其中 $U_d$ 为单 qudit 门
  - 已有的门都能够如此控制 `.on(obj_qudits=i, ctrl_qudits=j, ctrl_states=m)` 
```math
C^m[U_d]\ket{i,j}=\left\{\begin{array}{c}
\ket{i}\otimes U_d\ket{j} & i=m \\
\ket{i,j} & i\ne m
\end{array}\right.
```
```math
C^m[U_d]=\ket{m}\bra{m}\otimes U_d+\sum_{i\ne m}\ket{i}\bra{i}\otimes\mathbb{I}_{d}
=\begin{pmatrix}
\mathbb{I}_{dm} & \\ & U_d \\ && \mathbb{I}_{d(d-m-1)}
\end{pmatrix}
```
```math
C^m[U_{d^2}]=
 \ket{m}\bra{m}\otimes U_{d^2}+\sum_{i\ne m}\ket{i}\bra{i}\otimes\mathbb{I}_{d^2}
=\begin{pmatrix}
\mathbb{I}_{d^2m} & \\ & U_{d^2} \\ && \mathbb{I}_{d^2(d-m-1)}
\end{pmatrix}
```

- 控制位和受控态
  - `X(dim, ind).on(obj_qudits=0, ctrl_qudits=1)` 作用在 q0 位的 X 门，控制位为 q1, 默认受控态为 $\ket{d-1}$ 
  - `X(dim, ind).on(obj_qudits=0, ctrl_qudits=1, ctrl_states=m)` 控制位为 q1，受控态为 $\ket{m}$ 
  - `X(dim, ind).on(obj_qudits=0, ctrl_qudits=[1, 2], ctrl_states=m)` 控制位为 q1,q2，受控态均为 $\ket{m}$ 
  - `X(dim, ind).on(obj_qudits=0, ctrl_qudits=[1, 2], ctrl_states=[m1, m2])` 控制位为 q1,q2，受控态分别为 $\ket{m_1},\ket{m_2}$，控制位的列表长度与受控态相等



参考文献

1. Brennen, O'Leary & Bullock - Criteria for exact qudit universality
2. Bullock, O'Leary & Brennen - Asymptotically Optimal Quantum Circuits for d-Level Systems
3. Brennen, Bullock & O'Leary - Efficient Circuits for Exact-Universal Computation With Qudits
4. Khan & Perkowski - Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition
5. Di & Wei - Synthesis of multivalued quantum logic circuits by elementary gates
6. Luo & Wang - Universal quantum computation with qudits
7. Wang, Hu, Sanders & Kais - Qudits and High-Dimensional Quantum Computing
8. Jiang, Wei, Song & Hua - Synthesis and upper bound of Schmidt rank of the bipartite controlled-unitary gates
9. Jiang, Liu & Wei - Optimal synthesis of general multi-qutrit quantum computation
10. Shende, Bullock & Markov - Synthesis of quantum logic circuits
11. Nielsen & Chuang - Quantum Computation and Quantum Information