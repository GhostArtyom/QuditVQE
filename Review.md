# 高维量子门的最优通用合成

分享人：朱彦铮  电子科技大学  硕士研究生

## 内容简介





## 相关论文

论文1：Synthesis of multivalued quantum logic circuits by elementary gates

- 作者：Yao-Min Di, Hai-Rui Wei

- 期刊：Physical Review A

论文2：Optimal synthesis of multivalued quantum circuits

- 作者：Yao-Min Di, Hai-Rui Wei

- 期刊：Physical Review A



## 1. 引言

量子计算作为一种基于量子物理的新兴计算范式，具有原理上远超经典计算的强大并行运算能力[1]。它为密码破解、人工智能、气象预报、生物医药、材料科学等所需的大规模计算难题提供了解决方案，并可揭示高温超导、量子相变、量子霍尔效应等复杂物理机制。在量子物理的底层，通常由人工可操控的二能级体系来实现量子比特 (qubit)，例如超导、光子、离子阱和中性原子等物理体系。然而在这些量子体系中天然具有多个量子化的本征态，包括原子中电子能级结构、分子振动能级等，其中蕴含了非常丰富的物理化学性质。虽然上述可操控的量子体系通常含有多个本征态，只是由于高维量子操控技术还不成熟，使得过去的研究更多地关注于量子比特体系下的量子计算和量子模拟。

近年来，高维量子信息科学与技术通过人工操控多能级的高维量子位 (qudit)，进而实现量子信息的编码、传输、处理与存储，有望实现更加强大的量子计算、量子模拟和量子通信等功能。在实验上，高维量子位和高维量子纠缠态已在超导、光子、离子阱和中性原子等体系中实现。在理论上，基于电路模型和测量模型的高维通用量子计算已被证明为可行的[2,3]，且有助于提升量子计算算法的性能、降低量子纠错所需物理资源等。

如何做到通用的高维量子计算？这里的通用指的是什么？

通用的意思是有一组通用量子门集合，任意的酉变换都可以由这些门实现。

本文介绍了一种对于任意 $d$ 维量子门的最优通用合成方法。

> 然而，当下大多数的量子计算平台和数值模拟库都是基于二进制量子比特体系构建与研发的，对于高维量子系统的实验和数值研究都较为稀缺。高维量子系统是指拥有高维度 $(d>2)$ 态空间的量子体系，其相对于量子比特 $(d=2)$ 体系而言，它具有更高的自由度、更多的信息容量以及更复杂的态空间结构，可以降低电路复杂度、简化实验设置、提高计算效率和问题求解能力。高维量子系统在面对噪声和干扰时表现出更强的抗干扰能力，因此在实际应用中具有更高的可行性。高维量子系统的优势使其成为量子计算和量子信息处理领域重要的研究方向，具备广阔的应用前景，如量子模拟、量子纠错、量子优化和量子机器学习等领域。 




## 2. 高维量子门

正如二进制位是经典计算中信息的基础对象一样，量子比特 (quantum bit, qubit) 是量子计算中信息的基础对象。而高维量子位 (quantum digit, qudit) 便是基于 $d$ 进制数的高维量子计算中的信息基础对象，其状态可以通过 $d$ 维希尔伯特 (Hilbert) 空间 $\mathcal{H}_d$ 中的向量来描述，该空间可以由一组标准正交基 ${\ket{0},\ket{1},\dots,\ket{d-1}}$ 构成[4]。一个高维量子态通常可以写成如下形式：
$$
\ket{\alpha}=\alpha_0\ket{0}+\alpha_1\ket{1}+\cdots+\alpha_{d-1}\ket{d-1}\in\mathbb{C}^d
=\begin{pmatrix}\alpha_0\\\alpha_1\\\vdots\\\alpha_{d-1}\end{pmatrix},\quad
\sum_{j=0}^{d-1}|\alpha_j|^2=1
$$
高维量子位可以替代量子比特作为量子计算的基本元素，高维量子态可以由高维量子门变换？

在常见的量子比特体系中，常用的量子逻辑门有泡利门 $X,Y,Z$、旋转门 $Rx,Ry,Rz$ 以及受控门 $\text{CNOT}$，它们的矩阵形式分别如下[1]：
$$
X=\begin{pmatrix}
0 & 1 \\ 1 & 0
\end{pmatrix},\quad
Y=\begin{pmatrix}
0 & -i \\ i & 0
\end{pmatrix},\quad
Z=\begin{pmatrix}
1 & 0 \\ 0 & -1
\end{pmatrix},\quad
\text{CNOT}
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{pmatrix} \\
Rx(\theta)
=\begin{pmatrix}
\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
-i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
\end{pmatrix},\quad
Ry(\theta)
=\begin{pmatrix}
\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
\end{pmatrix},\quad
Rz(\theta)
=\begin{pmatrix}
e^{-i\theta/2} & 0 \\
0 & e^{i\theta/2}
\end{pmatrix}
$$
在高维量子位体系中，基础的高维量子门由量子比特门衍生而来，？由标准正交基 $\{\ket{i},\ket{j}\}$ 张成的二维希尔伯特子空间 $\mathcal{H}_{ij}$ [5]

任意 $d$ 维的量子门可以由 $U(d)$ 群的生成元 $\sigma_\alpha^{(i,j)}$ 构成，其中 $0\le i<j\le d-1$，
$$
\begin{aligned}
\sigma_x^{(i,j)}&=\ket{i}\bra{j}+\ket{j}\bra{i},
&X_d^{(i,j)}=\sigma_x^{(i,j)}+\sum_{k\ne i,j}\ket{k}\bra{k},\qquad
&Rx_d^{(i,j)}=\exp\{-\mathrm{i}\theta \sigma_x^{(i,j)}/2\} \\
\sigma_y^{(i,j)}&=-\mathrm{i}\ket{i}\bra{j}+\mathrm{i}\ket{j}\bra{i},
&Y_d^{(i,j)}=\sigma_y^{(i,j)}+\sum_{k\ne i,j}\ket{k}\bra{k},\qquad
&Ry_d^{(i,j)}=\exp\{-\mathrm{i}\theta \sigma_y^{(i,j)}/2\} \\
\sigma_z^{(i,j)}&=\ket{i}\bra{i}-\ket{j}\bra{j},
&Z_d^{(i,j)}=\sigma_z^{(i,j)}+\sum_{k\ne i,j}\ket{k}\bra{k},\qquad
&Rz_d^{(i,j)}=\exp\{-\mathrm{i}\theta \sigma_z^{(i,j)}/2\}
\end{aligned}
$$

例如在 $d=3$ 的 qutirt 体系下，泡利门和旋转门的矩阵形式分别如下：
$$
X_3^{(0,1)}=\begin{pmatrix}
0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1
\end{pmatrix},\quad
Y_3^{(0,2)}=\begin{pmatrix}
0 & 0 & -\mathrm{i} \\ 0 & 1 & 0 \\ \mathrm{i} & 0 & 0
\end{pmatrix},\quad
Z_3^{(1,2)}=\begin{pmatrix}
1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1
\end{pmatrix}
$$

$$
Rx_3^{(0,1)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & -\mathrm{i}\sin\frac{\theta}{2} & 0 \\
-\mathrm{i}\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
0 & 0 & 1
\end{pmatrix},\;\;
Ry_3^{(0,2)}(\theta)=\begin{pmatrix}
\cos\frac{\theta}{2} & 0 & -\sin\frac{\theta}{2} \\ 0 & 1 & 0 \\ \sin\frac{\theta}{2} & 0 & \cos\frac{\theta}{2}
\end{pmatrix},\;\;
Rz_3^{(1,2)}(\theta)=\begin{pmatrix}
1 & 0 & 0 \\ 0 & \mathrm{e}^{-i\theta/2} & 0 \\ 0 & 0 & \mathrm{e}^{i\theta/2}
\end{pmatrix}
$$

对于受控量子门，与量子比特受控门 $\text{CNOT}$ 只能控制 $\ket{1}$ 态不同，高维通用受控门可以控制不同的态[5]。例如，通用受控 $X$ 门 (general controlled X, GCX) 为 $\text{CNOT}$ 门在高维量子计算中的推广。当控制位为 $\ket{m}$ 态时，作用泡利 $X_d^{(i,j)}$ 门到目标位上，其中受控态需满足 $0\le m\le d-1$，
$$
\text{GCX}\Big(m\to X_d^{(i,j)}\Big)
=\ket{m}\bra{m}\otimes X_d^{(i,j)}+\sum_{n\ne m}\ket{n}\bra{n}\otimes I_{d}
=\begin{pmatrix}
I_{dm} & \\ & X_d^{(i,j)} \\ && I_{d(d-m-1)}
\end{pmatrix}
$$
![GCX](./img/GCX.png)

对于任意的高维泡利门和旋转门这样的单量子位门，都可以生成类似的高维通用受控门。



## 3. 最优通用合成

> 通用合成的意义 gate set 泡利+旋转+GCX，GCX数量用于衡量电路复杂度

设 $U\in U(d)$ 为单量子门矩阵，可以使用 Cartan 分解成[6]
$$
U=\mathrm{e}^{\mathrm{i}\varphi}U_1^{(i,j)}U^{(i',j')}U_2^{(i,j)}
$$
与量子比特体系类似，可以使用 ZYZ 分解对 $U^{(i,j)}$ ？
$$
U^{(i,j)}=Rz^{(i,j)}(\phi)\,Ry^{(i,j)}(\theta)\,Rz^{(i,j)}(\lambda)
$$

### 3.1. 余弦正弦分解 (cosine-sine decomposition, CSD)[7]

将酉矩阵 $W\in\mathbb{C}^{m\times m}$ 利用余弦正弦分解成 $W=U\Gamma V$ 

$r=[d/2]d^{n−1}$ 
$$
U=\,\begin{array}{r}
r \\ m-r
\end{array}
\begin{pmatrix}
U_1 \\ & U_2
\end{pmatrix},\quad
\Gamma=\begin{array}{r}
r \\ r \\ m-2r
\end{array}
\begin{pmatrix}
C & -S \\
S & C \\
&& I_{m-2r}
\end{pmatrix},\quad
V=\,\begin{array}{r}
r \\ m-r
\end{array}
\begin{pmatrix}
V_1 \\ & V_2
\end{pmatrix}
$$

其中 $U_1,V_1$ 为 $r\times r$ 酉矩阵，$U_2,V_2$ 为 $(m-r)\times(m-r)$ 酉矩阵，$C,S$ 为 $r\times r$ 对角余弦正弦矩阵
$$
C=\begin{pmatrix}
\cos\theta_1 \\
& \cos\theta_2 \\
&& \ddots \\
&&& \cos\theta_r
\end{pmatrix},\quad
S=\begin{pmatrix}
\sin\theta_1 \\
& \sin\theta_2 \\
&& \ddots \\
&&& \sin\theta_r
\end{pmatrix}
$$
此处对于任意 $\theta_i\,(1\le i\le r)$ 都有 $\sin^2\theta_i+\cos^2\theta_i=1$。任意的 $n$ 位 $d$ 维量子门由相应的 $d^n\times d^n$ 酉矩阵表示，

### 3.2. 量子香农分解 (quantum Shannon decomposition, QSD)[8]

递归

受控酉矩阵门
$$
U=V\Delta V^\dagger,\quad
\Delta=\mathrm{e}^{\mathrm{i}\varphi}Rz^{(0,1)}(\theta_1)\,Rz^{(0,2)}(\theta_2)\cdots Rz^{(0,d-1)}(\theta_{d-1})
$$
![CU](./img/CU.png)

均匀受控门

![UCG](./img/UCG.png)



以 $d=3$ 的 qutirt 体系为例展示 QSD 分解过程，设 $W$ 为 $3^n\times 3^n$ 酉矩阵，对其执行 CSD 分解得到
$$
W=U\Gamma V=
\begin{pmatrix}
U_1 \\ & U_2
\end{pmatrix}
\begin{pmatrix}
C & -S \\
S & C \\
&& I
\end{pmatrix}
\begin{pmatrix}
V_1 \\ & V_2
\end{pmatrix}
$$

对于 $2\times3^{n-1}$ 阶酉矩阵 $U_2,V_2$ 继续执行 CSD 分解得到
$$
U_2=\begin{pmatrix}
U_{2,1} \\ & U_{2,2}
\end{pmatrix}
\begin{pmatrix}
C_1 & -S_1 \\ S_1 & C_1
\end{pmatrix}
\begin{pmatrix}
U_{2,3} \\ & U_{2,4}
\end{pmatrix},\quad
V_2=\begin{pmatrix}
V_{2,1} \\ & V_{2,2}
\end{pmatrix}
\begin{pmatrix}
C_2 & -S_2 \\ S_2 & C_2
\end{pmatrix}
\begin{pmatrix}
V_{2,3} \\ & V_{2,4}
\end{pmatrix}
$$

$$
U=A\Gamma_1B=
\begin{pmatrix}
U_1 \\ & U_{2,1} \\ && U_{2,2}
\end{pmatrix}
\begin{pmatrix}
I \\ & C_1 & -S_1 \\ & S_1 & C_1
\end{pmatrix}
\begin{pmatrix}
I \\ & U_{2,3} \\ && U_{2,4}
\end{pmatrix}
$$

$$
V=C\Gamma_2D=
\begin{pmatrix}
V_1 \\ & V_{2,1} \\ && V_{2,2}
\end{pmatrix}
\begin{pmatrix}
I \\ & C_2 & -S_2 \\ & S_2 & C_2
\end{pmatrix}
\begin{pmatrix}
I \\ & V_{2,3} \\ && V_{2,4}
\end{pmatrix}
$$

至此每个分块矩阵均为 $3^{n-1}\times 3^{n-1}$ 酉矩阵，继续将部分矩阵分解成张量积形式
$$
A'=(I\otimes U_{2,1})
\begin{pmatrix}
I \\ & I \\ && U_{2,2}'
\end{pmatrix},\quad
B'=(I\otimes U_{2,3})
\begin{pmatrix}
U_2' \\ & I \\ && I
\end{pmatrix} \\
C'=(I\otimes V_{2,1})
\begin{pmatrix}
I \\ & I \\ && V_{2,2}'
\end{pmatrix},\quad
D'=\begin{pmatrix}
V_2' \\ & V_{2,3} \\ && V_{2,4}
\end{pmatrix}
$$

$$
U_{2,2}'=U_{2,1}^{-1},\quad U_2'=U_{2,3}^{-1}U_{2,1}^{-1}U_1,\quad
V_{2,2}'=V_{2,1}^{-1}U_{2,3}^{-1}U_{2,4}V_{2,2},\quad V_2'=V_{2,1}^{-1}U_2'
$$

因此得到分解：$W=A'\,\Gamma_1\,B'\,\Gamma C'\,\Gamma_2\,D'$。可以将 $A',C'$ 对应于一个控制 $\ket{2}$ 态的受控酉门， $B'$ 对应于一个控制 $\ket{0}$ 态的受控酉门， $D'$ 对应于两个分别控制 $\ket{1},\ket{2}$ 态的受控酉门。

而对于余弦正弦矩阵 $\Gamma$ 可以分解成一组均匀受控 $Ry^{(0,1)}$ 门，用方块 $\square$ 表示控制位。同理 $\Gamma_1,\Gamma_2$ 可以由一组均匀受控 $Ry^{(1,2)}$ 门表示


![QSD3](./img/QSD3.png)

其中通用受控对角门可以分解成如下形式

![CD3](./img/CD3.png)

其中 $S_m$ 为关于 $\ket{m}$ 态的相位门

$$
S_m=\mathrm{e}^{\mathrm{i}\varphi}\ket{m}\bra{m}+\sum_{i\ne m}\ket{i}\bra{i}
$$




## 4. 总结







## 参考文献

[1] Nielsen M A, Chuang I L. Quantum Computation and Quantum Information[M]. Cambridge University Press, 2010.

[2] Luo M, Wang X. Universal quantum computation with qudits[J]. Science China Physics, Mechanics & Astronomy, 2014, 57(9): 1712-1717.

[3] Wang Y, Hu Z, Sanders B C, et al. Qudits and High-Dimensional Quantum Computing[J]. Frontiers in Physics, 2020, 8: 589504.

[4] Brylinski J L, Brylinski R. Universal quantum gates[J]. Mathematics of quantum computation, 2002, 79.

[5] Di Y M, Wei H R. Synthesis of multivalued quantum logic circuits by elementary gates[J]. Physical Review A, 2013, 87(1): 012325.

[6] Helgason S. Differential geometry, Lie Groups, and Symmetric Spaces[M]. Academic Press, 1979.

[7] Khan F S, Perkowski M. Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition[J]. Theoretical Computer Science, 2006, 367(3): 336-346.

[8] Shende V V, Bullock S S, Markov I L. Synthesis of Quantum-Logic Circuits[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2006, 25(6): 1000-1010.

[9] Di Y M, Wei H R. Optimal synthesis of multivalued quantum circuits[J]. Physical Review A, 2015, 92(6): 062317.