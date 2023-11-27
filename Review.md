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

量子计算作为一种基于量子物理的新兴计算范式，具有原理上远超经典计算的强大并行运算能力[1]。它为密码破解、人工智能、气象预报、生物医药、材料科学等所需的大规模计算难题提供了解决方案，并可揭示高温超导、量子相变、量子霍尔效应等复杂物理机制。在量子物理的底层，通常由人工可操控的二能级体系来实现量子比特(qubit)，例如超导、光子、离子阱和中性原子等物理体系。然而在这些量子体系中天然具有多个量子化的本征态，包括原子中电子能级结构、分子振动能级等，其中蕴含了非常丰富的物理化学性质。虽然上述可操控的量子体系通常含有多个本征态，只是由于高维量子操控技术还不成熟，使得过去的研究更多地关注于量子比特体系下的量子计算和量子模拟。

近年来，高维量子信息科学与技术通过人工操控多能级的高维量子位(qudit)，进而实现量子信息的编码、传输、处理与存储，有望实现更加强大的量子计算、量子模拟和量子通信等功能。在实验上，高维量子位和高维量子纠缠态已在超导、光子、离子阱和中性原子等体系中实现。在理论上，基于电路模型和测量模型的高维通用量子计算已被证明为可行的[2,3]，且有助于提升量子计算算法的性能、降低量子纠错所需物理资源等。

如何做到通用的高维量子计算？这里的通用指的是什么？

通用的意思是有一组通用量子门集合，任意的酉变换都可以由这些门实现。

本文介绍了一种对于任意 $d$ 维量子门的最优通用合成方法。

> 然而，当下大多数的量子计算平台和数值模拟库都是基于二进制量子比特体系构建与研发的，对于高维量子系统的实验和数值研究都较为稀缺。高维量子系统是指拥有高维度 $(d>2)$ 态空间的量子体系，其相对于量子比特 $(d=2)$ 体系而言，它具有更高的自由度、更多的信息容量以及更复杂的态空间结构，可以降低电路复杂度、简化实验设置、提高计算效率和问题求解能力。高维量子系统在面对噪声和干扰时表现出更强的抗干扰能力，因此在实际应用中具有更高的可行性。高维量子系统的优势使其成为量子计算和量子信息处理领域重要的研究方向，具备广阔的应用前景，如量子模拟、量子纠错、量子优化和量子机器学习等领域。 




## 2. 

单量子位门 X, Rx01 02 12, $\Delta$ 

通用受控门 GCX, GCRx, GC$\Delta$ 





## 3. 

ZYZ

CSD

QSD







## 4. 总结







## 参考文献

[1] Nielsen M A, Chuang I L. Quantum Computation and Quantum Information[M]. Cambridge University Press, 2010.

[2] Luo M, Wang X. Universal quantum computation with qudits[J]. Science China Physics, Mechanics & Astronomy, 2014, 57(9): 1712-1717.

[3] Wang Y, Hu Z, Sanders B C, et al. Qudits and High-Dimensional Quantum Computing[J]. Frontiers in Physics, 2020, 8: 589504.

Khan F S, Perkowski M. Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition[J]. Theoretical Computer Science, 2006, 367(3): 336-346.

Di Y M, Wei H R. Synthesis of multivalued quantum logic circuits by elementary gates[J]. Physical Review A, 2013, 87(1): 012325.

Di Y M, Wei H R. Optimal synthesis of multivalued quantum circuits[J]. Physical Review A, 2015, 92(6): 062317.



Muthukrishnan & Stroud Jr - Multivalued logic gates for quantum computation, 2000 PRA, $\Gamma_2[Y_d]$ 
JL & R Brylinski - Universal Quantum Gates, 2002 Mathematics of Quantum Computation & arXiv
Brennen, O'Leary & Bullock - Criteria for exact qudit universality, 2005 PRA, CINC
Bullock, O'Leary & Brennen - Asymptotically Optimal Quantum Circuits for d-Level Systems, 2005 PRL & 2004 arXiv, ancilla QR & CINC
Brennen, Bullock & O'Leary - Efficient Circuits for Exact-Universal Computation With Qudits, 2006 QIC & 2005 arXiv, QR & CINC
Khan & Perkowski - Synthesis of multi-qudit hybrid and d-valued quantum logic circuits by decomposition, 2006 Theor Comput Sci, CSD & uniformly controlled $R_d$
Sawicki & Karnas - Universality of Single-Qudit Gates, 2017 Annales Henri Poincaré
Di, Jie & Wei - Cartan decomposition of a two-qutrit gate, 2008 Sci China Ser G, KAK & $\mathrm{SWAP}_3$
Di & Wei - Elementary gates of ternary quantum logic circuit, 2012 arXiv, TCX & TCZ
Di & Wei - Synthesis of multivalued quantum logic circuits by elementary gates, 2013 PRA, GCX & QSD
Di & Wei - Optimal synthesis of multivalued quantum circuits, 2015 PRA, GCX & QSD_optimal
Li, Gu, et al. - Efficient universal quantum computation with auxiliary Hilbert space, 2013 PRA, ququart CDNOT
Luo, Chen, Yang & Wang - Geometry of Quantum Computation with Qudits, 2014 Sci Rep
Luo & Wang - Universal quantum computation with qudits, 2014 Sci China Phys Mech Astron, $C_2[R_d]$ 
Wang, Hu, Sanders & Kais - Qudits and High-Dimensional Quantum Computing, 2020 Front Phys, $C_2[R_d]$ 
Jiang, Wei, Song & Hua - Synthesis and upper bound of Schmidt rank of the bipartite controlled-unitary gates, 2022 arXiv, $\mathbb{C}^M\otimes\mathbb{C}^N$ & GCX
Jiang, Liu & Wei - Optimal synthesis of general multi-qutrit quantum computation, 2023 arXiv, GCX & CINC
Zi, Li & Sun - Optimal Synthesis of Multi-Controlled Qudit Gate, 2023 arXiv, ancilla $\ket{0}\text{-}U$ 
Fischer, Tavernelli, et al. - Universal Qudit Gate Synthesis for Transmons, 2023 PRX Quantum, $C^m[U]$ 