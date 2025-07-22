# MARL

# Autoregressive methods

## Combinatorial optimization Markov Decision Process

Definition extracted from [[Sym-NCO - Leveraging Symmetricity for Neural Combinatorial Optimization|Sym-NCO]]

A CO-MDP is defined as the sequential construction of a solution of COP. For a given problem instance $\mathbf{P}$, the components of the corresponding CO-MDP are as follows: 

* **State.** The state $s_t = (\mathbf{a}_{1:t}, \mathbf{x}, \mathbf{f})$ is the $t$-th (partially complete) solution, where $\mathbf{a}_{1:t}$ represents the previously selected nodes. The initial and terminal states $s_0$ and $s_T$ are equivalent to the empty and completed solution, respectively. In this paper, we denote the solution $\pi(P)$ as the completed solution. 
* **Action.** The action $a_t$ is the selection of a node from the unvisited nodes (i.e., $a_t \in \mathcal{A}_t = \{\{1,...,N\} \setminus \{\mathbf{a}_{1:t-1}\}\})$.
* **Reward.** The reward function $R(\pi; P)$ maps the objective value from given $\pi$ of problem $P$. 
## Formulation

The problem $\textbf{x}$ is first encoded using a **trainable** encoder $f_\theta$, obtaining an encoded representation $\textbf{h}$: 
$$
\mathbf{h}=f_\theta(\mathbf{x})
$$
then, the **trained policy decoder** $g_\theta$ outputs the best action probability distribution at each timestep $t$, based on the **encoded representation of the problem and the current partial solution** (result of the past actions), formally we express this as follows:
$$
a_t \sim g_\theta\left(a_t \mid a_{t-1}, \ldots, a_0, \mathbf{h}\right)
$$
at each timestep we **select an action that keeps the solution valid**, eventually obtaining a complete solution. Thus, given the problem $\mathbf{x}$, the policy $\pi_\theta$ outputs a probability distribution for a solution a built in $T$ decoding steps:
$$
\pi_\theta(\mathbf{a} \mid \mathbf{x}) \triangleq \prod_{t=1}^T g_\theta\left(a_t \mid a_{t-1}, \ldots, a_0, f_\theta(\mathbf{x})\right)
$$