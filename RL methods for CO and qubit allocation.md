# MARL

Learning from other intelligent agents in the environment

Generate **feasible** solutions step-by-step taking into account constraints -> generalize to new environments and learn more complex behavior. 
Multi-agent reinforcement learning for microprocessor design space exploration: many components of a microprocessor, each is assigned an independent agent. This works much better than a single-agent setting. 

May be difficult to actually discover the optimal solution - high sample complexity or failed training
Can generalize poorly to new distribution

How can MARL help? -> decentralized training to improve sample complexity
Adversarial training to improve robustness

MARL for sequential satellite assignment problems. POMDP

Traveling salesman problem: 

# Autoregressive methods



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