
# Motivation

In simple terms: the motivation is to enhance the sample efficiency of DRL methods for a broad spectrum of CO problems and methods. 

Note that in the constructive method, a policy generates a trajectory, and the reward is evaluated in a terminal state, i.e., episodic rewards. To enhance stability, CO methods typically perform many evaluations (baseline in REINFORCE). In practical scenarios, CO problems often involve computationally expensive objective functions to evaluate, (e.g., a black-box function), introducing significant restrictions on the problem-solving process.

This method leverages **solution symmetricity**. In DRL for CO, multiple action sequences (i.e., action trajectories) can be mapped to a single combinatorial solution, and the reward is defined on the terminal state (i.e., solution). Thus, while [[Sym-NCO - Leveraging Symmetricity for Neural Combinatorial Optimization|Sym-NCO]] takes into account of symmetries in the input space, this method uses symmetries in output (action trajectories) space. 

The training process can be decomposed in two steps:
- Reward maximizing: seeks high reward samples via RL
- Symmetric replay training: recycles the explored samples to explore symmetric regions *for free*. In other words, promotes exploration of the symmetric space by imitating the collected high-reward samples with symmetric transformation.

![[symreptrain-method.png|500]]

Note that the model isn't equivariant by construction. 

# Method

In summary, we:
- Run RL as usual (reward maximizing)
- Take high-reward trajectories, apply symmetry transformations to create new (but solution-equivalent) trajectories
- Add these transformed trajectories to the replay buffer and train with them using imitation learning

## Reward maximizing training


## Symmetric replay training


> [!NOTE] Symmetric trajectory policy
> The symmetric trajectory policy $p(\tau_{to x}|x)$ is a probabilistic distribution that samples symmetric trajectories $\tau_{to x}$, which have an identical terminal state (i.e., an identical solution $x$).

For instance, in the case of TSP with four cities, the sequences 1-2-3-4-1 and 3-4-1-2-3 represent the identical
Hamiltonian cycle $x$. Next we show possible choices of $p(\tau_{to x}|x)$.

### Maximum entropy transformation policy

### Adversarial transformation policy

### Importance sampling transformation policy