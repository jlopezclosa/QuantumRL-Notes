# Open issues

##### First, since we're not in a POMDP case, this formulation might be overly complicated.
Usually, simple MDPs are solved with single-agent policies. In the case of MARL, the natural choice is a [[MARL Concepts#Centralized training and execution (CTE)|CTE]] method. However this is problematic in our case because:
1) the solution would be dependent on the number of agents (either cores or agents)
2) the action space would grow exponentially with the number of agents
Perhaps I should do more research on CTE methods that overcome these challenges.
##### Could we simplify the algorithm considering that all agents can know the information?
The question is: how? The problem implicitly assumes POMDP. What allows it to be scalable is also what implicitly assumes partial observability.
##### Would representing the exact same information in the transformer mixer and the agent transformer be beneficial?
Probably not, so we should still represent $S_t$ with absolute observations, and $\mathbf{O}_{t}^a$ with relative values. Because each agent should reason only about itself, and the information would then be combined. 
##### TranfQMix "solves" the problem of coordination. But can we guarantee that no invalid actions will be taken? (this is extremely important in our case)

##### If TransfQMix takes longer at inference time than [[Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures|Attention-Based Deep RL]], then is it really useful?
This might not be the case since the method would generate an instant solution (all agents at once). However we'd need to be sure of it. 
# Definition of agents

There are two choices here:
- **Qubit agents**: at each $t$, each qubit agent selects one action (which core it will occupy)
- **Core agents**: at each $t$, each core selects actions about which qubits it wants to host
## Qubit agents

There are $Q$ agents. The global state is a graph of entities:
$$S_{t}=[\text{core}_{1}, \dots, \text{core}_{C}, \text{qubit}_{1}, \dots, \text{qubit}_{1}, \dots, \text{qubit}_{Q}]$$
There are two types of entities, cores and qubits. The nodes (features) for cores could be:
- Current occupancy (perhaps as a ratio)
- Core capacity. Dynamic, dependent on the core topology (do we ever want to change it?). Perhaps, on the contrary, including it enhances transferability (only if we train for more than one topology)?
- Transition costs to other cores. Again, a $C \times C$ matrix that introduces dependency on the number of cores. However, we probably assume uniform transfer costs. Should we include them then?

The nodes for the agents could be:
- Current core allocation
- An indicator if the qubit has gate interactions? For instance, it would be 0 if it's free. Perhaps it's redundant.
- Gates relations -  which qubits it shares gate with. At first glance, this might problematic to encode, because as the number of qubits increases, so does the state. Thus:
	- $z$ depends on the number of agents. Recall that TransfQmix is independent of the number of entities in the sense that it doesn't matter how many rows (entities) we add. However it should matter if we add or remove columns (change feature dimensionality)!
	- BUT! [[Quantum circuit slicing|Circuit slices]] ONLY contain gates that are not sharing any of the logical qubits they act on. So one qubit can be interacting at most with another qubit. So perhaps we can encode this feature as a categorial number, not with one-hot encoding. 
	
## Core agents

There are $C$ agents. 