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

### TransfQMix definition 

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
	
### Qubit agents with graphs

**Idea**: An agent's observation is a graph with all circuit interconnections in slice $t$. However, to make this a POMDP instance instead of a traditional MDP, each agent presents the circuit in a different way - i.e. permuted, so that the first qubit in the adjency matrix is the qubit agent. The full state contains the graph with all connections and the core assignation for each qubit, somehow. 

**Hypothesis**: This will preserve permutation equivariance. 

**Potential problems**: 
- How can the network know which qubit is which? Perhaps it's the most critical problem. If we use as a feature qubit IDs with one-hot encoding, the input has fixed size Q. So perhaps we should present each node with a trainable qubit embedding. 
	- Some ideas: hashing, use a shared gnn to compute qubit embeddings from the circuit graph, attention-based encoders, etc. 
- At which point do we consider temporal information? (suppose each agent is encoded with a RNN. But still). This would work better with GATs. 

#### Another proposal: mixing MARL with autoregressive methods for CO

Each qubit is an agent. Each agent has its own embedding, Q-network, and local observation.

Instead of all agents acting simultaneously, we maintain a queue of agents. At each time step, one agent selects its core assignment. Other agents are passive.

The active agent $i$ observes its local state, computed Q-values only over feasible cores, and the Q-value is passed to the QMIX mixer, which also sees the global state (the partial assignment). 
##### Qubit embedding

- **Input**: the embedding of its own qubit (from GNN over the current partial assignment graph), current status (assigned / not assigned), core capacities so far, masking due to constraints, current graph structure

Then, for all qubits, we run a GAT on the assignment graph. The active agent at step $t$ uses its embedding $h^i_t$, where $i$ is the active agent.
##### Temporal transformer

- **Input**: $[g^{(1)}, g^{(2)}, \dots, g^{(t)}]$ where $g^{(s)}$ is a pooled embedding - attention pool over node embeddings at slice $s$ 
##### QMixer

- **Global state**: The full partial assignment matrix (which qubits are assigned where, which cores are full, etc.), global circuit graph structure, current core capacities, etc. 

The Q-values of each agent are computed with a MLP with the next information concatenated: $h_i^t$, action $a$, and perhaps also the output of the temporal transformer (does this make sense?)

##### Pseudo algorithm

**Input:** Circuit graph, set of qubits $\{q_1, \dots, q_n\}$, set of cores $\{c_1, \dots, c_m\}$
For each circuit slice or time step $t$
    Compute node embeddings $\{h_1, \dots, h_n\}$ for all qubits using a GNN on current circuit graph and partial assignment
    Collect global embeddings $[g^{(1)}, \dots, g^{(t)}]$ (graph pooling)
    Compute global temporal context $G^t = \text{Transformer}([g^{(1)}, \dots, g^{(t)}])$
    Initialize *full_cores* $\gets \emptyset$, *unassigned_qubits* $\gets \{q_1, \dots, q_n\}$
    Optionally: randomize or symmetry-invariant ordering of _unassigned\_qubits_ (we can follow the Sym-NCO formula)
    For each $q_i$ in *unassigned_qubits*:
        Identify feasible cores for $q_i$ 
        For each feasible core $c_j$:
            Form assignment embedding $x_{ij} = [h_i \| \text{core\_feat}(c_j) \| (G^t)]$
            Compute $Q$-value: $Q_{ij} = \text{QNet}(x_{ij})$
	    Select $c_{j^*} = \arg\max Q_{ij}$ over feasible $j$ (with exploration)
        Assign $q_i \to c_{j^*}$
        Update core and qubit state information
        If core $c_{j^*}$ is now full
            Add $c_{j^*}$ to _full\_scores_
    Update Q-values and agent networks via VDN or QMIX (centralized training, with $G^t$ if available)
#### Two proposals to be refined

![[mario_proposal_1.png]]

Some potential problems: 
- Qubit $q_{0}$ always has priority over $q_1$: position bias. 
- Large action space, how can the Q-network scale?
- Many network evaluations: $O(n^2)$
- What about temporal relations? 
- Is making decisions sequentially the best (or only) approach? Does it make sense using MARL, then, where decisions are taken all at once?
- Can still lead to infeasible solutions

![[mario_proposal_2.png]] 

Potential pitfalls: 
- Broken constraints: the same discussed in the Attention paper. 

