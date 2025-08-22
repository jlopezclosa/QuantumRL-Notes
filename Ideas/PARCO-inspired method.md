In this document, I propose a mix of two methods:
- [[Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]]
- [[Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization]]
# Qubit allocation as a multi-agent problem

## Justification

Qubit allocation can be reframed to a multi-agent problem. In Russo's method, solution generation is slow: at each timeslice $t$, each qubit $q$ must choose a core $c$. If agents (qubits or cores) act concurrently, specially for large topologies and high number of qubits, we can make inference time much smaller. 

Moreover, we could argue that the qubit allocation problem naturally fits a multi-agent framework. 
For instance, in the TSP problem, there is but one decision to be made at each timestep: which city to travel to. _Only one route is being constructed_. Russo's method artificially divides the problem in $T \times Q$ steps - so sequential latency scales with this quantity. By doing so, it imposes an ordering in qubits that breaks solution invariance. _NOTE: By solution invariance here I mean the permutation symmetry of the slice-level decision._

Note that the benefit is not so obvious as in PARCO's considered problems (see Figure 1). In such settings, a single-agent autoregressive model is clearly inefficient because all agents wait for the first to finish its 'trajectory', and so on. In our problem, many choices compete for shared capacity. A **fixed (and arbitrary) ordering over qubits might bias outcomes** because later qubits may face full cores, thus reducing exploration of joint placements compared to a parallel move. Earlier decisions don't negotiate with later ones; these merely react. _NOTE: all this considering that qubits do indeed choose core according to global context, so one could argue there is indeed some form of "coordination"_.

![[parco_example.png|300]]

### Why use PARCO

Some reasons to use a variant of PARCO are:
- Generalizes to different number of agents $M$ number of nodes $N$ (**at inference time**)
- Can be trained with different $M$ and $N$ sizes. [[Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization|The proof is in the appendix (and the code)]]

However:
- $N$ and $M$ must be fixed within a single problem instance. 
- Variable $M$ implies that there is padding to ensure that action tensors have the same length across a batch
### Agents

The way I see it, we have three options.
- Gate agents: interacting pairs / friends in the current slice are the agents. Each pair chooses a core and consumes 2 capacity. Single qubits consume 1 capacity.
- Qubits
- Cores
#### Gate agents 

- **Action space**: Choosing a core $c \in \{1, ..., C\}$ for logical qubit $q$ in time slice $t$

Using gates as agents has a very clear advantage: it implicitly bakes the "friends must co-locate" constraint into the action itself, thus preventing many conflicts. There would still be conflicts, such as:
- Many pairs would choose the same core simultaneously, surpassing its capacity

In theory, a slice $t$ could be resolved in a single round. A round would then be a time-slice. This fits naturally with PARCO. 

The main disadvantage that makes this model virtually infeasible is that $M$ changes between slices - i.e. varies *within a single problem instance*. 

Another disadvantage is that the number of agents can be both very high ($Q-1$ in the worst case) and very variable between episodes (since it depends on $Q$ and the number of gates in the slice, which depends on the circuit "density"). Batching needs to handle a wide $M$ range.
#### Qubit agents

- **Action space**: Choosing a core $c \in \{1, ..., C\}$ for logical qubit $q$ in time slice $t$

Again, $M$ can be very high, but it's less variable. However, it requires more careful masking and more coordination logic. For instance, two friends qubits might select different cores at once (more on that later, for the core agent case). If we let the assignment for $t$ happen in a single round, then all conflicts that the coordination mechanism has failed to prevent would be resolved with the Priority-Based conflict handling - so special care should be put in its design.
#### Core agents

- **Action space**: Choosing a qubit $q \in \{1, ..., Q\}$ for logical qubit $q$ in time slice $t$, round $i$. 

Here, rounds $\neq$ time slices. We could model rounds as additional PARCO steps with stricter masks (for instance, cores at 0 capacity masked out). If two cores choose the same qubit, we would solve the conflict with 1) checking if one of said cores has already allocated its friend, 2) otherwise, run the Priority-based Conflict Handler. If two cores choose at once $q_i$ and $q_j$ and there is a gate $(i,j)$, we run the Priority-based Conflict Handler and let the winner keep one of the pair and book a reservation. 

A consideration is that, in real-world settings, $Q \gg C$. Therefore, the action space is bigger and solving the assignment is more difficult. 

# Case 1: qubits as agents

In this case, we have $M$ qubits and $N$ cores. 

Two ways to think about it:
- **Scenario 1**: Each round is a time slice $t$. We only update the context once an episode is finished. This means we must assign all qubits within one round, even if we need to overuse the priority-base conflict handler, potentially arriving to worse solutions. _Doubt: is there a risk that the model ends up relying too much on the conflict handler? 
- **Scenario 2**: One time slice $t$ can span multiple rounds. Every time a slice is finished, we call _reset()_ and update the context. Intuitively, this contradicts a bit the purpose of PARCO - episodes would be very short, not all qubits make the decision at the same time. As I understand it, we'd also need a per-slice reward. However, it's easier to enforce the constraints this way. 

I think the most natural approach in this setting is scenario 1. _However_, we could have a mix of both. For instance, don't force a time slice to be a single decision. In PARCO, only the winning agent acts, the others wait until the next round. We could do something similar. For instance, if two friends qubits choose different cores, let the winning core decide where both will be located. But imagine two other pairs choose that same core and there are only 4 spots there. Let the winner choose the core and the others wait until the next round - then we can update capacities, etc. 
## Context embedding

### Agent features

These features represent qubits. In **scenario 1**, there aren't very relevant features to include here. Perhaps we could add a qubit embedding based on:
- Interactions in the whole circuit (this would probably break temporal symmetry: it's not a relative measure). This feature could be the sum of the edges, or something more complex (an agent embedding reflecting the whole circuit). 
- Other properties
The current allocation is the same for all qubits: none. This is one of the reasons option 1 doesn't seem very useful here. If we were to somehow replicate Russo's method (i.e. we don't care about symmetry), we could use qubit learned embeddings for the qubits. 

In **scenario 2**, there would be more information to include. For instance, we could use feature nodes as lookahead weights (would break symmetry but not in long-term, since the weights "fade" in time). Or simply a summary of the current slice interactions like in Russo (we wouldn't have to MaxPool because we don't have to summarize the current slice). But what about long-term information? It's more difficult to 
### Core features

These features should contain information about the cores connectivity (in case we don't assume full connectivity - I personally wouldn't) and other information. Therefore, core features should include both node and edge features. 

In **scenario 1**, the context isn't refreshed until we finish a circuit. Therefore, we're only concerned in features like initial capacity and connectivity with the other cores, which doesn't change. 

In **scenario 2**, the context is refreshed each time slice. Therefore, features could include current capacity and qubits that are currently allocated there (how to do this? Adding an identifier for the qubits? This would break symmetry). Like in Russo, core embeddings could be computed as a maxpool the current embeddings of the qubits allocated there together with a core adjency matrix. Again, this would break equivariance. 
## Dynamic embedding

### Agents states

I've written an in-depth discussion about getting dynamical encodings for future qubit interactions in [[PARCO-inspired method#4. No notion about the future|the last section]] (applied to **scenario 1**). For scenario 2, we could include:
- Flag if it has been allocated
- Flag if it has qubit firiend 
- Flag if its friend has been allocated
- Perhaps a priority feature: qubits with pairs should have more priority in assignments, the others can redestribute. 
All these features would be included in the mixed method (let a circuit be a single problem instance, but don't force a round to be a single time slice: instead, update circuit embeddings once a full allocation is achieved satisfying the current constraints)
### Nodes states

Neither in **scenario 1** or **scenario 2** do we need to include core adjacency information, because it's fixed for the entire problem instance. However, in **scenario 2** and in the **mixed** one we do need to encode the current capacities. Moreover, since we are allowed to use masking, we can mask cores that are full or virtually full (i.e., can't accept a qubit and its friend). 
If we had qubit IDs, we could do like Russo et. al and provide a core embedding dependent on the qubits that are currently allocated on it. Perhaps we can still do the same if we:
- compute it with the future embedding of each agent (used in the agent states)
- compute it with a static qubit embedding representing the whole circuit - but doesn't help with temporal equivariance either. 
# Doubts and considerations
#### 1. PARCO's suitability

While PARCO significantly improves performance in multi-agent problems for CO (for instance HCVRP, where M
agents serve customer demands while adhering to heterogeneous vehicle capacity constraints). These problems are an ideal fit for PARCO because they are _true_ multi-agent settings for which autoregressive methods prove to be suboptimal. 
#### 2. Where are edge features?

PARCO doesn't use edge features. We should adapt it to work with graphs. 
#### 3. No penalization in the reward

The reward is directly the objective function. We need to think whether we should penalize the reward based on the illegal actions. For instance, if even with the conflict handler and masking, we cannot finish the allocation. 
#### 4. No notion about the future

PARCO has no notion about the future. Our problem has some sort of _planning_ , so we need some sort of representation of **future constraints** (here by constraints I mean which qubits should be together each timeslice).

An idea: whenever the context is renewed in **Scenario** 2, process each future graph with a GNN to produce agent embeddings for each future step. Then, aggregate the resulting embeddings for each agent. Perhaps we could do this with attention. This would be used as a feature vector in the agent context embedding. Different options for the aggregation method:
- Sum or mean (already invariant)
- Equivariant attention (perhaps overly complicated?). For instance, a TransformerConv layer. Perhaps more interesting because the near future is more important. 
In **Scenario 1**, this should be used in the dynamic embeddings. We would precompute embeddings of each time slice once for the whole circuit. Then, for each round (time-slice here) $i$, we would compute the aggregated embedding slices $t+1$ to $t+K$, where $K$ is a window of interest (an hyperparameter).  
A con in every set-up is that we would need positional encoding. Perhaps we could use a relative positional encoder?