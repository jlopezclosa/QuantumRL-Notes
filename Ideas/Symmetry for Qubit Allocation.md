In this section, we define a circuit as an ordered set of graphs: 
$$\mathcal{C} = (G_{1}, G_{2}, \dots, G_{T})$$
Where each graph $G_t = (V, E_{t})$ is the interaction graph at slice $t$, with the vertex set $V$ corresponding to logical qubits, and the egde set $E_t$ corresponding to gates. 

We treat each slice as a graph. There is a possible extension in defining a quantum circuit as a spatio-temporal graph, in which nodes represent qubits at specific time steps $(q_i, t)$ and edges represent both gates and temporal flow

## Cyclic permutation (qubits)

We can rotate all qubit indices by $n$ and the circuit structure and the possible hardware allocations are essentially the same: the problem is invariant under the action of the cyclic group $C_n$. 

The image below is an example of $C_n$, where the next mapping has been applied:
$$\{q_{0} \to q_{1}, q_{1} \to q_{1}, q_{2} \to q_{3}, q_{3}\to q_{0}\}$$
We say that the cost of the two solutions found by the method should be the same (invariant), but the action (mapping to cores) should be equivariant. 

![[graph_automorphism.png|400]]

There is another way to formulate this, although perhaps less convenient. Qubits are not relabeled, but the edges of the graph are transformed, such that the connectivity pattern is shifted according to the action of $C_n$. The nodes (qubits) then keep their original labels, but the relationships between them are updated to reflect the group action. In the example above, each edge $(q_i, q_j)$ is mapped to $(q_{i+1 \ \text{mod}\ n}, q_{j+1 \ \text{mod}\ n})$. 

We call a **graph automorphism** a permutation of the vertices of a graphs that preserves adjency. Thus, if there is an edge between $u$ and $v$, after applying the automorphism there is still and edge between $\pi(u)$ and $\pi(v)$. 

> An automorphism of a graph $G = (V, E)$ is a bijection $\pi: V \to V$ such that $(u, v) \in E ⟺ (\pi(u), \pi(v))  \in E$

This formulation might be helpful in building equivariant networks and augmenting data. 

## Slice / circuit repetition (temporal symmetry)

In quantum circuits, there can be structural or repeated patterns that make some slices symmetric under certain group actions; for instance, repeating layers or identical blocks of gates applied at different times. Rearranging these slices shouldn't change the essential behaviour of the circuit. 

An example: a circuit can apply the same entangling operation 10 times in sequence. 
## Solution symmetries

### Core permutation symmetry

If all quantum cores are identical (all interconnected, same gate-set, and same SWAP cost), permuting them changes nothing. Therefore, the solution could also take this into account. Imagine the next qubit assignment:
$$C_{1} = \{q_{1}, q_{2}\}, C_{2} = \{q_{3}, q_{4}\}$$
This assignment is the same as $C_{1} = \{q_{3}, q_{4}\}, C_{2} = \{q_{1}, q_{2}\}$. Therefore, they should be Q-invariant. 
However, it's not clear that we can make all these assumptions. 
### Swap symmetry

A particular solution symmetry regards the SWAPS: $Q(\text{swap}(q_{1}, q_{2}) = Q(\text{swap}(q_{2}, q_{1})$. Namely, the Q-value of the 2 possible swaps combining a pair of gates should be invariant to the action of $C_2$. 

# Equivariance analysis of [[Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]]

GATs are, in theory, inherently equivariant to node permutations / graph automorphisms. Since cyclic permutation is a special case of node permutation, this symmetry should be guaranteed as well. 
However,  in practice, equivariance is only preserved if the node and edge features don't break the symmetry. 

## Initial embeddings

Initial embeddings contain trainable logic qubit embeddings $E^{(Q)}\in \mathbb{R}^{Q\times d_E}$
$$
\mathbf{H}_t^{(I,Q)} = \mathbf{\tilde{D}}_t^{-\frac{1}{2}} \mathbf{\tilde{Z}}_t \mathbf{\tilde{D}}_t^{-\frac{1}{2}} \mathbf{E}^{(Q)} \mathbf{W}^{(I)}
$$
This allows to obtain qubit-wise embeddings (check Figure 4), which are then MaxPooled. Note that $E^{(Q)}$ is a learned parameter matrix with **a fixed row for each qubit index**. This means that the embeddings are tied to the qubit indices (e.g. qubit 1 always has embedding $E_1^{Q}$).  Therefore, when relabeling qubits by permutating their indices, this matrix **doesn't permute accordingly**. 

Mathematically, imagine we have a permutation matrix $\mathbf{P}$ with which we permute qubits. Then, since the rows of $\mathbf{E}^{(Q)}$ are fixed, $\mathbf{E}^{(Q)}$ doesn't transform as $\mathbf{P}\mathbf{E}^{(Q)}$. It remains fixed as a lookup table.

### Positional encoding

 Positional encoding is added to $E^{(Q)}$ and $\mathbf{H}_t^{(I)}$ is finally obtained. Note that this embedding is standard sinusoidal encoding applied at the slice level, not at the qubit level. Therefore, it doesn't break equivariance at the qubit level; but does break slice equivariance. The authors acknowledge this problem in the future work section of the paper:
 
  > Additionally, the policy could be enhanced by introducing techniques such as relative positional encoding of slices, taking into consideration the **intrinsic symmetry of quantum circuits**.
  
An intuition: the fact that a slice follows a similar one, or is followed by another of the same kind, might be more relevant than the absolute position of the slice!
## Decoder

The qubit representation $E^{(Q)}_q$ changes at each decoding step, because they use only the representation of the qubit being allocated. It's easy to see that this will also break equivariance in the decoder. 

# Breaking symmetries

Here, we define breaking symmetries as imposing restrictions on the solution / using heuristics that use the existence of a symmetry to break it, thus reducing the action space. Some examples are:
### Breaking core-permutation symmetry

We could:
- Always assign $q_{0}$ to $c_{0}$
- Always select $q_0$ for the first unassigned bit
### Breaking swap symmetry

We could constrain the action space such that only swaps from the qubit with the lowest index to the qubit with the highest indices are allowed. Formally, we enforce that $\text{swap}(q_{i}, q_{j})$ is only valid if $i < j$. This would prevent redundant solutions. 



