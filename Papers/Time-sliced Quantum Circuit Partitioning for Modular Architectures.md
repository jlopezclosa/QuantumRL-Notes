# Problem formulation

> [!NOTE] Problem definition
> A graph partitioning problem with a goal of minimizing edge crossings between partitions. 

A sequence of assignments, one for each time slice in the circuit, is found. 

![[fgpoee_directedgraph.png|300]]

# Method

## Graph partitioning

- **Input**: The quantum circuit
- **Output**: A path, i.e. a sequence of assignments of the qubits with the condition that every partitioning in the sequence is valid

A _valid_ assignment is defined as an assignment where each pair of interacting qubits in a time slice are located within the same partition. 
The _non-local communication_ between consecutive assignments is defined as the total number of operations which must be executed to transition the system from the first assignment to the second assignment.

### The directed graph

A directed graph _(between 2 consecutive assignments)_ is constructed where:
- The **nodes** are the partitions (cores)
- The edges indicate a qubit moving from partition $i$ to partition $j$. 
- Multiple edges between the same pair of partitions are allowed because multiple qubits can move the same way in the same step.

1. All 2-cycles (A $\rightarrow$ B, B $\rightarrow$ A) are extracted from this graph and those edges are removed from the graph. $c_{k}$ is the number of $k$-cycles extracted.
2. Then, 3 cycles are extracted, and os on and so on. Again, we record $c_{k}$.
3. When there are no cycles remaining, the total number of remaining edges is $r$, and the total communication overhead $C$ is given by
$$ C = r + \sum_{k=2}^{n} (k-1) \cdot c_k $$
A $k$-cycle can be executed in $k-1$ swap operations instead of $k$ separate moves.
The remaining edges after cycle removal are unavoidable direct moves.
### FGP: Fine-Grained Partitioning

FGP uses lookahead weights to capture more structure in the circuit than the default time slice graphs. To construct the lookahead graph at time $t$, we begin with the original time slice graph and give the edges present infinite weight. For every pair of qubits we add the weight
$$ w_t(q_i, q_j) = \sum_{t<m\leq T} I(m, q_i, q_j) \cdot D(m-t) $$
to their edge, where:
- $D$ is some monotonically decreasing, non-negative function - the lookahead function,
- $I(m,q_{i}, q_{j})$ is an indicator that is 1 if $q_{i}$ and $q_{j}$ interact in the time slice $m$ and 0 otherwise
- $T$ is the number of time slices in the circuit

![[fgpoee_lookageadweights.png|300]]
### OEE algorithm

Iteratively, the next assignment is found with a partitioning algorithm, seeded with the assignment obtained from the previous time slice. The first can choose a seed randomly or use the static assignment. 

The partitioning algorithm is called Overall Extreme Exchange and:
- Finds a sequence of pairs of vertices to exchange
- Makes as many changes exchanges as give it an overall benefit

However, OEE over corrects:
_If a qubit needs to interact in another partition, then it can “drag along” a qubit it is about to interact with because OEE attempts to minimize weight between partitions regardless of its relation to the previous or next time slice graphs_.

#### Relaxed OEE

It's a relaxed version of OEE - more generally speaking, a partitioning algorithm using the KL heuristic. 

> [!NOTE] The KL heuristic...
> selects pairs of vertices in a graph to exchange between partitions based on the weights between the vertices themselves and the total weight between the vertices and the partitions

- OEE is ran until the partition is valid for the time slice (all interacting qubits are in the same partition) and then make no more exchanges
- This also speeds up OEE

### Lookahead functions

Several lookahead functions are tested, but the exponential decay is the best one in circuits with lots of local structure. In random circuits, there is not much difference. However, in their benchmarks the exponential decay with $\sigma=1$ is used - a small $\sigma$ is better for circuits with very little structure - that is, since they give functions which decay quickly even for small $n$.

The function is $D(n)=2^{-n/\sigma}$. When $\sigma \leq 1$, any interaction will always have a weight at least as high as the sum of interactions after it. 

![[fgpoee_lookaheadfunctions.png|300]]
# Assumptions

- The main simplification of the paper is that _since non-local communication is dominant, they focus only on non-local costs._
- Latency is approximated by the total number of times qubits must be shuttled between different regions of the device
- Full connectivity between cores is assumed. 

# Evaluation

- **Width of the circuit**: the total number of qubits used.
- **Depth of the circuit**: total number of time slices required to execute the program. Also referred to as runtime. Qubit movement operations which are inserted in order to move interacting qubits into the same partition contribute to the overall depth of the circuit.

# Results

![[fgpoee_resultscomparison.png|500]]
# Appendix: example

Ex,aple with  6 qubits and 3 partitions (A, B, C).  
Two consecutive assignments (before and after one time slice):

**At time t**
- A: {q0, q1, q5}
- B: {q2}
- C: {q3, q4}

**At time t+1**
- A: {q1, q3}
- B: {q0}
- C: {q2, q4, q5}

First, we build the directed multigraph. We make nodes = {A, B, C}.  Then we add one directed edge for each qubit that changes partition:

- q0: A → B
- q2: B → C
- q3: C → A
- q5: A → C  

So the edge multiset is: {A→B, B→C, C→A, A→C}. Then, we extract the cycles: 

- There’s a **3-cycle**: A→B→C→A (edges: A→B, B→C, C→A).  We remove it and record **c₃ = 1**.

Remaining edges: {A→C}. There are no more cycles. Thus,  we have **r = 1** edge left (A→C).  This represents a move that can’t be paired with others (swap with an empty slot). We use the previous formula and get that $C=1  +  (3−1)⋅1  =  1+2  =  3$. Thus, the 3-cycle (A→B→C→A) can be done with 2 swaps instead of 3 separate moves. The leftover A→C move has no partner and costs 1 on its own.

# Appendix: OEE

## Problem  setup

We have:
- An **undirected weighted graph** $G$ with $n$ vertices.
- We want to split the vertices into $k$ equal-size partitions (clusters).
- The **goal**: Minimize the cut cost i.e., the sum of weights of edges crossing between partitions.
## Kernighan–Lin (KL) background

The classic KL heuristic works like this for _two-way_ partitioning:

1. **Locking passes**:
    - Start with the current partitioning.
    - Repeatedly choose the _best single swap_ of a vertex from partition A with one from partition B that improves the cut cost the most.
    - Lock those two vertices (so they’re not moved again this pass) and update scores.
    - Keep going until all vertices are locked.
2. **Apply best prefix**:
    - Keep track of the cumulative gain after each swap in the pass.
    - Only commit to the prefix of swaps that gave the highest total gain.

## OEE Extension

OEE generalizes KL in two ways:
- **Multiple partitions** (k-way, not just 2-way).
- **Extreme exchanges**: In each pass, OEE still builds a _sequence_ of swaps (two vertices in different partitions) but searches across **all pairs of partitions** to find the single best pair at each step.
### Step-by-step OEE (per pass)

**Inputs**:
- Graph $G$
- Current partitioning (balanced: $n/k$ per partition)

**Data structures**:

- $W[i,l]$: Sum of weights from vertex $i$ to all vertices in cluster $l$
- $D[i,l]$: Gain if vertex $i$ were moved to cluster $l$, computed as:  $$D[i,l]=W[i,l]−W[i, \text{current\_cluster}(i)]$$
- Unlocked set: vertices not yet moved in this pass.

**Algorithm**: 

1. Initialize **W**, **D** from the current partitioning.
2. Repeat until all vertices locked:
	* Consider every ordered pair of distinct clusters (**A**, **B**).
	* For each unlocked $i \in A$ and $j \in B$, compute the exchange gain: $$
		g(i,j) = D[i,B] + D[j,A] - 2 \cdot w_{ij}$$
		This says:
		* $D[i, B]$: benefit of moving i to B
		* $D[j, A]$: benefit of moving j to A
		* Subtract $2w_{ij}$ because the i-j edge is counted twice and will flip from cut→internal or vice versa.
	* Choose the pair with maximum $g(i,j)$.
	* Record the gain and the pair.
	* Lock $i$ and $j$ for the rest of this pass.
	* Temporarily swap them in the working partitioning, and update **W** and **D** efficiently.

3. Best prefix rule:
After the pass, sum gains cumulatively; pick the prefix of swaps with the highest total gain > 0.

4. Commit those swaps to the **real** partitioning.
