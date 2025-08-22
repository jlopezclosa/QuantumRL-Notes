# Problem formulation

### Constraints

- Friends qubit (involved in the same gate) are allocated in the same core
- The amount of logical qubits allocated in each core cannot exceed its capacity

We want to minimize the **number of intercore communications needed to move the logical qubits from the core in which they are allocated in a time slice, to the cores where they are allocated in the next slice**.

### Combinatorial optimization problem

$$
\begin{array}{lll}
\min _x & \sum_{t=1}^{T-1} \sum_{q=1}^Q \sum_{s=1}^C \sum_{d=1}^C x_{t, q, s} x_{t+1, q, d} D_{s, d} \\
\text { s.t. } & x \in\{0,1\} & \\
& \sum_{c=1}^C x_{t, q, c}=1 & \forall q \quad \forall t \\
& \sum_{q=1}^Q x_{t, q, c} \leq P_c & \forall t \\
& \sum_{c=1}^C x_{t, g_1, c} x_{t, g_2, c} \geq 1 & \forall\left(g_1, g_2\right) \in G_t
\end{array}
$$
- where $x$ are the binary decision variables,
- $x_{t, q, c}=1$ if at timestep $t$ the $q$-th logical qubit is allocated in the $c$-th core, 
-  $T$ is the number of time slices of the circuit,
- $Q$ is the number of logical qubits, 
- $C$ is the number of cores,
- $D_{s, d}$ is the state transfer cost or distance between $s$-th and $d$-th cores,
- $P_c$ is the capacity of the $c$-th core, 
- $G_t$ is the set of logical qubits pairs representing 2-qubit gates in the $t$-th slice

Explanation:
- The first equation constraints each logical qubit to be allocated in exactly one core in each time slice
- The second takes into account core capacities
- The third enforces in each slice and for each gate that qubits involved are allocated in the same core
- The objective is the amount of inter-core communications

Note that the space of solutions is very high: $2^{TQC}$, e.g. about $10^{9030}$  when mapping a 30-slice circuit of 100 qubits on a 10-core architecture

# Environment definition

### State
Circuit context, represented as a sequence of time slices, embeddings, etc. + partial solution so far (explained below). 
### Action
Choosing a core $c \in \{1, ..., C\}$ for logical qubit $q$ in time slice $t$
### Transition
After choosing an action (allocating the current qubit to a core), the environment transitions to:
* Update the allocation vector for the current slice.
* Update the core capacities for the slice.
* Prepare the state for the next decoding step ($t, q+1$ or $t+1, 1$), possibly updating constraints (e.g., reserved spaces for friends).
### Reward
The total number of inter-core communications needed to bring a qubit in its assigned core from the core where it was allocated in the previous circuit slice:
$$
R(A) = \sum_{t=2}^{T} \sum_{q=1}^{Q} D_{A_{t-1,q}, A_{t,q}}
$$
where $D$ is the distance matrix and $A_{t,q}$ is the core (action) selected for the $q$-th qubit at the $t$-th slice. 

### Episode
An episode is one complete allocation for all qubits in all slices. 

# Method

They use an [[RL methods for CO and qubit allocation#Autoregressive methods|autoregressive method with a transformer architecture]]. To represent the circuit, [[Problem representation#GNN|GNN]] are used. 

The first step of the autoregressive policy is obtaining a learned representation of the input circuit. This way, instead of just considering a representation of the current state (result of past state and past actions) like a normal MDP, we can **first build a global representation of the input and then sequentially consider contexts that focus on a part of it**. 

### Initial embedding

- **Input**: sequence of $T$ circuit slices (where qubit pairs are defined as in [[Quantum circuit slicing]])
- **Output**: Initial embeddings embeddings $\mathbf{H}_t^{(I)} \in \mathbb{R}^{d_E}$ for each slice $t$ where $d_E$ is the size of such embedding. 
- **Purpose**: encode each slice into a fixed-size vector representation that captures the structure and interactions of the cubits within that slice. 

In fact, we can consider $G_t$ as the set of a unidercted disconnected graph $Z_t$. Therefore, **each slice is encoded as a GNN**. 

The GNN produces a quibit-wise representation  $\mathbf{H}_t^{(I, Q)}$ such that:
$$
   H^{(I,Q)}_t = \tilde{D}^{-1/2}_t \tilde{Z}_t \tilde{D}^{-1/2}_t E^{(Q)} W^{(I)}
   $$
   * $E^{(Q)}$ are *trainable embeddings* for each logical qubit.
   * $\tilde{Z}_t$ is the adjacency matrix of the interaction graph with self-loops.
   * $\tilde{D}_t$ is the corresponding degree matrix.
   * $W^{(I)}$ is a trainable weight matrix.

Then, we apply a **max pooling** across the qubit dimension $Q$, and sinusoidal positional encoding is applied. 

![[kool_initial_embedding.png|400]]

####  In the code... 

```python unwrap:false fold

class CircuitSliceEncoder(nn.Module):
  ''' Given a circuit returns the slice embeddings H^(S) and circuit embeddings H^(X).

  This class implements section III B 1 & 2 from Ref. [1].

  Args:
    - emb_shape: shape of the time slice embeddings, d_E in Ref. [1].
    - num_enc_transf: number of transformer blocks for the encoder, b in Ref. [1].
    - num_enc_transf_heads: number of transformer heads per block in the encoder, b in Ref. [1].
  '''

  @staticmethod
  def getPositionalEmbedding(T, d_model):
    position = torch.arange(T).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d_model]

  def __init__(self,
               emb_shape: int,
               num_enc_transf: int,
               num_enc_transf_heads: int
      ):
    super().__init__()
    self.emb_shape = emb_shape
    self.gnn = GNN(emb_shape=emb_shape)
    self.enc_transf = TransformerEncoder(num_layers=num_enc_transf,
                                         embed_dim=emb_shape,
                                         num_heads=num_enc_transf_heads,
                                         ff_hiden_dim=emb_shape,
                                         dropout=0.0)
  
  def forward(self, circuit_slices: Tuple[torch.Tensor], q_embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Args:
      - circuit_slices: adjacency matrix of gate connections for each time slice in the circuit.
      - q_embeddings shape: [Q, d_E] where Q = number of logical qubits, d_E slice emb. dim.
    Returns:
      - H_S: encoded slice embeddings of shape: [T, d_H] where T = num. slices and d_H = d_E
      - H_X: circuit representation of shape d_H = d_E
    '''
    device = next(self.parameters()).device
    T = len(circuit_slices)
    self.gnn.setGraphs(graphs=circuit_slices)
    # Section III. B.1 InitEmbedding
    Ht_IQ = self.gnn(q_embeddings) # H_t^{(I,Q)} shape = [T, Q, d_E]
    Ht_I, _ = torch.max(Ht_IQ, dim=1) # Max pool across qubit dimension, shape = [T, d_E]
    Ht_I += CircuitSliceEncoder.getPositionalEmbedding(T, self.emb_shape).to(device)
    # Section III. B.2 EncoderBlocks
    H_S = self.enc_transf(Ht_I)  # shape = [T, d_E] (we force d_E = d_H)
    H_X = torch.mean(H_S, dim=0) # Circuit embedding, shape = [d_E]
    return H_S, H_X
```

## Encoder 

- **Input**: Initial embeddings $\mathbf{H}^{(I)} \in \mathbb{R}^{T\times d_E}$
- **Output**: Embeddings $\mathbf{H}^{(I)} \in \mathbb{R}^{T\times d_H}$
- **Purpose**: Achieve a slice representation that takes into account relations with all the other slices in the input circuit - i.e., to incorporate in each slice embedding information from other relevant circuit slices. 

*Why*? Because, in selecting core for the qubits of a slice considering its embedding, the policy can be aware of the next circuit slices. 

In the first encoder, the MHA is applied in **self-attention mode** to the projections of the initial slice embeddings, incorporating in each slice embedding information from other *compatible* slices. Subsequent encoder blocks perform the same operation on the output of the previous encoder block.

Self-attention computes a _weighted sum_ of all other slice embeddings for each slice, using the learned attention weights. This means **each slice embedding gets updated** based not only on its own original information, but also based on the embeddings of all other slices. 

*Why self-attention?* Because all the circuit slice embeddings are input together to the transformer encoder - and they are the only ones to be used in computation. Remember:


> [!NOTE] > When each slice embedding is updated based on all other slice embeddings, it’s self-attention. Cross-attention only applies when you’re attending _across_ different sets of representations (for instance, in the decoder).


So, in summary, each slice's embedding incorporates information from the whole circuit. This is all "pre-decoding", before any specific allocation decisions are made. 

For more information on the different transformer blocks, see []. 

![[kool_encoder.png|400]]

### In the code...

```python fold
class TransformerEncoderBlock(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
    super().__init__()
    self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
    self.ff = nn.Sequential(nn.Linear(embed_dim, ff_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(ff_hidden_dim, embed_dim))
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: [T, B, E] where T = num_slices, B = batch, E = embed_dim
    attn_out, _ = self.attention(X, X, X)
    X = self.norm1(X + self.dropout(attn_out))
    ff_out = self.ff(X)
    return self.norm2(X + self.dropout(ff_out))


class TransformerEncoder(nn.Module):
  def __init__(self, num_layers: int, embed_dim: int, num_heads: int,
               ff_hiden_dim: int, dropout: float = 0.1):
    super().__init__()
    self.layers = nn.ModuleList([
      TransformerEncoderBlock(embed_dim, num_heads, ff_hiden_dim) for _ in range(num_layers)
    ])
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: [T, B, E] where T = num_slices, B = batch, E = embed_dim
    for layer in self.layers:
      X = layer(X)
    return X
```
## Core snapshot encoder

- **Input**: The output action of the previous slice $t-1$, a vector $A_{t-1}\in\{1, \dots, C\}^Q$, that is, the mapping between each logical qubit and the core index. 
- **Output**: Embeddings $\mathbf{H}^{(C)} \in \mathbb{R}^{C\times d_H}$
- **Purpose**: provide an encoding of the qubit allocations in the previous circuit time slice.

*Why?* Because we want the policy to also take into account the previous qubit allocations in the previous circuit time slice, possibly trying to make a compromise between the **state transfer distances from the previous slices** and the **qubit allocation for the next slices** still to be decided. In other words, we need awareness of where each qubit currently is and the cost of moving them. 

So: *given what has already been assigned, what does the current allocation look like?*

*Note*: These embeddings are calculated when moving from the decoding of one slice to the next!

Let's move on to the architecture. The input adjency matrix is fixed and represents the connectivity of the multi-core hardware architecture. We already have the embeddings of the logical qubits; max pooling is performed over the qubits allocated at each core. If a physical qubit is not associated to logical qubit, a padding embedding is considered. 

![[kool_snapshot_encoder.png|400]]

### In the code...

```python fold
class CoreSnapshotEncoder(nn.Module):
  ''' Given last core assignments and qubit embeddings generate an embedding of previous qubit allocations.

  This class implements section III C from Ref. [1].

  Args:
    - core_con: matrix of core connectivities of the architecture.
    - core_emb_shape: length of the output core embedding, d_H in Ref. [1].
  '''

  def __init__(self, core_con: torch.Tensor, core_emb_shape: int):
    super().__init__()
    self.n_cores = core_con.shape[0]
    self.padding_emb = nn.Parameter(torch.zeros((core_emb_shape,)))
    self.gnn = GNN(emb_shape=core_emb_shape)
    self.gnn.setGraphs(graphs=core_con.unsqueeze(0).float()) # Add batch dim at the front
  
  def forward(self, prev_assign: torch.Tensor, q_embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Args:
      - prev_assign: for each logical qubit, the core to which it has been mapped. Shape: [Q]
      - q_embeddings shape: [Q, d_E] where Q = number of logical qubits, d_E slice emb. dim.
    Returns:
      - Core embeddings of shape [C, d_H] where C = number of cores, d_H = embedding size.
    '''
    core_embs = []
    for C in range(self.n_cores):
      mask = (prev_assign == C)
      if mask.any():
        core_embs.append(q_embeddings[mask].max(dim=0)[0]) # Take max pool of all q embs. in core C
      else:
        core_embs.append(self.padding_emb) # If no qubits in core C append learnable padding emb.
    core_embs = torch.stack(core_embs).to(q_embeddings.device)
    return self.gnn(core_embs).squeeze() # Transform core embs through GNN and remove "batch" dim
```
## Decoder

- **Input**: circuit representation $\mathbf{H}^{(X)}$, $t$-th slice embedding $\mathbf{H}_{t}^{(S)}$, $q$-th qubit embedding $\mathbf{E}_{q}^{(Q)}$, core embeddings $\mathbf{H}_{t}^{(C)}$
- **Output**: (after all the decoding steps) action $A_{t}\in\{1, \dots, C\}^Q$. 
- **Purpose**: 

The decoding process is carried for $T \times Q$ decoding steps. *Why*? Because $T$ is the number of slices and $Q$ the number of llogical qubits, and we output one logical qubit at a time, for each time slice. 

The decoding process is 3-level hierarchical. Note that throughout the decoding process:
- the circuit representation remains the same
- the slice representation remains the same for $Q$ decoding steps
- the qubit representation changes at each decoding timestep ($t$, $q$)

![[kool_decoder.png|400]]

**Dynamic embedding**: for each logical qubit $q$ being allocated, the core embeddings are augmented with useful information: 
- **Current remaining capacity for each core**: decreased by 1 each time a qubit is allocated in a particular core. Reset to its number of physical qubits when the allocation of a new slice $t$ begins (every $Q$)
- **Distance of $q$ being allocated from each core**: computed considering the core where $q$ was allocated the previous slice and the architecture core distance matrix. This gives **hypotetical state transfer cost for qubit being mapped**
These information vectors are projected to embeddings with the same dimensionality $d_H$ and summed to $\mathbb{H}_t^{(C)}$. The output is $\mathbf{G}_{t, q}^{(C)}$. 

### Cross-attention MHA

First, obtain the query vectors:
$$
\mathbf{Q}_{t, q}=\operatorname{concat}\left(\mathbf{H}^{(X)}, \mathbf{H}_t^{(S)}, \mathbf{E}_q^{(Q)}\right) \mathbf{W}^{(V, G)}
$$
where $\mathbf{W}^{(V, G)}$ is the trainable weight matrix for query projection from the context. Then, the cross attention is performed **between the query and the core embeddings**, obtaining the attended query vector $\tilde{\mathbf{Q}}_{t, q}$, also known as glimpse:
$$
\begin{aligned}
\mathbf{K}_{t, q} & =\mathbf{G}_{t, q}^{(C)} \mathbf{W}^{(K, G)} \\
\mathbf{V}_{t, q} & =\mathbf{G}_{t, q}^{(C)} \mathbf{W}^{(V, G)} \\
\tilde{\mathbf{Q}}_{t, q} & =\operatorname{MHA}\left(\mathbf{Q}_{t, q}, \mathbf{K}_{t, q}, \mathbf{V}_{t, q}\right)
\end{aligned}
$$
where $\mathbf{W}^{(K, G)}$ and $\mathbf{W}^{(V, G)}$ are the trainable weights matrix projecting key and value vectors from the augmented core embeddings. 

Subsequently, compatibilities are calculated between the attended query and the core embeddings, similarly the classical attention function.
### Masked attention: how to respect the constraints

However, we need a means to mask "illegal" actions. Thus, compatibilities with cores on which mapping the current $q$ would result in an invalid solution are masked: 

$$
u_{t, q, c}= \begin{cases}-\infty, & \text { if masked } \\ \frac{\tilde{\mathbf{Q}}_{t, q} \mathbf{K}_{t, q}^{\top}}{\sqrt{d_K}}, & \text { otherwise }\end{cases}
$$

The resulting vector $\mathbf{U}_{t, q}$ can be interpreted as logarithm of probabilities (logits) and the final output action probability for the ( $t, q$ ) decoding timestep can be computed using the softmax function:
$$
p\left(a_{t, q}=c\right)=\frac{e^{u_{t, q, c}}}{\sum_{j=1}^C e^{u_{t, q, j}}} \quad \text { for } c=1,2, \ldots, C
$$

**So when do we mask the action**?

- Core capacity constraint: a core logit is masked if the current remaining capacity is 0
- Qubits friendship: if $q_{1}$ and $q_2$ need share a gate and $q_1$ is allocated first, all other cores are masked when assigning $q_2$. Moreover, when assigning $q_1$ the capacity is reduced by 2.
- When allocating a qubit, the remaining amount of interacting couples to be mapped $g$ is taken into account, and only cores in which allocating the qubit would result in $\sum_{c=1}^C|\text{capacity}_c/2|\geq g$ are not masked.

### In the code...

Beware! There are some differences. Mainly, for pairs of qubits in a gate, instead of allocating them separately with careful masking and state tracking, they are allocated in one step. 

```python fold
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
  ''' Decoder for the autoregressive qubit allocation core assignment.

  The decoding process should always be applied first for the qubits that belong to a gate in the
  given time slice. After that all other "lone" qubits can be allocated. Otherwise the impossible
  allocation condition described in the last paragraph of section III E from Ref. [1] 
  could happen.
  
  When using forward with qubits that do not belong to a quantum gate the qubit embedding Eq_Q is
  set to that individual qubit and double is to be set to False. When using forward with a pair
  of qubits that belong to a quantum gate double is to be set to True and the qubit embedding 
  must be a mix of the embeddings of both qubits (avg or mix, for example). The returned core 
  allocation is for both qubits in the gate.

  Args:
    - core_capacities: rank 1 tensor that contains the number of qubits that can be held per 
    core.
    - core_emb_size: length of the output core embedding, d_H in Ref. [1].
    - slice_emb_size: length of the slice embedding, d_E in Ref. [1].
  '''
  def __init__(self, core_capacities: torch.Tensor, core_emb_size: int, slice_emb_size: int):
    super().__init__()
    self.slice_emb_size = slice_emb_size
    self.capacity_emb = nn.Embedding(core_capacities.max().item()+1, core_emb_size)
    # For now distance is binary (0 disconnected / 1 connected), but as there might be two qubits
    # and the distances add up then the possible values are 0, 1 and 2
    self.dist_emb = nn.Embedding(3, core_emb_size)
    self.W_q = nn.Linear(3*slice_emb_size, core_emb_size, bias=False)
    self.W_k = nn.Linear(slice_emb_size, slice_emb_size, bias=False)
    self.W_v = nn.Linear(slice_emb_size, slice_emb_size, bias=False)
  
  def _getDynamicCoreEmbeddings(self, Ht_C: torch.Tensor, core_capacities: torch.Tensor,
                                distances: torch.Tensor) -> torch.Tensor:
    ''' Get the dynamic core embeddings G_{t,q}^{(C)}.
    '''
    return Ht_C + self.capacity_emb(core_capacities) + self.dist_emb(distances)
  
  def _getInvalidMask(self, core_capacities: torch.Tensor, double: bool) -> torch.Tensor:
    ''' Return True for cores where there is not enough space for allocating the qubit.
    '''
    if double:
      return (core_capacities < 2)
    return (core_capacities < 1)
  
  def forward(self, Ht_C: torch.Tensor, core_capacities: torch.Tensor, distances: torch.Tensor,
              H_X: torch.Tensor, Ht_S: torch.Tensor, Eq_Q: torch.Tensor, double: bool) -> torch.Tensor:
    '''
    Args:
      - Ht_C: core embeddings of shape: [C, d_H]
      - core_capacities: remaining unallocated qubits for each core. Shape: [C].
      - distances: distances from the previous allocation of qubit q to all other cores. Shape: [C].
      - H_X: circuit embedding. Shape: [d_H].
      - Ht_S: t-th circuit slice embedding. Shape: [d_H].
      - Eq_Q: q-th qubit embedding. Shape: [d_H].
      - double: wether this decoding corresponds to two qubits that act in a gate.
    Returns:
      - Vector with the probabilities of allocating qubit q to each core. Shape: [C].
    '''
    sqrt_dH = math.sqrt(self.slice_emb_size)
    Gtq_C = self._getDynamicCoreEmbeddings(Ht_C, core_capacities, distances) # [C, d_H]
    # Apply pointer attention mechanism
    context = torch.cat([H_X, Ht_S, Eq_Q], dim=-1) # [3*d_H]
    Q = self.W_q(context) # [d_H]
    K = self.W_k(Gtq_C)   # [C, d_H]
    V = self.W_v(Gtq_C)   # [C, d_H]
    attn_logits = torch.matmul(K,Q)/sqrt_dH # [C, d_H]x[d_H] -> [C]
    attn_weights = F.softmax(attn_logits, dim=0) # [C]
    glimpse = torch.matmul(attn_weights, V)      # [C]x[C, d_H] -> [d_H]
    # Compute scores and mask invalid qubits
    invalid_mask = self._getInvalidMask(core_capacities, double)
    u_tqc = torch.matmul(K, glimpse)/sqrt_dH # [C, d_H]x[d_H] -> [C]
    u_tqc = u_tqc.masked_fill(invalid_mask, float('-inf')) # [C]
    return F.softmax(u_tqc, dim=0) # [C]
```

# Results
# Notes

A policy trained on 100 qubits can still be used to allocate circuits with fewer qubits, but with some performance degradation as the circuits depth increases.

According to the paper, they trained the models on random circuit with a fixed amount of used logical qubits, namely 50 and 100. However, the trained models can be used to map circuits with a number of used qubits **less or equal the number on which they were trained**.

Thus, they note that training the policies with circuits with a variable number of used qubits might increase generalization capability. 

