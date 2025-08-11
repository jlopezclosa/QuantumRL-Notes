

![[graphquantum_scheme.png]]
# Data

## Random circuit generation

The random circuit generator is adapted for curriculum learning. It selects $2g$ nodes to connect, where $g$ is the number of gates in the slice.

>[!NOTE] Limitation
>The random circuit generator does not ensure that at least one qubit involved in the gates of the current slice also appears in the subsequent slice. In a real partitioning scenario, such slices would typically be merged. 
# Environment

## Observation space

### Node features

- Current allocation: $C$ columns with one-hot encoding of the current allocation state
- Past allocation: $C$ columns with one-hot encoding of the allocation state from the previous timeslice $t-1$.
- Target qubit: flag indicating the qubit being allocated 
- _Optional_: additional column indicating interaction between the qubit $q_f$ being allocated and another qubit $q_c$ in the same slice: $X_{q_{c},2C+2}$ = $1\ \text{if}\ A_{1,f,c}=1\ \text{else}\ 0$

Node features vector: $X \in \mathbb{R}^{Q\times (2C+1)}$. $Q$ is rows, $(2C+1)$ is columns

*Drawbacks*: 
- Dependent on the number of cores

![[graphquantum_staterepresentation.png|500]]

### Edge features

Set $\mathbb{A} = [A_{1}, A_{2}$] where:
- $A_{1}$ is the current interaction: $A_{1,i,j}$ means that $q_i$, $q_j$ interact in the current slice
- $A_2$ is the lookahead function (as in FGP-rOEE). $\sigma=1$
These features can be squeezed into a single one by addition. However, this isn't used in the final solution.

### Core capacity

>[!NOTE] Important:
> Although the model is required to operate with a fixed number of cores, it can adapt
to different scenarios and core capacities

To encode core capacities, a vector $Z \in \mathbb{R}^C$, where $Z_c$ represents the remaining capacity of core $c \in C$. This is used to facilitate early-stage training (there's no ablation study done on this)

### Notes

Since the code is implemented in Stable Baselines 3 and OpenAI's gymnasium, which require fixed state and action space size, $Q_{\text{max}}=50$ and $A \in \mathbb{R}^{50\times 50}$. Since padding is needed for less qubits, the observation incorporates an integer $N$ representing the number of qubits in the current circuit. 

## Action space

The action space is $Y \in \mathbb{R}^C$. Each action corresponds to the selection of a core for the allocation of the current qubit. 

### Reward function

Since episodes have a fixed length, $\gamma=1$.  Thus, the reward is: $$r(s,a) = \alpha \cdot \text{nonlocal\_comm}(a,s)-\beta \cdot \text{intervention}(a,s) - \gamma \cdot \text{direct\_capacity\_violation}(a,s)$$Where:
- *nonlocal\_comm*: an action that differs from the core where the target qubit was allocated in $s$
- *intervention*: penalizes interventions
-  *direct\_capacity\_violation*: penalizes direct capacity violations

Note that $r(a,s)\leq 0$ so the cumulative reward is always negative. 

## Transition function

### Node features
  $$X_{q_f, C+a} = 1, \quad X_{q_f, 2C+1} = 0, \quad X_{q_f+1, 2a+1} = 1$$These updates achieve:
  - Allocating $q_f$ to the core represented by action $a$.
  - De-flagging the current qubit $q_f$.
  - Flagging the next qubit $q_{f+1}$.
  
  If the extended $X$ column is used: $$X_{q_c, 2C+2} = 0 \text{ where } A_{1,f,c} = 1, \quad X_{q_k, 2C+2} = 1 \text{ where } A_{1,f+1,k} = 1$$  These handle:
  - De-flagging $q_f$'s interacting qubit.
  - Flagging $q_{f+1}$'s interacting qubit.

### Core capacity

$$Z_{a} = Z_{a} -1$$
### Reservations


- Reservations $R$:
  $$R_c = a \text{ where } A_{1,f,c}=1, \quad c > f$$
  Reserves the core for future interactions involving the qubit just allocated, if applicable.
  
### Last slice

When the flagged qubit $q_f$ is the last in the slice ($f = |Q|-1$) any operations involving $q_{f+1}$ will be applied. In this case, the following adjustments are made:
  
- Core Capacity and Reservations: Both $Z$ and $R$ are reset to their initial states.

- Node Features $X$:
  $$X_{::0:C} = X_{::C:2C}, \quad X_{::C:2C} = 0$$
  The current allocation is copied to represent the past allocation, and the current allocation is reset.
# Evaluation

Experiments trained on circuits with $|Q| = 8$, $|C|=2$. 

# Method

## GNN Feature extractor

- **Input**: State
- **Output**: State abstraction
- **Structure**: GNN + pooling function + concatenation with additional components of the state
- Shared between the actor and the critic of the PPO agent, to reduce training time and to ensure a consistent state abstraction between both. 
### GNN

Can be either GCNN or GAT. 
- GCNN: two convolutional layers with hidden and output dimensions of 16 per node
- GAT: four attention heads
### Pooling method

See [[Pooling methods]]. 

In this case, the pooling function used is a simple of the final hidden states of all nodes, which are concatenated in any order (applicable to fixed-size graphs). In other words, as they say in the work:

> For this preliminary phase, the pooling method is a simple readout function. The final hidden states of the nodes are concatenated from 0 to $|Q|$ to form a $\mathbb{R}^{16|Q|}$ vector. This vector is then concatenated with the $C$-dimensional vector containing the remaining core capacities, yielding the final state abstraction. This abstraction serves as the pseudo-input to the PPO agent.

## PPO

They use Stable Baseline 3's PPO. 
## Curriculum learning

![[graphquantum_scheme.png]]

# Assumptions

- Uniform connectivity between cores
- Presence of all intra-core connections
- Movements between certain cores are never constrained
- Specific positions inside the core aren't considered: non-local communications are multiple scales of magnitude more expensive than those intra-core, so the latter are neglected.