# Motivation

TransfQMix is a [[MARL Concepts#Centralized-training decentralized-execution (CTDE)|CTDE]] method that builds on [[MARL Concepts#QMIX|QMIX]]. It attempts to address the problem of key information loss due to the representation of the individual agents, where a observation vectors are large concatenations of various types of features. 

TransQMix considers:
- The structure of the observation space
- The architecture used to deploy the agents and the centralized functions (CFs)
Rather than relaxing the monotonicity constraint like other methods (QPlex, CWQMix)

Thus, instead of chaining together many features to generate a vector that describes
the state of the world observed by the agent, this method **generalizes a set of features** and uses them to **describe the state** of the *entities* observed by the agent (or the CFs).

![[transqmix_features.png|400]]

# Method

![[transqmix_method.png]]
## Graph observations and state

### Graph observation

- **Entities** ($k$): Any type of information channel: information coming from agent's own senses, messages from other agents or information relative to any kind of object in the environment. 
- **Features** ($z$): A set of $z$ features defines an entity. Features can take different values for each agent. Some examples of features are map location, velocity, remaining life points, etc. 

Thus: 
$$ ent^a_{i,t} = [f_1, \ldots, f_z]^a_{i,t}$$

defines the entity $i$ as it is observed by the agent $a$ at the time step $t$. 

Traditional observation vectors are replaced with observation matrices with dimensions $k \times z$ which includes all the $k$ entities observed by an agent $a$ at $t$:

$$ O_t^a = \begin{bmatrix} ent^1 \\ \vdots \\ ent^k \end{bmatrix}_t = \begin{bmatrix} f_{1,1} & \cdots & f_{1,z} \\ \vdots & \ddots & \vdots \\ f_{k,1} & \cdots & f_{k,z} \end{bmatrix}_t $$

This structure allows the agents to **process the features of the same type using the same weights** of a parameterized matrix $\text{Emb}^a$ with shape $z \times h$, where $h$ is an embedding dimension. 

The resulting matrix $E_t^a = O_t^a \text{Emb}^a$ is formed by vertices embeddings $[e_1, \ldots, e_k]^T$ that will be further processed by transformers (see Figure 1). Thus, for each row (entity), we multiply its feature vector (size $z$) by the same weight matrix $\mathbf{Emb}_{a}$. 

Notice that $\text{Emb}^a$ is independent from $k$. Conversely, the encoding feed-forward layer used by RNN agents has approximately $k \times z \times h$ parameters.

All this is useful because:
- Shared weights are used to process all entities/features of the same type
- It's more scalable (doesn't depend on $k$)
- Transformers can then let the model learn relations between these entities

To summarize:
- Each entity is a vertex, or node. In code, each row of the observation matrix is a vertex's feature network.
- Edges aren't hard-coded, it's the transformer's self-attention mechanism that implicitly learns the relationships or edges (see Figure 1)

#### Problem: loss of positional information

A consequence of this representation is that the network can no longer learn to associate certain positions in the vector with certain entities, due to our $k \times z$ matrix and using the same weights (via shared embedding layer) for every row/entity.

This was an advantage of traditional encoders, where we'd have a vector like ``[own_x, own_y, other1_x, other1_y, ...]``. Thus, **positional information was implicitly present**. 

**Solution**: add two additional binary features: 
- **IS_SELF**: informs by if the described entity is the agent to which the observation matrix belongs
- **IS_AGENT**: tells if the entity described is a cooperative agent or not:

$$ f^a_{i,\text{IS\_SELF}} = \begin{cases} 1, & \text{if } i = a \\ 0, & \text{otherwise.} \end{cases} $$
$$ f^a_{i,\text{IS\_AGENT}} = \begin{cases} 1, & \text{if } i \in A \\ 0, & \text{otherwise} \end{cases} $$
### State vector

The state vector defines the vertex features for all entities from a global point of view:
$$ S_t = \begin{bmatrix} ent_1 \\ \vdots \\ ent_k \end{bmatrix}_t = \begin{bmatrix} f_{1,1} & \cdots & f_{1,z} \\ \vdots & \ddots & \vdots \\ f_{k,1} & \cdots & f_{k,z} \end{bmatrix}_t $$
where $S_{t}$ has dimensions $k \times z$. 

Notice that adding IS_SELF to $S$ doesn't make sense, since the features aren't defined in respect to any agent. However, it does make sense to include IS_AGENT. 

$\mathbf{Emb}^S$ is a weight matrix that converts a raw feature vector (size $z$) - which is an entity- into an embedding, such that we get $E_{t}= [e_{1}, \dots, e_{k}]^T_{t} = S_{t}\mathbf{Emb}_{S}$. This is used in the Transformer mixer. 
### An example

We illustrate both graph observations and state with an example in Spread. 

**Agent observation matrix:** Observations are converted into a matrix where each row represents an entity (agent or landmark) described by its **relative position** to the agent and two binary features: IS_SELF (is it the observing agent?) and IS_AGENT (is it an agent or not).

- Each entity’s features: $[pos_x, pos_y, IS\_SELF, IS\_AGENT]$

**State matrix:** The global state is built by concatenating all agent observations, but only features that exist for all entities are kept: **absolute positions**, **velocities**, and IS_AGENT.

- Each entity’s state features: $[\hat{pos}_x, \hat{pos}_y, v_x, v_y, IS\_AGENT]$
- For landmarks, velocity entries are zero.

## Transformer Agent

- **Input**: $E_t^a = [e_1, \ldots, e_k]_t^T$ plus a hidden vector $h_{t-1}^a$, which has the same size of any vector $e_i^a$. The final input matrix at step $t$ is $X_t^a = [h_{t-1}^a, e_{1,t}^a, \ldots, e_{k,t}^a]^T$.
- **Output**: $Q_a(\tau^a, \cdot)$

Note that the output of $l$ transformer blocks: $\tilde{X}_t^a = \text{MultiHeadSelfAttn}(X_t^a)$ is a refined graph in which all the nodes were altered based on the attention given to the others. In particular, $h_t^a = \tilde{h}_{t-1}^a$ can be considered as a transformation of the agent’s hidden state according to the attention given to the new state of the entities.

*Why is the hidden state special?* Just like the [CLS] token in BERT, the hidden state $h_{t}^a$ isn't tied to any physical entity; instead, it acts as a placeholder for the agent's own summary of information (past and present). 

> [!NOTE] About [CLS]...
> In BERT, the hidden state:
> - Is the only row that gets passed to the next time step recurrently (creating a memory effect)
> - Its output is inferred by all other words in the sentence, thus it contains all information in other words
> All other rows are essentially context. Therefore, the model learns to encode global information into the hidden state. 

In this case, we consider $h_{t}^a$ to **encode the general coordination reasoning of an agent**.

Therefore, the agent’s actions-values are sampled from $h^a_t$ using a feed-forward layer $W^u$ with dimensions $h \times u$, where $u$ is the number of actions: $Q_a(\tau^a, \cdot) = h_t^a W^u$. 

Finally, we pass $h^a_t$ to the next time step so that the agent can update its coordination reasoning recurrently.

### Decoupling mechanism

When some agent’s actions are directly related to some of the observed entities (e.g., “attack the enemy $i$” in StarCraft II), the transformer agents use a decoupling mechanism. 

*Why*? Because in this case, we want the Q-value for "act on entity $i$" to be computed based on information about that entity, not just the overall scene summary. 

Therefore, the action-values of the **entity-related** actions are derived from their respective entity embeddings.

So after the transformer, we have an updated embedding $\tilde{e}^a_{i,t}$ for each entity $i$. For each entity-targeted action, we use another feed-forward (linear) layer, $W^u_e$, of shape $h \times 1$.
For entity $i$, the Q-value is:  $$
  Q_a(\tau^a, \text{act on entity } i) = \tilde{e}^a_{i,t} W^u_e
  $$This produces a scalar. Note that all entities use the same $W^u_e$, since the network doesn't need a different set of weights for each $k$ because the embedding already contains the distinguishing information. 

So the final action-value is composed of:
* **Non-entity-related Q-values**: From the hidden state ($h^a_t W^u$), vector length = # of non-entity actions.
* **Entity-related Q-values**: For each entity, $\tilde{e}^a_{i,t} W^u_e$, scalar per entity-targeted action.
$$
  Q_a(\tau^a, \cdot) = [Q_{\text{non-entity}}, Q_{\text{entity-1}}, Q_{\text{entity-2}}, ...]
$$
## Transformer Mixer

TransfQMix uses a MLP in order to project $Q_a$  over $Q_{tot}$ (See Figure 2). 

Note that only three values are relevant: $n$, the number of agents; $h$, a hidden dimension; and 1, which accounts for $Q_{tot}$ being a scalar. 
$$ Q_{tot} = (Q_A^{(1 \times n)} W_1^{(n \times h)} + b_1^{(1 \times h)}) W_2^{(h \times 1)} + b_2^{(1 \times 1)} $$
Therefore, in order to arrange the MLP mixer we need $n+2$ vectors of size $h$ plus a scalar.  

In particular, for the first hidden layer, we need $n$ weight vectors, one per agent, + 1 vector from $b_1$. For $W_2$, we need another vector of weights. And finally, a scalar (from $b_2$). Thus, QMix generates the vectors using 4 MLP hypernetworks.  

The **outputs of the transformer** are used to generate the weights of the mixer’s MLP. 
### Input graph (transformer)

The input graph of the transformer mixer is:
$$ X_t = \left[h_t^1, \ldots, h_t^n, w_{t-1}^{b_1}, w_{t-1}^{W_1}, w_{t-1}^{W_2}, w_{t-1}^{b_2}, e_{1,t}, \ldots, e_{k,t}\right]^T $$
 - **$h_t^1, \ldots, h_t^n$**: Each agent’s hidden state at time $t$ (these are the outputs from the last agent transformer block). We can think of each as a summary of agent $i$’s perspective and reasoning so far.
-  **$w_{t-1}^{b_1}, w_{t-1}^{W_1}, w_{t-1}^{W_2}, w_{t-1}^{b_2}$**: Recurrent vectors (used as memory for the mixer itself). At the   start, they’re all zeros. As episodes proceed, they get updated, letting the mixer have a form of temporal memory or continuity.
 - **$e_{1,t}, \ldots, e_{k,t}$**: Embeddings for $k$ global entities (from the state $S_t$ mapped through the state embedding, $\text{Emb}^S$).

All these vectors have length $h$ (the embedding dimension). 
### Output (transformer)

The output consist in a matrix $\tilde{X}_t = \text{MultiHeadSelfAttn}(X_t)$ that contains the same vertices of $X_t$ transformed by the multi-head self-attention mechanism. 

#### Coordination reasonings

In particular, $\tilde{h}_t^1, \ldots, \tilde{h}_t^n$ are the coordination reasonings of agents enhanced by global information to which the agents had no access, **namely the hidden state of the other agents and the true state of the environment**.

These $n$ refined vectors (the hidden states) are used to build $W_1$. $Q_A W_1$ is therefore a **re-projection of the individual q-values $Q_A$** over a transformation of the agents’ hidden states. 

> [!NOTE] Key point
> Notice that the individual q-values were generated (or conditioned) exactly from $h_t^1, \ldots, h_t^n$ by the agents. This means that the primary goal of the transformer mixer is to combine and refine the independent agents’ reasoning so that they represent the team coordination.

The team Q-value $Q_{tot}$ is thus not a naive sum or mix of individual agent Qs, but a “contextualized” combination! 

#### Recurrent vectors

The transformed embeddings of the recurrent vectors:
- $w_t^{b_1} = \tilde{w}_{t-1}^{b_1}$, 
- $w_t^{W_2} = \tilde{w}_{t-1}^{W_2}$,
- $w_t^{b_2} = \tilde{w}_{t-1}^{b_2}$ 
are used to generate $b_1$, $W_2$, $b_2$, respectively. Since $b_2$ is a scalar, an additional parameterized matrix with dimensions $h \times 1$ is applied on $w_t^{b_2}$. 

There are two main reasons to introduce the recurrent mechanism:
- **Indepencence from number of entities**: These vectors are always present, always the same size, always in the same positions: They provide an anchor.we can ensure that the **transformer mixer** *(not the mixer network!)* is totally independent of the number of entities in the environment. 
- **Temporal dependence**: The mixer can take into account how the environment and agents evolved over previous timesteps. This aligns with the Markov Decision Process (MDP) formulation, where history can matter

$Q_{tot}$ is heavily dependent on prior states and this reliance should be encoded explicitly on the mixer network. This recurrent process allows the mixer to provide more consistent targets across time steps, resulting in more stable training.
# Notes

When we say that TransQMix is totally transferable, it's because we can transfer the individual agent policies - since all agents share the same network, which isn't dependent on $k$ (the number of entities). However, the mixer isn't transferable across different $n$ (number of agents), because the input/output weights and biases depend on $n$, the number of agents. 
# Advantages

1) We can employ the same weights of an embedded feed forward network to process the same vertex features, reducing the complexity of the feature space.
2) We can learn the edges of the latent coordination graph using a self-attention mechanism.
3) TransQMix is totally transferable, meaning that the same parameters can be applied to control and train larger or smaller teams. 
4) Transformers are naturally fitted for MARL settings, because these problems can be represented with graphs. 

