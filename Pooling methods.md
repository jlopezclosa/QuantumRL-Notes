# When to use them

## GNN

- Each node $v_i$ has a feature vector $\mathbf{h}_i \in \mathbb{R}^F$ after passing through a GNN layer (hidden state vector), where $F$ is the number of features per node
- Without pooling, the output of the GNN is: $H = [\mathbf{h}_1; \mathbf{h}_2; ...; \mathbf{h}_N]$, i.e., an $N \times F$ matrix
- But different graphs have different $|V|$
- We need pooling to condense information from the graph into a fixed-size representation. 

# Concatenation

Concatenation is a straightforward non-pooling method where a readout of the final hidden states of all nodes is performed and they are concatenated in any order. 

Pros:
- Simplicity
- Absence of information loss
- Applicability to fixed-size graphs (the output dimension depends on the product of the hidden state size and the number of nodes)
Cons:
- Cannot upscale to various qubit settings 
# Mean pooling

Straightforward method that calculates the average of the final hidden states of all nodes in the graph:
$$
\frac{\sum_{v_i \in V} \mathbf{h}_{v_i}^t}{|V|}
$$
- $t$ is the final step of the graph NN,
- $|V|$ is the number of nodes in the graph,
- $\mathbf{h}_{v_i}^t$ is the hidden state of node $v_i$ at timestep $t$.

Pros: 
- Computationally efficient
- Simple to implement
Cons:
- Can result in significant information loss, which makes it less effective for tasks requiring detailed preservations of graph structure (our case: cliques or disconnected neighbourhoods). This is because averaging erases differences between nodes. 

# Self-Attention Graph Pooling (SAG Pool)

Hierarchical pooling mechanism designed to reduce the graph size while retaining the most informative nodes. It employs a **trainable attention mechanism** to assign scores to nodes based on their features. The attention score is calculated as:
$$
Z=\sigma\left(\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} \mathbf{X} \Theta_{\mathrm{att}}\right)
$$
where:
- $\sigma$ is the activation function (e.g., tanh),
- $\mathbf{D} \in \mathbb{R}^{N \times N}$ is the degree matrix of the graph, which can also be replaced by an edge weight matrix if relevant,
- $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the adjacency matrix of the graph,
- $\mathbf{X} \in \mathbb{R}^{N \times F}$ is the node feature matrix, where $F$ is the feature dimension,
- $\Theta_{\text {att }} \in \mathbb{R}^{F \times 1}$ is a trainable parameter.

The attention scores $Z$ are used to select the top $k$ nodes, where $k$ is a predefined parameter. $k$ can either be an integer $k \geq 1$, specifying the exact number of nodes to select, or a ratio $k \in(0,1]$, where $\lceil k N\rceil$ nodes are retained. Therefore, form a smaller graph with the selected nodes which always has fixed size.
# Set Transformer

The Set Transformer is an attention-based aggregation mechanism specifically designed for set-structured data.

- Uses Encoder-Decoder architecture with Self-Attention.
- Introduces Inducing Points (ISAB) to reduce computational cost from $O(n^2)O(n^2)$ to $O(l⋅n⋅m)O(l⋅n⋅m)$.
- Decoder uses Pooling by Multi-Head Attention (PMA) with trainable seed vectors.

Pros:
- Produces permutation-invariant, fixed-size outputs.
- Good scalability and expressiveness for variable-sized graphs.
- Particularly useful for quantum circuit optimization.