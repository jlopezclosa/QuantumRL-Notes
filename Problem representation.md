# GNN

Problem formulation:
	Graphs are represented as $G=(V,E)$, where $V$ is the set of nodes (or vertices) and $E \subseteq V \times V$ is the set of edges.
	Each node $v \in V$ can be associated with a feature vector $\textbf{x}_{v}$, and each edge $e \in E$ may have an associated weight or feature. 
	The length (number of elements) in  $\textbf{x}_{v}$ is $F$. Therefore, if we stack all node features together, we get a node feature matrix $X\in \mathbb{R}^{|V|\times F}$
	In graph terminology, $|V| = N$

Primary objective:
	Learn node embeddings, edge embeddings, or graph-level embeddings by aggregating information from the graph's structure and the features of its nodes and edges. 


# Multi-Agent

