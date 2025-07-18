# PPO
Selected due to its good training efficiency. Plus, it's well-suited for scenarios involving a large number of **discrete** actions. This aligns with the case where the number of **cores** grows. 
The model is constrained to a fixed number of actions, which **correspond to the available cores**. 
However the approach retains a degree of flexibility in adapting to varying number of cores. 
# GNNs
This work uses GATs (Graph Attention Networks)