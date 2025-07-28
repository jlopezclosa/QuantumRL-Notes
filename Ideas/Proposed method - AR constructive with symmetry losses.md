For this proposal, we take inspiration from both [[Sym-NCO - Leveraging Symmetricity for Neural Combinatorial Optimization|Sym-NCO]] and [[Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures|Attention-Based Deep RL]]. The idea is to exploit all the symmetries in the quantum allocation domain (see [[Symmetry for Qubit Allocation]]) using a similar formula as Sym-NCO. 

We represent the information just as in [[Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures|Attention-Based Deep RL]], however, we make the following modifications:
- Use a relative positional encoding mechanism presented in the paper [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155), 
- Employ both solution and problem symmetry losses

We employ exactly this loss: $$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Sym-RL}} + \alpha \mathcal{L}_{\text{inv}} = \mathcal{L}_{\text{ps}} + \beta \mathcal{L}_{\text{ss}} + \alpha \mathcal{L}_{\text{inv}} $$We describe how we compute each term next. 

