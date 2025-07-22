
## Motivation

- Symmetricity is a **strong inductive bias** that can support the training process of DRL by making compact training space.
- Learning symmetricity is beneficial to **increasing generalization capability for unseen CO problems** because symmetricity induces the invariant representation that every COP contains.

## Introduction 

Sym-NCO considers two symmetricity types. First, since routing problems, e.g., TSP and VRP, are often represented in the 2D Euclidean space, their solutions must be invariant to geometric transformations of the input city coordinates, such as rotation, permutation, translation, and scaling (see Figure 1). Formally:

> [!NOTE] Problem symmetricity
> Problem $P^i$ and $P^j$ are problem symmetric ($P^i \stackrel{\text{sym}}{\longleftrightarrow} P^j$) if their optimal solution sets are identical.

Second is a type of symmetricity that was already considered by POMO in the case of TSP, and it refers to redundance of solutions: starting from node 2 should yield the same graph as starting from node 1 (see Figure 1).

> [!NOTE] Solution symmetricity: 
> Two solutions $\pi^i$ and $\pi^j$ are solution symmetric ($\pi^i \stackrel{\text{sym}}{\longleftrightarrow} \pi^j$) on problem $P$ if $R(\pi^i; P) = R(\pi^j; P)$.

![[symnco_symmetricity.png|400]]

The REINFORCE algorithm is used with a modified baseline to include symmetricity-considered terms:
- Multiple solutions are sampled from the transformed problems
- Their average return is used

This paper considers rotational problem symmetricity only. 
# Method

The total loss function is:
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Sym-RL}} + \alpha \mathcal{L}_{\text{inv}} = \mathcal{L}_{\text{ps}} + \beta \mathcal{L}_{\text{ss}} + \alpha \mathcal{L}_{\text{inv}} $$
![[symnco_method.png|500]]

### Solution symmetricity loss

Here, $\mathcal{L}_{\text{ss}}$ is a REINFORCE loss with the baseline specially designed to exploit the solutions symmetry of CO:

$$
\mathcal{L}_{\text{ss}} = -\mathbb{E}_{\pi \sim F_\theta(\cdot|P)} \left[ R(\pi; P) \right]
$$
$$
\nabla_\theta \mathcal{L}_{\text{ss}} = -\mathbb{E}_{\pi \sim F_\theta(\cdot|P)} \left[ \left( R(\pi; P) - b(P) \right) \nabla_\theta \log F_\theta \right] 
$$
$$
\approx -\frac{1}{K} \sum_{k=1}^K \left[ \left( R(\pi^k; P) - \frac{1}{K} \sum_{k=1}^K R(\pi^k; P) \right) \nabla_\theta \log F_\theta \right]
$$
where $\{\pi^k\}_{k=1}^K$ are the solutions of $P$ (the problem instance) sampled from $F_\theta(\cdot|P)$, $\log F_\theta$ is the log-likelihood of $F_\theta$, $K$ is the number of sampled solutions, $b(P)$ is a shared baseline which is the average reward from $K$ solutions for the identical problem $P$.

Note that the sum of advantage in the solution group $\{\pi^k\}_{k=1}^K$ must be 0:
$$
\frac{1}{K} \sum_{k=1}^K \left[ \left( R(\pi^k; P) - \frac{1}{K} \sum_{k=1}^K R(\pi^k; P) \right) \right] = 0
$$
This is the same as saying that the baseline is chosen such that the mean advantage over all sampled solutions for the same problem instance is zero. This is a zero-sum game:

- **The outer average** ensures the final gradient estimate is the mean contribution per sample, so the gradient estimate is unbiased with respect to sampling -> generate higher-reward solutions over time
- **The inner average** ensures the baseline is the **average performance** over all samples for the same problem instance, not just a fixed or global baseline -> give a small reward deviation between equal solutions

### Problem symmetricity loss

$$
\mathcal{L}_{\text{ps}} = -\mathbb{E}_{Q^l \sim Q} \mathbb{E}_{\pi \sim F_\theta(\cdot|Q^l(P))} \left[ R(\pi; P) \right]
$$
$$
\nabla_\theta \mathcal{L}_{\text{ps}} = -\mathbb{E}_{Q^l \sim Q} \left[ \mathbb{E}_{\pi \sim F_\theta(\cdot|Q^l(P))} \left[ \left( R(\pi; P) - b(P, Q) \right) \nabla_\theta \log F_\theta \right] \right]
$$
$$
\approx -\frac{1}{LK} \sum_{l=1}^L \sum_{k=1}^K \left[ \left( R(\pi^{l,k}; P) - \frac{1}{LK} \sum_{l=1}^L \sum_{k=1}^K R(\pi^{l,k}; P) \right) \nabla_\theta \log F_\theta \right]
$$

where $Q$ is the distribution of random orthogonal matrices, $Q^l$ is the $l^{\text{th}}$ sampled rotation matrix, and $\pi^{l,k}$ is the $k^{\text{th}}$ sample solution of the $l^{\text{th}}$ rotated problem. They $L$ problem symmetric problems, $Q^1(P), ..., Q^L(P)$, by using the sampled rotation matrices, and sample $K$ symmetric solutions from each of the $L$ problems. Then, the shared baseline $b(P, Q)$ is constructed by averaging $L \times K$ solutions.

This loss is actually an extension of $\mathcal{L}_{ss}$ that can both leverage problem symmetricity and solution symmetricity simultaneously. However, some specific CO problems such as TSP have cyclic nature with pre-identifiable solution symmetricity that it's advantageous to exploit. $\beta$ can be set to 0 if the solution symmetricity cannot be pre-identified.

### Regularization loss $\mathcal{L}_{\text{inv}}$ 

The encoder of $F_\theta$ can be enforced to have invariant representation by leveraging a pre-identified symmetricity.

We denote $h(x)$ and $h(Q(x))$ as the hidden representations of $x$ and $Q(x)$, respectively. **NOTE**: In this paper, Q isn't the Q-value, but a rotational matrix. The original problem $x$ and its rotated problem $Q(x)$ have identical solutions.

To impose the rotational invariant property on $h(x)$, solver $F_\theta$ is trained with an additional regularization loss term $\mathcal{L}_{\text{inv}}$ defined as:
$$
\mathcal{L}_{\text{inv}} = -S_{\cos}(g(h(x)), g(h(Q(x))))
$$
where $S_{\cos}(a, b)$ is the cosine similarity between $a$ and $b$. $g$ is the MLP-parameterized **projection head**.

Penalizing the difference between $g(h(x))$ and $g(h(Q(x)))$ instead of  $h(x)$ and $h(Q(x))$ allows the use of an arbitrary encoder network architecture while maintaining the diversity of $h$.

### Projection head

The projection head is a simple two-layer perception with the ReLU activation function, where input/output/hidden dimensions are equals to encoder’s embedding dimension. 

Its use is inspired by contrastive learning, where it's commonly used so that the **backbone feature extractor** isn't compromised by the loss applied to the **projection space**. 

# Results

Some interesting findings are:
- Using a projection head does indeed improve performance, which proves the importance of maintaining the expression power of the encoder (**IMPORTANT**!)
- $\mathcal{L}_\text{inv}$ increases the cosine similarity of the projected representation (i.e. $g(h)$)

![[symnco_results.png]]

https://arxiv.org/pdf/2310.15543 