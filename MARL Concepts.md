# MG


# POMG

A Partially Observed Markov Game is the extension of a POMDP for Multi-agents. 

> [!NOTE] Partially Observable Markov Games
> The POMG is mathematically denoted by the tuple $(\mathcal{N}, \mathcal{X}, \{\mathcal{U}^i\}, \{\mathcal{O}^i\}, \mathcal{P}, \{R^i\}, \gamma)$, where $\mathcal{N} = \{1, \ldots, N\}$ denotes the set of $N > 1$ interacting agents, $\mathcal{X}$ is the set of global but unobserved system states, and $\mathcal{U}$ is the set of individual action spaces $\mathcal{U}^i$. The observation space $\mathcal{O}$ denotes the collection of individual observation spaces $\mathcal{O}^i$. The transition probability function is denoted by $\mathcal{P}$, the reward function associated with agent $i$ by $R^i$, and the discount factor is $\gamma$.

When agents face a **cooperative task** with a **shared reward function**, the POMG is then known as *decentralized Partially Observable Markov decision process (dec-POMDP)* 

NOTE: In partially observable domains, the inference of good policies is extended in complexity since the history of interactions becomes meaningful. *Why?* The current observation often isn't enough to figure out the state of the environment, so there might be important information from previous steps (observations, actions, rewards) that helps the agent infer what the hidden state is now.

Hence, the agents usually incorporate history-dependent policies $\pi^i_t: (\mathcal{O}^i)_{t \ge 0} \to P(\mathcal{U}^i)$, which map from a history of observations to a distribution over actions.
# MARL types

![[marl_schemes.png|400]]

## Centralized training and execution (CTE)

This family of methods assumes centralization during training and execution. This means that the actions of each agent can depend on the information from all agents. 

For a multi-agent POMDP, at each step of the problem we assume that:
- A centralized controller chooses actions for each agents, $\mathbf{a}$, based on the current histories for all agents, $\mathbf{h}$ (which are maintained and updated)
- Each agent takes the chosen actions $a = \langle a_1, \ldots, a_n \rangle$. 
- The centralized controller observes the resulting observations $o = \langle o_1, \ldots, o_n \rangle$.
- The (centralized) algorithm observes the current agent histories, $h$, (or this could be stored from the previous steps), the actions taken by the agents $a$, the resulting observations, and the joint reward $r$.

Thus, a simple form of CTE can be achieved with a single-agent RL method with centralized action and observation spaces. 

- **Pros**: Generally outperform decentralized execution methods (since they allow centralized control) 
- **Cons**: Less scalable, as the (centralized) action and observation spaces scale exponentially with the number of agents

CTE is typically only used in the cooperative MARL case, since centralized control implies coordination on what actions will be selected by each agent.
## Decentralized training and execution (DTE)

Decentralized training and execution methods make the fewest assumptions and are often simple to implement. 

At each step of the problem we assume that:
- Each agent observes (or maintains) its current history, $h_i$, takes an action at that history, $a_i$, and sees the resulting observation, $o_i$.
- The (decentralized) algorithm observes the same information ($h_i$, $a_i$, and $o_i$) as well as the joint reward $r$.

It is worth noting that DTE is required if no centralized training phase is available (e.g., though a centralized simulator), requiring all agents to learn during online interactions without prior coordination. DTE methods can be applied in cooperative, competitive, or mixed cases. 
## Centralized-training decentralized-execution (CTDE)

CTDE methods can use centralized information during training but execute in a decentralized manner, using only information available to that agent during execution. 

> [!NOTE] Where does the main idea of CTDE come from?
> It comes from team decision making, since it's natural to think of a deriving solution for the team as a whole, and then assign corresponding parts to the team members. 

Thus, it requires a separate training *offline* phase where any available information (e.g., other agents' policies, underlying states) can be used. 

- **Pros**: More scalable than CTE methods, do not require communication during execution, and can often perform well. 

CTDE fits most naturally with the cooperative case, but can be potentially applied in competitive or mixed settings depending on what information is assumed to be observed.

Formally:

> [!NOTE] Definition of CTDE
> Each agent $i$ holds an individual policy $\pi^i: \mathcal{O}^i \to P(\mathcal{U}^i)$ which maps local observations to a distribution over individual actions. During training, agents are endowed with additional information, which is then discarded at test time.

Another way to see it is that **decentralized policies can often be learned in a centralized fashion in a simulated or laboratory setting**. This often grants access to additional state information, otherwise hidden from agents, and removes inter-agent communication constraints.

**So what is the main challenge**?

In MARL, the environment is a joint system of all agent states and actions. The system's full action at any step is a joint action $\mathbf{a} = (a_1, \dots, a_N)$. So the action-value function generalizes to: $Q_{\text{tot}}(s, a_1, a_2, ..., a_N)$, where $N$ is the number of agents. 

However, learning $Q_{\text{tot}}$ can is much harder than in a single-agent setting:
- The joint action space grows exponentially with $N$
- **Credit assignment problem**: When a team gets a reward, how do you assign credit or blame to individual agents for their contributions to the outcome? A naive $Q_{\text{tot}}$ gives the overall value, but doesn’t help with this decomposition. *This problem is about learning*
- **Decentralized Policy extraction problem**: Even if $Q_{\text{tot}}$ can be learned, it offers no obvious way to extract decentralized policies that allow each agent to select only an individual action based on an individual information. *This problem is about acting (constructing policies)*

### The IGM condition

> [!NOTE] The Individual Global-Max principle
> A condition for implementing CTDE effectively in multi-agent Q-Learning is that a greedy sampling of the joint action is equivalent to sampling the actions greedily from the individual agent. 

*In simpler terms*: If **each agent picks the action that is best for itself** (according to its own value function, using only its own information), then **the team as a whole** is also picking the best possible joint action (according to the global/team value function).

This lets us train with a central value function, but execute in a decentralized way (thus allowing CTDE)

In more detail, MARL value-based methods can be represented as:

$\arg \max_{\mathbf{u}} Q_{tot}(s, \mathbf{u}) = \begin{pmatrix} \arg \max_{u^1} q_1(\tau^1, u^1) \\ \vdots \\ \arg \max_{u^n} q_n(\tau^n, u^n) \end{pmatrix}$, (1)

where the individual agent utility is represented by $q_i, i \in N$. The **IGM principle** affirms the consistency of global and local action selection, as well as the factorization relationship between the joint-action-value function $Q_{tot}$ and the local-action-value function $q_i$.

Respecting the IGM also implies (as a side effect) that taking the $\arg \max$ of $Q_{\text{tot}}$, **required by off-policy updates**, is trivially tractable ([QMIX](https://arxiv.org/pdf/1803.11485|)). *Why*? Because maximizing $Q_\text{tot}$ can be done by simply taking each agent's individual maximizing action, bypassing the need for an exponential search over all joint actions. 
### Decentralized action-value function

Let each agent $a$ learn an individual action-value function $Q_a$ independently. An example of this instance is IQL. However, this approach cannot explicitly represent interactions between the agents and may not converge, as each agent’s learning is confounded by the learning and exploration of others.
### Centralized state-action value function

Learn a fully centralized state-action value function $Q_\text{tot}$ and then use it to guide the optimization of decentralized policies in an actor-critic framework, an approach taken by counterfactual multi-agent (COMA) policy gradients. 

However, this requires on-policy learning, which can be sample-inefficient, and training the **fully centralized critic becomes impractical** when there are more than a handful of agents.
### Value function factorization

The main idea is to factor the centralized action-value function $Q_\text{tot}$ into functions of the individual agents, often in a way that ensures decentralized policies can be optimal (or at least not too suboptimal). 
#### VDN

By representing $Q_\text{tot}$ as a sum of individual value functions $Q_{a}$ that condition only on individual observations and actions, a decentralised policy arises simply from each agent selecting actions greedily with respect to its $Q_{a}$. 

The main assumption is that the joint action-value function for the system can be additively decomposed into value functions across agents:

**Pros**:
- Avoids the spurious rewards problem (agents receive rewards signals originating from their teammates due to PO)
- A decentralised policy arises simply from each agent performing greedy action selection with respect to its $Q_{a}$.
- Respects the IGM
**Cons**: 
- It severely limits the complexity of centralised action-value functions that can be represented
- Ignores any extra state information available during training

![[marl-vdn.png|400]]
#### QMIX

To respect the IGM, the full factorization employed by VDN is not necessary. Instead,  it suffices to enforce a monotonicity constraint on the relationship between $Q_{tot}$ and each $Q_a$:
$$ \frac{\partial Q_{tot}}{\partial Q_a} \ge 0, \forall a $$
QMIX consists of agent networks representing each $Q_{a}$, and a mixing network that combines them into $Q_{\text{tot}}$, not as a simple sum as in VDN, but in a complex non-linear way that ensures consistency between the centralised and decentralised policies. 

At the same time, it enforces the constraint in the last equation by restricting the mixing network to have positive weights. 

In comparison with VDN, the sum-function is substituted with a MLP that can learn sophisticated non-linear projections of several action-values over $Q_\text{tot}$. 
##### Hypernetworks

Note that the weights of the mixing network are produced by separate hypernetworks, which take $s$. However, we want $Q_{\text{tot}}$  to **flexibly depend** on the global state $s$, potentially in complex, non-monotonic ways.

If we just fed $s$ as another input to the mixing network, it would be forced to combine $s$ with the agent $Q_{a}$  *monotonically*, which is too restrictive.

With this approach, the relationship between $s$ and $Q_{\text{tot}}$ can be as complex as we want, since the hypernetworks can produce very different weights for the mixing network depending on the global state. However, the relationship between the agent $Q_{a}$ and $Q_{\text{tot}}$ is **still monotonic**, because the mixing network’s structure (positive weights) enforces this.

**Architecture**: Each hypernetwork consists of a single linear layer, followed by an absolute activation function, to ensure that the mixing network weights are non-negative. The biases are produced in the same manner but are not restricted to being non-negative. The final bias is produced by a 2 layer hypernetwork with a ReLU non-linearity

![[marl_qmix.png|400]]

##### Training

In a training step, we:

1. **Sample a mini-batch**  
   Draw $b$ transitions from the replay buffer:   $$(s_t, \mathbf{u}_t, r_t, s_{t+1}, \mathbf{o}_{t+1},\text{done})$$where $\mathbf{u}_t$ is the joint action of all agents at time $t$, and $\mathbf{o}_t = (o_t^1, o_t^2, ..., o_t^n)$ is the vector of all agents’ observations at time $t$

2. **Forward pass through agent networks and mixing network**  
   - For each agent $a$, compute its action-value:  $$Q_a(\tau_a, u_a; \theta)$$where $\tau_a$ is the agent’s action-observation history.
   - Combine these using the mixing network (with weights generated by hypernetworks from the state $s$) to get the joint value:  $$Q_\text{tot}(\boldsymbol{\tau}, \mathbf{u}, s; \theta)$$where $\boldsymbol{\tau}$ denotes the tuple of all agent histories.

3. **Compute the target value using the target network**  
   - The target for each transition is:  $$y_\text{tot} = r + \gamma \max_{\mathbf{u}'} Q_\text{tot}(\boldsymbol{\tau}', \mathbf{u}', s'; \theta^-)$$ where $\theta^-$ is the target network’s parameters and $\gamma$ is the discount factor.

4. **Calculate the loss**  
   - The TD error loss for the mini-batch is:  $$L(\theta) = \sum_{i=1}^b \left(y_{\text{tot},i} - Q_\text{tot}(\boldsymbol{\tau}_i, \mathbf{u}_i, s_i; \theta)\right)^2$$
5. **Gradient descent update**  
   - Update $\theta$ (parameters of all agent networks, mixing network, and hypernetworks) by minimizing the loss $L(\theta)$.

6. **Periodically update target network**  
   - Set $\theta^- \leftarrow \theta$ every $C$ steps (for some target update interval $C$).

**Pros** 
- Can represent a richer class of action-value functions
- The factored representation scales well in the number of agents and allows decentralised policies to be easily extracted via linear-time individual argmax operations.
**Cons**:
- Individual agents are kept simple by employing recurrent neural networks (RNN) fed by observation vectors that are large concatenations of various types of features. By performing these concatenations a key information is lost: **the fact that many of the features are exactly of the same type despite referring to separate entities (e.g., the position in a map).** See [[TransfQMix - Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems|TransQMix]] for a method that addresses the same problem. 

