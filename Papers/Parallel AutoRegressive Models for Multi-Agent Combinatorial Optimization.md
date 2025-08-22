# Motivation

This paper extends constructive autoregressive cominatorial optimization methods for [[MARL Concepts|multi-agent systems]]. In aims to solve common problems when extending this framework to problems with multiple agents:
- Suboptimal agent coordination, resulting in unsatisfactory solution quality and poor generalization across varying problem sizes and agent configurations. 
- High computational latency, since AR sequence generation is associated with high latency due to each single action or token depending on each single previous one.

![[parco_example.png|300]]
# Advantages

## $N$ and $M$ can vary

PARCO:
- Generalizes to different number of agents $M$ number of nodes $N$ (**at inference time**)
- Can be trained with different $M$ and $N$ sizes. 

Here we present proof that PARCO is trained with different $N$ and $M$ sizes:

> [!NOTE] Traning data in HCVRP 
> Neural baselines were trained with the specific number of nodes $N$ and number of agents $M$ they were tested on. In PARCO, we select a **varying size and number of customer training schemes**: at each training step, we sample $N \sim \mathcal{U}(60, 100)$ and $m \sim \mathcal{U}(3, 7)$. As we show in **Table 1**, a single PARCO model can outperform baseline models even when they were fitted on a specific distribution. The coordinates of each customer location $(x_i, y_i)$, where $i=1, \dots, N$, are sampled from a uniform distribution $\mathcal{U}((0.0, 1.0))$ within a two-dimensional space. The depot location is similarly sampled using the same uniform distribution. The demand $d_i$ for each customer $i$ is also drawn from a uniform distribution $\mathcal{U}(1, 10)$, with the depot having no demand, i.e., $d_0 = 0$. Each vehicle $m$, where $m=1, \dots, M$, is assigned a capacity $Q_m$ sampled from a uniform distribution $\mathcal{U}(20, 41)$. The speed $f_m$ of each vehicle is uniformly distributed within the range $\mathcal{U}((0.5, 1.0))$.

In the generator class of the OMDCPDP:
```python
def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # NOTE: deprecated
        num_agents = None  # done inside the sampling
        # Sample number of agents during training step
        if phase == "train":
            # Idea: we always have batches of the same size from the dataloader.
            # however, here we sample a subset of agents and locations from the batch.
            # For instance: if we have always 10 depots and 100 cities, we sample a random number of depots and cities
            # from the batch. This way, we can train on different number of agents and locations.
            num_agents = randint(self.train_min_agents, self.train_max_agents)
            num_locs = randint(self.train_min_size, self.train_max_size)
            batch = resample_batch(batch, num_agents, num_locs)
        else:
            if self.allow_multi_dataloaders:
                # Get number of agents to test on based on dataloader name
                if dataloader_idx is not None and self.dataloader_names is not None:
                    # e.g. n50_m7 take only number after "m" until _
                    num_agents = int(
                        self.dataloader_names[dataloader_idx].split("_")[-1][1:]
                    )
                else:
                    num_agents = self.val_test_num_agents

                # NOTE: trick: we subsample number of agents by setting num_agents
                # in such case, use the same number of agents for all batches
                batch["num_agents"] = torch.full(
                    (batch.shape[0],), num_agents, device=batch.device
                )

```

Dataloaders typically provide batches with a fixed shape - this is convenient for batching in PyTorch, but doesn't provide variability. Therefore, they randomly select the number of agents and locations for this training iteration, uniformly within the specified ranges. This simulates training on a diverse set of problem sizes.

Then they **take the original batch and selects only the required number of agents and locations**. For example, if you sampled 6 agents and 80 locations, it keeps only those from the batch.

However:
- $N$ and $M$ must be fixed within a single problem instance. 
- All instances in the batch should have the same $N$ and $M$, for efficient batching. 


![[parco_overview.png]]![[parco_priorityhandler.png|300]]
