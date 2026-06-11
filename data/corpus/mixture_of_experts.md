# Mixture-of-Experts Models

Mixture-of-experts (MoE) layers replace a transformer's single feed-forward block with
many parallel expert networks and a learned router that sends each token to only one
or two of them. Total parameter count grows enormously while the compute per token
stays roughly constant, decoupling model capacity from inference cost.

The router is the delicate part. Without intervention it collapses onto a few popular
experts, so training adds load-balancing auxiliary losses and capacity limits that
drop or reroute overflow tokens. Expert parallelism distributes experts across
devices, making all-to-all communication the dominant systems cost.

Sparse models like Switch Transformer and Mixtral demonstrated that MoE matches or
beats dense models of equal training compute. The trade-offs are higher memory to hold
all experts, more complex serving, and fine-tuning instabilities that dense models do
not exhibit.
