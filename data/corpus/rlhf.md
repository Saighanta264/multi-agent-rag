# RLHF and Preference Tuning

Reinforcement learning from human feedback (RLHF) aligns a pretrained language model
with human preferences. Annotators rank pairs of model responses; a reward model is
trained to predict those rankings; and the policy is then optimised against the reward
model with an algorithm such as PPO, with a KL penalty keeping it close to the
original model so it does not drift into reward hacking.

Direct preference optimization (DPO) simplifies the recipe by skipping the explicit
reward model and RL loop entirely: it derives a closed-form classification loss on the
preference pairs themselves, which is more stable and far cheaper to run.

Preference tuning is what turns a raw next-token predictor into a helpful assistant —
it shapes refusals, tone, formatting, and instruction following. Its main risks are
sycophancy, where the model tells users what they want to hear, and over-refusal.
