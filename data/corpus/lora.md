# LoRA: Low-Rank Adaptation

Fine-tuning every parameter of a large model is expensive in memory and storage. LoRA
freezes the pretrained weights and injects small trainable low-rank matrices alongside
selected weight matrices, typically the attention projections. If a weight update is
approximated as the product of two thin matrices A and B with rank r much smaller than
the layer width, the trainable parameter count drops by orders of magnitude.

At inference time the low-rank update can be merged into the frozen weights, so LoRA
adds no latency. Multiple adapters can be trained for different tasks and swapped on
top of one shared base model, which makes multi-tenant serving practical.

QLoRA combines the idea with 4-bit quantization of the frozen base model, enabling
fine-tuning of very large models on a single GPU. The rank r and the scaling factor
alpha are the main hyperparameters; common ranks are 8 to 64.
