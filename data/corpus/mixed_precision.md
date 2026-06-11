# Mixed-Precision Training

Mixed-precision training runs most of a network's forward and backward passes in
16-bit floating point while keeping a 32-bit master copy of the weights, roughly
halving memory traffic and exploiting tensor cores for large speedups on modern GPUs.

The fp16 format's narrow exponent range can underflow small gradients, so loss scaling
multiplies the loss before backpropagation and unscales gradients before the optimiser
step, skipping steps whose gradients contain infinities. The bfloat16 format avoids
most of this machinery: it keeps fp32's exponent range with reduced mantissa
precision, so training is stable without loss scaling, which is why it is the default
on TPUs and recent NVIDIA hardware.

Autocast layers choose precision per operation: matrix multiplies run in low
precision, while reductions like softmax and layer norm stay in fp32 for numerical
safety.
