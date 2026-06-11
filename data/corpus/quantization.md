# Model Quantization

Quantization reduces the numeric precision of a network's weights and sometimes its
activations — from 16-bit floats down to 8-bit or 4-bit integers — to shrink memory
footprint and speed up inference on commodity hardware.

Post-training quantization converts a finished model without further training; methods
like GPTQ and AWQ calibrate per-channel scales on a small sample of data to minimise
the rounding error that matters most. Quantization-aware training instead simulates
low precision during fine-tuning so the network learns to compensate.

The standard trade-off is accuracy versus compression: 8-bit quantization is usually
lossless in practice, while 4-bit schemes such as NF4 lose a little quality in
exchange for fitting a model into a quarter of the memory. Outlier features in large
models are the main failure mode, which mixed-precision schemes handle by keeping a
few sensitive channels in higher precision.
