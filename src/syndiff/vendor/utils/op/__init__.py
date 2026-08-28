import os
# Inference-only path on Windows without MSVC/CUDA build chain: bypass the
# C++ JIT-loaded StyleGAN2 kernels and use pure-PyTorch replacements.
if os.environ.get("SYNDIFF_NATIVE_OPS", "1") == "1":
    from ._native_ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
else:
    from .fused_act import FusedLeakyReLU, fused_leaky_relu
    from .upfirdn2d import upfirdn2d
