"""
.. include:: ../README

Architecture:
* `vgrad.allocator`: custom allocation algorithm that can split allocations, to deal with 4GB limits.
* `vgrad.vulkan`: the guts of setting up a Vulkan device, uploading/downloading data, and dispatching kernels.
* `vgrad.spirv`: SPIR-V instructions and utilities for manipulating kernels.
* `vgrad.jit`: computation graph optimizer, fusing kernels etc
* `vgrad.autograd`: [autograd](https://github.com/HIPS/autograd)-like automatic differentiation library.
* `vgrad.np`: [numpy](https://numpy.org/)-like arrays and linear algebra with autograd support.
* `vgrad.module`: PyTorch / Flax-like module system1
* `vgrad.optax`: [Objax](https://optax.readthedocs.io/)-like loss and optimization functions.
* `vgrad.dataloader`: simple data loader.
"""
