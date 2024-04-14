"""
.. include:: ../README

Architecture:
* `vutur.allocator`: custom allocation algorithm that can split allocations, to deal with 4GB limits.
* `vutur.vulkan`: the guts of setting up a Vulkan device, uploading/downloading data, and dispatching kernels.
* `vutur.spirv`: SPIR-V instructions and utilities for manipulating kernels.
* `vutur.jit`: computation graph optimizer, fusing kernels etc
* `vutur.autograd`: [autograd](https://github.com/HIPS/autograd)-like automatic differentiation library.
* `vutur.np`: [numpy](https://numpy.org/)-like arrays and linear algebra with autograd support.
* `vutur.flax`: [Flax](https://flax.readthedocs.io/)-like neural network library.
* `vutur.optax`: [Objax](https://optax.readthedocs.io/)-like loss and optimization functions.
* `vutur.dataloader`: simple data loader.
"""
