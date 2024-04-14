"""
TODO

Do something similar to jax's tracer stuff to create a graph from a function.

Then do some kernel fusion to reduce memory traffic (e.g. https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467 ).

Then topologically sort the operations into a list.
Try to find the topological sort that has the least total memory usage (NP-hard in general but best effort is fine).
"""

pass
