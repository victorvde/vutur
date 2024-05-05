"""
TODO

* automatically generate instructions from spirv.xml
* make some functions to manipulate the kernel as e.g. a (R)V(S)DG
* write it out as bytes
"""

from typing import Union
from dataclasses import dataclass


base_argtype = Union[None, "SpirvInstruction", str, int, float]
argtype = Union[tuple["argtype", ...], base_argtype]


@dataclass
class SpirvInstruction:
    opcode: int
    args: list[argtype]
    hasresult: bool
