"""
TODO

* automatically generate instructions from spirv.xml
* make some functions to manipulate the kernel as e.g. a (R)V(S)DG
* write it out as bytes
"""

from __future__ import annotations
from typing import Union, assert_never
from dataclasses import dataclass, field
from io import BytesIO
import struct

base_argtype = Union[None, "SpirvInstruction", str, int, float]
argtype = Union[tuple["argtype", ...], base_argtype]


@dataclass(frozen=True)
class SpirvInstruction:
    opcode: int
    args: tuple[argtype, ...]
    hasresult: bool
    hasrtype: bool

    def serialize(self, s: Serializer) -> None:
        args_b = b""

        args = self.args
        if self.hasrtype:
            args_b += encode_arg(s, args[0])
            args = args[1:]
        if self.hasresult:
            rid = s.gensym(self)
            args_b += rid.to_bytes(length=4, byteorder="little")
        for arg in args:
            args_b += encode_arg(s, arg)
        wordcount = len(args_b) // 4 + 1
        first = (wordcount << 16) + self.opcode
        first_b = first.to_bytes(length=4, byteorder="little")
        s.write(first_b)
        s.write(args_b)


def encode_arg(s: Serializer, arg: argtype) -> bytes:
    match arg:
        case tuple():
            return b"".join((encode_arg(s, e) for e in arg))
        case None:
            return b""
        case str():
            rb = arg.encode()
            while len(rb) % 4 != 0:
                rb += b"\0"
            return rb
        case int():
            return arg.to_bytes(length=4, byteorder="little")
        case float():
            return struct.pack("<f", arg)
        case SpirvInstruction():
            ri = s.gensym(arg)
            return ri.to_bytes(length=4, byteorder="little")
        case _:
            assert_never(arg)


@dataclass
class Serializer:
    out: BytesIO
    ids: dict[SpirvInstruction, int] = field(default_factory=dict)
    max_id: int = 0

    def gensym(self, ins: SpirvInstruction) -> int:
        i = self.ids.get(ins)
        if i is None:
            i = self.max_id
            self.max_id += 1
            self.ids[ins] = i
        return i

    def write(self, b: bytes) -> None:
        self.out.write(b)
