"""
TODO

* automatically generate instructions from spirv.xml
* make some functions to manipulate the kernel as e.g. a (R)V(S)DG
* write it out as bytes
"""

from __future__ import annotations
from typing import Union, assert_never, TYPE_CHECKING
from dataclasses import dataclass, field
from io import BytesIO
import struct

if TYPE_CHECKING:
    from vutur.spirv_instructions import Op

base_argtype = Union[None, "SpirvInstruction", str, int, float]
argtype = Union[tuple["argtype", ...], base_argtype]


@dataclass(frozen=True)
class SpirvInstruction:
    opcode: Op
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
        s.write_i(first)
        s.write(args_b)


def encode_arg(s: Serializer, arg: argtype) -> bytes:
    match arg:
        case tuple():
            r = b"".join((encode_arg(s, e) for e in arg))
        case None:
            r = b""
        case str():
            r = arg.encode()
            while len(r) % 4 != 0:
                r += b"\0"
        case int():
            r = arg.to_bytes(length=4, byteorder="little")
        case float():
            r = struct.pack("<f", arg)
        case SpirvInstruction():
            ri = s.gensym(arg)
            r = ri.to_bytes(length=4, byteorder="little")
        case _:
            assert_never(arg)
    return r


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

    def write_i(self, i: int) -> None:
        self.write(i.to_bytes(length=4, byteorder="little"))
