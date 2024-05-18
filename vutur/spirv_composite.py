from vutur.spirv_base import SpirvInstruction, Serializer
from vutur.spirv_instructions import (
    OpFunctionEnd,
    Op,
    ANNOTATION_OPS,
    CONSTANT_OPS,
    TYPEDECL_OPS,
    SPIRV_MAGIC_NUMBER,
)
from io import BytesIO
from dataclasses import dataclass
import subprocess
from typing import Iterator


@dataclass(frozen=True)
class SpirvBlock:
    label: SpirvInstruction
    instructions: list[SpirvInstruction]

    def iter_instructions(self) -> Iterator[SpirvInstruction]:
        yield self.label
        for ins in self.instructions:
            yield ins


@dataclass(frozen=True)
class SpirvFunction:
    function: SpirvInstruction
    parameters: list[SpirvInstruction]
    blocks: list[SpirvBlock]

    def iter_instructions(self) -> Iterator[SpirvInstruction]:
        yield self.function
        for p in self.parameters:
            yield p
        for b in self.blocks:
            yield from b.iter_instructions()
        yield OpFunctionEnd()


# https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_logical_layout_of_a_module
module_globals = [
    [Op.Capability],
    [Op.Extension],
    [Op.ExtInstImport],
    [Op.MemoryModel],
    [Op.EntryPoint],
    [Op.ExecutionMode, Op.ExecutionModeId],
    [Op.String, Op.SourceExtension, Op.Source, Op.SourceContinued],
    [Op.Name, Op.MemberName],
    [Op.ModuleProcessed],
    ANNOTATION_OPS,
    TYPEDECL_OPS,  # todo: toposort these?
    CONSTANT_OPS,  # todo: toposort these?
    # todo: OpVariable globals
    [Op.Undef],
]
module_globals_lookup = {}
for i, ops in enumerate(module_globals):
    for op in ops:
        module_globals_lookup[op] = i


def is_global(ins: SpirvInstruction) -> bool:
    return ins.opcode in module_globals_lookup


@dataclass(frozen=True)
class SpirvModule:
    global_instructions: list[SpirvInstruction]
    func_decls: list[SpirvFunction]
    func_defs: list[SpirvFunction]

    def iter_instructions(self) -> Iterator[SpirvInstruction]:
        for f in self.func_decls:
            yield from f.iter_instructions()

        for f in self.func_defs:
            yield from f.iter_instructions()

    def get_globals(self) -> list[SpirvInstruction]:
        d: dict[SpirvInstruction, None] = {}

        def dfs(ins: SpirvInstruction) -> None:
            for arg in ins.args:
                if not isinstance(arg, SpirvInstruction):
                    continue
                dfs(arg)
            if is_global(ins):
                d.setdefault(ins, None)

        for ins in self.iter_instructions():
            dfs(ins)

        r = self.global_instructions + list(d.keys())
        return sorted(r, key=lambda x: module_globals_lookup[x.opcode])

    def serialize(self) -> bytes:
        s = Serializer(BytesIO())

        for ins in self.get_globals():
            ins.serialize(s)

        for ins in self.iter_instructions():
            ins.serialize(s)

        hs = Serializer(BytesIO())
        hs.write_i(SPIRV_MAGIC_NUMBER)
        hs.write_i((1 << 16) | (3 << 8))  # version 1.3
        hs.write_i(0)  # todo: register generator id with Khronos
        hs.write_i(s.max_id)
        hs.write_i(0)

        return hs.out.getvalue() + s.out.getvalue()


def validate(module: bytes) -> None:
    try:
        subprocess.run(["spirv-val"], input=module, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert False, e.stderr.decode()


def disasm(module: bytes) -> str:
    try:
        r = subprocess.run(
            ["spirv-dis", "--raw-id"], input=module, check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        assert False, e.stderr.decode()

    return r.stdout.decode()
