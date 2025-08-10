from vgrad.spirv_base import SpirvInstruction, Serializer
import vgrad.spirv_instructions as si
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
        yield si.OpFunctionEnd()


# https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_logical_layout_of_a_module
module_globals = [
    [si.Op.Capability],
    [si.Op.Extension],
    [si.Op.ExtInstImport],
    [si.Op.MemoryModel],
    [si.Op.EntryPoint],
    [si.Op.ExecutionMode, si.Op.ExecutionModeId],
    [si.Op.String, si.Op.SourceExtension, si.Op.Source, si.Op.SourceContinued],
    [si.Op.Name, si.Op.MemberName],
    [si.Op.ModuleProcessed],
    si.ANNOTATION_OPS,
    si.TYPEDECL_OPS,
    si.CONSTANT_OPS,
    # todo: OpVariable globals
    [si.Op.Undef],
]
module_globals_lookup = {}
for i, ops in enumerate(module_globals):
    for op in ops:
        module_globals_lookup[op] = i


def is_global(ins: SpirvInstruction) -> bool:
    return ins.opcode in module_globals_lookup


@dataclass(frozen=True)
class SpirvModule:
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

        defaults = (
            si.OpCapability(si.Capability.Shader),
            si.OpMemoryModel(si.AddressingModel.Logical, si.MemoryModel.GLSL450),
            si.OpEntryPoint(
                si.ExecutionModel.GLCompute, self.func_defs[0].function, "main"
            ),
        )

        r = defaults + tuple(d.keys())
        return sorted(r, key=lambda x: module_globals_lookup[x.opcode])

    def serialize(self) -> bytes:
        s = Serializer(BytesIO())

        for ins in self.get_globals():
            ins.serialize(s)

        for ins in self.iter_instructions():
            ins.serialize(s)

        hs = Serializer(BytesIO())
        hs.write_i(si.SPIRV_MAGIC_NUMBER)
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
