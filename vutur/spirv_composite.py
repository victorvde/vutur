from vutur.spirv_base import SpirvInstruction, Serializer
from vutur.spirv_instructions import (
    OpFunctionEnd,
    Op,
    ANNOTATION_OPS,
    CONSTANT_OPS,
    TYPEDECL_OPS,
)
from io import BytesIO
from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class SpirvBlock:
    label: SpirvInstruction
    instructions: list[SpirvInstruction]

    def serialize(self, s: Serializer) -> None:
        self.label.serialize(s)
        for ins in self.instructions:
            ins.serialize(s)


@dataclass(frozen=True)
class SpirvFunction:
    function: SpirvInstruction
    parameters: list[SpirvInstruction]
    blocks: list[SpirvBlock]

    def serialize(self, s: Serializer) -> None:
        self.function.serialize(s)
        for p in self.parameters:
            p.serialize(s)
        for b in self.blocks:
            b.serialize(s)
        OpFunctionEnd().serialize(s)


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
        module_globals_lookup[int(op)] = i


@dataclass(frozen=True)
class SpirvModule:
    global_instructions: list[SpirvInstruction]
    func_decls: list[SpirvFunction]
    func_defs: list[SpirvFunction]

    def serialize(self) -> bytes:
        # todo magic and headers

        # todo: get hidden global instuctions function bodies

        global_sections: list[list[SpirvInstruction]] = [] * len(module_globals)
        for ins in self.global_instructions:
            i = module_globals_lookup[ins.opcode]
            global_sections[i].append(ins)

        s = Serializer(BytesIO())

        for section in global_sections:
            for ins in section:
                ins.serialize(s)

        for f in self.func_decls:
            f.serialize(s)
        for f in self.func_defs:
            f.serialize(s)

        return s.out.getvalue()


def validate(module: bytes) -> None:
    try:
        subprocess.run(["spirv-val"], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert False, e.stderr.decode()
