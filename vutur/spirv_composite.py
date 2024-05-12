from vutur.spirv_base import SpirvInstruction, Serializer
from vutur.spirv_instructions import OpFunctionEnd
from io import BytesIO


class SpirvBlock:
    label: SpirvInstruction
    instructions: list[SpirvInstruction]

    def serialize(self, s: Serializer) -> None:
        self.label.serialize(s)
        for ins in self.instructions:
            ins.serialize(s)


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


class SpirvModule:
    # https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_logical_layout_of_a_module
    capabilities: list[SpirvInstruction]
    extensions: list[SpirvInstruction]
    extimports: list[SpirvInstruction]
    memorymodel: SpirvInstruction
    entrypoints: list[SpirvInstruction]
    executionmodes: list[SpirvInstruction]
    strings: list[SpirvInstruction]
    names: list[SpirvInstruction]
    moduleprocesseds: list[SpirvInstruction]
    annotations: list[SpirvInstruction]
    types: list[SpirvInstruction]
    constants: list[SpirvInstruction]
    global_vars: list[SpirvInstruction]
    func_decls: list[SpirvFunction]
    func_defs: list[SpirvFunction]

    def serialize(self) -> bytes:
        s = Serializer(BytesIO())

        for x in self.capabilities:
            x.serialize(s)
        for x in self.extensions:
            x.serialize(s)
        for x in self.extimports:
            x.serialize(s)
        self.memorymodel.serialize(s)
        for x in self.entrypoints:
            x.serialize(s)
        for x in self.executionmodes:
            x.serialize(s)
        for x in self.strings:
            x.serialize(s)
        for x in self.names:
            x.serialize(s)
        for x in self.moduleprocesseds:
            x.serialize(s)
        for x in self.annotations:
            x.serialize(s)
        for x in self.types:
            x.serialize(s)
        for x in self.constants:
            x.serialize(s)
        for x in self.global_vars:
            x.serialize(s)
        for f in self.func_decls:
            f.serialize(s)
        for f in self.func_defs:
            f.serialize(s)

        return s.out.getvalue()
