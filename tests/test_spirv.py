import vgrad.spirv_instructions as si
import vgrad.spirv_composite as sc


def test_empty_module() -> None:
    bb = sc.SpirvBlock(
        label=si.OpLabel(),
        instructions=[
            si.OpReturn(),
        ],
    )

    void = si.OpTypeVoid()

    ftype = si.OpTypeFunction(void)

    f = sc.SpirvFunction(
        function=si.OpFunction(void, si.FunctionControl(0), ftype),
        parameters=[],
        blocks=[bb],
    )

    module = sc.SpirvModule(
        func_decls=[],
        func_defs=[f],
    )

    b = module.serialize()

    print(sc.disasm(b))

    sc.validate(b)


def test_square_module() -> None:
    bb = sc.SpirvBlock(
        label=si.OpLabel(),
        instructions=[
            si.OpReturn(),
        ],
    )

    void = si.OpTypeVoid()

    ftype = si.OpTypeFunction(void)

    f = sc.SpirvFunction(
        function=si.OpFunction(void, si.FunctionControl(0), ftype),
        parameters=[],
        blocks=[bb],
    )

    module = sc.SpirvModule(
        func_decls=[],
        func_defs=[f],
    )

    b = module.serialize()

    print(sc.disasm(b))

    sc.validate(b)
