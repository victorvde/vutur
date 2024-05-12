# https://registry.khronos.org/SPIR-V/specs/unified1/MachineReadableGrammar.html

import sys
import json


def sanitize(identifier: str) -> str:
    if identifier[0].isdigit() or identifier == "None":
        return "_" + identifier
    return identifier


def load_kind_types(fn: str) -> dict[str, str]:
    kind_types = {}

    with open(fn, "rb") as f:
        data = json.load(f)
    operand_kinds = data["operand_kinds"]

    for operand_kind in operand_kinds:
        category = operand_kind["category"]
        kind = operand_kind["kind"]
        match category:
            case "BitEnum" | "ValueEnum":
                kind_types[kind] = kind
            case "Id":
                kind_types[kind] = "SpirvInstruction"
            case "Literal":
                match kind:
                    case (
                        "LiteralInteger"
                        | "LiteralExtInstInteger"
                        | "LiteralSpecConstantOpInteger"
                    ):
                        t = "int"
                    case "LiteralString":
                        t = "str"
                    case "LiteralFloat" | "LiteralContextDependentNumber":
                        t = "float"
                    case e:
                        assert False, e
                kind_types[kind] = t
            case "Composite":
                bases = operand_kind["bases"]
                ts = [kind_types[k] for k in bases]
                t = f"tuple[{", ".join(ts)}]"
                kind_types[kind] = t
            case e:
                assert False, e

    return kind_types


def main() -> None:
    """
    Convert spirv grammar json to Python file the way we like it.
    """
    kind_types = load_kind_types(sys.argv[1])

    fn = sys.argv[2]
    with open(fn, "rb") as f:
        data = json.load(f)

    _copyright = data.pop("copyright")
    major = data.pop("major_version", None)
    if major is None:
        version = data.pop("version")
    else:
        minor = data.pop("minor_version")
        version = f"{major}.{minor}"
    revision = data.pop("revision")
    magic = data.pop("magic_number", None)
    _instruction_printing_class = data.pop("instruction_printing_class", None)
    instructions = data.pop("instructions")
    operand_kinds = data.pop("operand_kinds", [])
    assert len(data) == 0, data.keys()

    print('"""')
    print(
        f"automatically generated by vutur/tools/spirv_generator.py from {fn} version {version}.{revision}"
    )
    print('"""')

    print("from enum import IntFlag, IntEnum")
    print("from vutur.spirv_base import SpirvInstruction")
    print("from typing import Optional")
    if magic is not None:
        print()
        print()
        print(f"SPIRV_MAGIC_NUMBER = {magic}")

    for operand_kind in operand_kinds:
        category = operand_kind.pop("category")
        kind = operand_kind.pop("kind")
        enumerants = operand_kind.pop("enumerants", None)
        match category:
            case "BitEnum":
                print()
                print()
                print(f"class {kind}(IntFlag):")
                for enumerant in enumerants:
                    enumerant_ = enumerant.pop("enumerant")
                    value = enumerant.pop("value")
                    _version = enumerant.pop("version", None)
                    _parameters = enumerant.pop("parameters", [])
                    _capabilities = enumerant.pop("capabilities", [])
                    _extensions = enumerant.pop("extensions", [])
                    assert len(enumerant) == 0, enumerant.keys()

                    if enumerant_ == "None":
                        assert value == "0x0000", value
                        continue

                    print(f"    {sanitize(enumerant_)} = {value}")
            case "ValueEnum":
                print()
                print()
                print(f"class {kind}(IntEnum):")
                for enumerant in enumerants:
                    enumerant_ = enumerant.pop("enumerant")
                    value = enumerant.pop("value")
                    print(f"    {sanitize(enumerant_)} = {value}")
            case "Id":
                _doc = operand_kind.pop("doc")
                # we don't differentiate between id's for now
            case "Literal":
                _doc = operand_kind.pop("doc")
                match kind:
                    case (
                        "LiteralInteger"
                        | "LiteralExtInstInteger"
                        | "LiteralSpecConstantOpInteger"
                    ):
                        t = "int"
                    case "LiteralString":
                        t = "str"
                    case "LiteralFloat" | "LiteralContextDependentNumber":
                        t = "float"
                    case e:
                        assert False, e
            case "Composite":
                _bases = operand_kind.pop("bases")
            case e:
                assert False, e

        assert len(operand_kind) == 0, operand_kind.keys()

    for instruction in instructions:
        opname = instruction.pop("opname")
        _class = instruction.pop("class", None)
        opcode = instruction.pop("opcode")
        _version = instruction.pop("version", None)
        _lastversion = instruction.pop("lastVersion", None)
        operands = instruction.pop("operands", [])
        _capabilities = instruction.pop("capabilities", [])
        _extensions = instruction.pop("extensions", [])
        assert len(instruction) == 0, instruction.keys()

        usednames = []
        varnames = [
            "x",
            "y",
            "z",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
        ]
        print()
        print()
        hasarguments = False
        hasresult = False
        hasrtype = False
        for operand in operands:
            kind = operand.pop("kind")
            quantifier = operand.pop("quantifier", None)
            name = operand.pop("name", None)
            match kind:
                case "IdResult":
                    hasresult = True
                    continue
                case "IdResultType":
                    hasrtype = True
                    varname = "rtype"
                case _:
                    varname = varnames.pop(0)
                    usednames.append(varname)
            if not hasarguments:
                print(f"def {opname}(")
                hasarguments = True
            assert len(operand) == 0, operand.keys()
            t = kind_types[kind]
            comment = ""
            if name is not None:
                comment = f"  # {" ".join(name.splitlines())}"
            match quantifier:
                case None:
                    print(f"    {varname}: {t},{comment}")
                case "?":
                    print(f"    {varname}: Optional[{t}] = None,{comment}")
                case "*":
                    print(f"    *{varname}: {t},{comment}")
        if not hasarguments:
            print(f"def {opname}() -> SpirvInstruction:")
        else:
            print(") -> SpirvInstruction:")
        print("    return SpirvInstruction(")
        print(f"        {opcode=},")
        print(f"        args=[{", ".join(usednames)}],")
        print(f"        {hasresult=},")
        print(f"        {hasrtype=},")
        print("    )")


if __name__ == "__main__":
    main()
