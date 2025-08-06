from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, TypeAlias


@dataclass(slots=True)
class ModuleField:
    pass


@dataclass(slots=True)
class Field:
    trainable: bool = False
    dynamic: bool = False


@dataclass(slots=True)
class FieldWrapper:
    value: object
    meta: Field


def Trainable[T](value: T) -> T:
    return FieldWrapper(value, Field(trainable=True))  # type: ignore


def Static[T](value: T) -> T:
    return FieldWrapper(value, Field())  # type: ignore


def Dynamic[T](value: T) -> T:
    return FieldWrapper(value, Field(dynamic=True))  # type: ignore


Path: TypeAlias = list[tuple["Module", str]]
ModuleFilterType: TypeAlias = Callable[[Path, Field, object], bool]


def is_trainable(_m: Path, f: Field, _v: object) -> bool:
    return f.trainable


def is_dynamic(_p: Path, f: Field, _v: object) -> bool:
    return f.dynamic


@dataclass(slots=True)
class ModulePath:
    module: "Module"
    field_name: str


@dataclass(slots=True)
class DirectValue:
    value: object


@dataclass(slots=True)
class Reference:
    source: int | None
    index: int


@dataclass
class ModuleDef[T]:
    skeleton: list[tuple[type, dict[str, Field | ModuleField]]]
    values: list[DirectValue | Reference]

    def combine(self, *splits: list[object]) -> T:
        skeleton_it = iter(self.skeleton)
        values_it = iter(self.values)

        def inner() -> T:
            mtype, fields = next(skeleton_it)
            m: T = object.__new__(mtype)
            object.__setattr__(m, "_fields", fields)
            for name, field in fields.items():
                match field:
                    case ModuleField():
                        value: object = inner()
                    case Field():
                        w = next(values_it)
                        match w:
                            case DirectValue():
                                value = w.value
                            case Reference():
                                if w.source is None:
                                    box = self.values[w.index]
                                    assert isinstance(box, DirectValue)
                                    value = box.value
                                else:
                                    value = splits[w.source][w.index]
                object.__setattr__(m, name, value)
            return m

        return inner()


class Module:
    _fields: dict[str, Field | ModuleField]

    def __setattr__(self, name: str, value: object) -> None:
        if not hasattr(self, "_fields"):
            object.__setattr__(self, "_fields", {})

        match value:
            case FieldWrapper():
                self._fields[name] = value.meta
                new_value = value.value
            case Module():
                self._fields[name] = ModuleField()
                new_value = value
            case _:
                assert name in self._fields, (
                    f"When creating a module attribute, you need to wrap value {value}"
                )
                new_value = value
        object.__setattr__(self, name, new_value)

    def __delattr__(self, name: str) -> None:
        del self._fields[name]
        object.__delattr__(self, name)

    def split(
        self, *filters: ModuleFilterType
    ) -> tuple[ModuleDef[Self], list[list[object]]]:
        # retyrb values
        skeleton = []
        values: list[DirectValue | Reference] = []
        split_values: list[list[object]] = [[] for _ in filters]

        # temporary
        path = []
        uniques: dict[int, Reference] = {}

        def append_value(v: object, i: int | None) -> None:
            ref = uniques.get(id(v))
            if ref is not None:
                values.append(ref)
                return

            if i is None:
                ref = Reference(i, len(values))
                values.append(DirectValue(v))
            else:
                dest = split_values[i]
                ref = Reference(i, len(dest))
                index = len(dest)
                dest.append(v)
                values.append(Reference(i, index))
            uniques[id(v)] = ref

        def inner(m: "Module") -> None:
            skeleton.append((type(m), m._fields))
            for name, field in m._fields.items():
                path.append((m, name))

                value = getattr(m, name)
                match field:
                    case ModuleField():
                        inner(value)
                    case Field():
                        for i, f in enumerate(filters):
                            if f(path, field, value):
                                append_value(value, i)
                                break
                        else:
                            append_value(value, None)

                path.pop()

        inner(self)

        return ModuleDef(skeleton, values), split_values
