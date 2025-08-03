from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(slots=True)
class ModuleField:
    pass


@dataclass(slots=True)
class Field:
    trainable: bool = False
    dynamic: bool = False
    save: bool = True


@dataclass(slots=True)
class FieldWrapper:
    value: Any
    meta: Field

def Parameter(value: Any) ->FieldWrapper:
    return FieldWrapper(value, Field(trainable=True))


def Static(value: Any) ->FieldWrapper:
    return FieldWrapper(value, Field())


def Dynamic(value: Any) ->FieldWrapper:
    return FieldWrapper(value, Field(dynamic=True))


def Ephemeral(value: Any) ->FieldWrapper:
    return FieldWrapper(value, Field(save=False))


ModuleFilterType: TypeAlias = Callable[[list[tuple["Module", str]], Any, Field], bool]


@dataclass(slots=True)
class ModulePath:
    module: "Module"
    field_name: str


@dataclass(slots=True)
class DirectValue:
    value: Any


@dataclass(slots=True)
class Reference:
    source: int | None
    index: int


@dataclass
class ModuleDef:
    skeleton: list[tuple[type, dict[str, Field | ModuleField]]]
    values: list[Any]

    def combine(self, *splits: list[list[Any]]) -> "Module":
        skeleton_it = iter(self.skeleton)
        values_it = iter(self.values)

        def inner() -> "Module":
            mtype, fields = next(skeleton_it)
            m : Module = object.__new__(mtype)
            for name, field in fields.items():
                match field:
                    case ModuleField():
                        value : Any = inner()
                    case Field():
                        w = next(values_it)
                        match w:
                            case DirectValue():
                                value = w.value
                            case Reference():
                                if w.source is None:
                                    value = self.values[w.index].value
                                else:
                                    value = splits[w.source][w.index]
                object.__setattr__(m, name, value)
            return m

        return inner()


class Module:
    _fields: dict[str, Field | ModuleField]

    def forward(*args: Any, **kwargs: Any) -> Any:
        """Override this method to define what this module does."""
        raise NotImplementedError

    def __init__(self) -> None:
        self._fields = {}

    def __setattr__(self, name:str, value:Any) -> None:
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

    def __delattr__(self, name:str) -> None:
        del self._fields[name]
        object.__delattr__(self, name)

    def split(self, *filters: ModuleFilterType) -> tuple[Any, ...]:
        # retyrb values
        skeleton = []
        values = []
        split_values : list[list[Any]] = [[] for _ in filters]

        # temporary
        path = []
        uniques : dict[int, Reference] = {}

        def append_value(v: Any, i: int) -> None:
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
                            append_value(value, i)

                path.pop()

        inner(self)

        return (ModuleDef(skeleton, values), *split_values)
