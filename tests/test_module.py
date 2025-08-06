from vutur.module import Module, Trainable, Static, Dynamic, is_trainable, is_dynamic


class SubModule(Module):
    w: float
    b: float

    def __init__(self, w: float, b: float) -> None:
        self.w = Trainable(w)
        self.b = Trainable(b)

    def __call__(self, x: float) -> float:
        return self.w * x + self.b


class Main(Module):
    def __init__(self) -> None:
        self.s = Static(3)
        self.p = Trainable([0.0] * self.s)
        self.d = Dynamic(0)
        self.m1 = SubModule(1.0, 0.0)
        self.m2 = SubModule(2.0, 3.0)

    def __call__(self, x: float) -> float:
        self.d += 1
        return self.m2(self.m1(self.s + x + sum(self.p)))


def test_module() -> None:
    m = Main()

    for i in range(5):
        assert m.s == 3
        assert m.p == [0.0, 0.0, 0.0]
        assert m.d == i
        assert m.m1.w == 1.0
        assert m.m2.b == 3.0

        r = m(1.0)
        assert r == 11.0

        md, (p, d) = m.split(is_trainable, is_dynamic)

        m = md.combine(p, d)
