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


def test_module_split_combine() -> None:
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


def test_module_split_combine_resplit() -> None:
    m = Main()

    md, (p, d) = m.split(is_trainable, is_dynamic)

    for i in range(5):
        assert m.s == 3
        assert m.p == [0.0, 0.0, 0.0]
        assert m.d == i
        assert m.m1.w == 1.0
        assert m.m2.b == 3.0

        r = m(1.0)
        assert r == 11.0

        pr, dr = md.resplit(m)
        assert len(pr) == len(p)
        assert len(dr) == len(d)

        m = md.combine(pr, dr)


def test_module_surgery() -> None:
    m = Main()

    m.m2 = m.m1
    m.s = Trainable(m.s)

    assert m(1.0) == 4.0

    del m.d

    m.m1.w = 2

    md, (p, d) = m.split(is_trainable, is_dynamic)

    i = p.index(m.s)
    p[i] = 11

    m = md.combine(p, d)

    assert m.s == 11

    m.d = Dynamic(1)

    assert m(1.0) == 48.0
