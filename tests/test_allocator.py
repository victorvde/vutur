import pytest

pytest.register_assert_rewrite("vutur.allocator")

from vutur.allocator import Allocator, OutOfMemory, NeedsFragmentation  # noqa: E402


class Memory:
    def __init__(self) -> None:
        self.chunks: list[int] = []

    def allocate(self, size: int) -> int:
        if sum(self.chunks) + size > 3072:
            raise OutOfMemory
        self.chunks.append(size)
        return size

    def free(self, chunk: int) -> None:
        self.chunks.remove(chunk)


@pytest.fixture
def m() -> Memory:
    return Memory()


@pytest.fixture
def a() -> Allocator:
    return Allocator(
        alignment=64,
        max_memory=4096,
        max_contiguous_size=2048,
        default_chunk_size=1024,
    )


def test_small(a: Allocator, m: Memory) -> None:
    u = a.allocate(1, m.allocate)
    assert u.chunk == 1024
    assert u.offset == 0
    assert u.size >= 1
    assert len(m.chunks) == 1
    a.free(u, m.free)
    assert len(m.chunks) == 0


def test_big(a: Allocator, m: Memory) -> None:
    u = a.allocate(2048, m.allocate)
    assert u.chunk == 2048
    assert u.offset == 0
    assert u.size >= 2048


def test_too_big(a: Allocator, m: Memory) -> None:
    with pytest.raises(NeedsFragmentation):
        a.allocate(3072, m.allocate)

    us = a.allocate_split(3072, m.allocate, m.free)
    assert len(us) == 2

    a.free_split(us, m.free)
    assert len(m.chunks) == 0


def test_way_too_big(a: Allocator, m: Memory) -> None:
    with pytest.raises(OutOfMemory):
        a.allocate(8192, m.allocate)


def test_oom(a: Allocator, m: Memory) -> None:
    u1 = a.allocate(1024, m.allocate)
    u2 = a.allocate(1024, m.allocate)
    u3 = a.allocate(1024, m.allocate)
    with pytest.raises(OutOfMemory):
        a.allocate(1024, m.allocate)
    a.free(u1, m.free)
    a.free(u2, m.free)
    a.free(u3, m.free)
    assert len(m.chunks) == 0


def test_fragment(a: Allocator, m: Memory) -> None:
    u1 = a.allocate(768, m.allocate)
    u2 = a.allocate(768, m.allocate)
    u3 = a.allocate(768, m.allocate)
    with pytest.raises(NeedsFragmentation):
        a.allocate(768, m.allocate)
    us = a.allocate_split(768, m.allocate, m.free)
    assert len(us) == 3
    a.free_split(us, m.free)
    a.free(u1, m.free)
    a.free(u2, m.free)
    a.free(u3, m.free)
    assert len(m.chunks) == 0


def test_random(a: Allocator, m: Memory) -> None:
    import random

    random.seed(0)

    ul = []
    usl = []

    for i in range(10000):
        c = random.choice(["ALLOC", "ALLOC_SPLIT", "FREE"])
        if c == "ALLOC":
            size = random.randrange(1, 2500)
            try:
                u = a.allocate(size, m.allocate)
            except NeedsFragmentation:
                pass
            except OutOfMemory:
                pass
            else:
                assert size <= 2048
                ul.append(u)
        elif c == "ALLOC_SPLIT":
            size = random.randrange(1, 2500)
            try:
                us = a.allocate_split(size, m.allocate, m.free)
            except OutOfMemory:
                assert a.calculate_currently_allocated() + size > 3072
                pass
            else:
                assert size <= 2048
                usl.append(us)
        elif c == "FREE":
            if len(ul) > 0:
                ui = random.randrange(len(ul))
                a.free(ul[ui], m.free)
                del ul[ui]
        elif c == "FREE_SPLIT":
            if len(usl) > 0:
                ui = random.randrange(len(usl))
                a.free_split(usl[ui], m.free)
                del usl[ui]
        else:
            assert False, c
        a.sanity_check()

    for u in ul:
        a.free(u, m.free)
    for us in usl:
        a.free_split(us, m.free)
    assert len(m.chunks) == 0
