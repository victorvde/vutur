import pytest

from vutur.allocator import Allocator, OutOfMemory, NeedsFragmentation

class Memory:
    def __init__(self):
        self.chunks = []

    def allocate_succeed(self, size: int) -> int:
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

    self.allocations: list[int] = list()


def test_small(a:Allocator,m:Memory) -> None:
    u = a.allocate(1, m.allocate_succeed)
    assert u.chunk == 1024
    assert u.offset == 0
    assert u.size >= 1
    assert len(m.chunks) == 1
    a.free(u, m.free)
    assert len(m.chunks) == 0

def test_big(a:Allocator,m:Memory) -> None:
    u = a.allocate(2048, m.allocate_succeed)
    assert u.chunk == 2048
    assert u.offset == 0
    assert u.size >= 2048

def test_too_big(a:Allocator,m:Memory) -> None:
    with pytest.raises(NeedsFragmentation):
        a.allocate(3072, m.allocate_succeed)

    a.allocate(3072 // 2, m.allocate_succeed)
    a.allocate(3072 // 2, m.allocate_succeed)

def test_way_too_big(a:Allocator,m:Memory) -> None:
    with pytest.raises(OutOfMemory):
        a.allocate(8192, m.allocate_succeed)
