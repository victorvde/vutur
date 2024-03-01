from typing import Iterator, Optional

from dataclasses import dataclass


@dataclass
class SubFree:
    index: int
    offset: int
    size: int


@dataclass
class SubUsed:
    offset: int
    size: int


@dataclass
class Free:
    allocation: int
    index: int
    offset: int
    size: int


@dataclass
class Used:
    allocation: int
    index: int
    offset: int
    size: int


class SubAllocator:
    def __init__(self, size: int):
        self.size = size

        self.used: list[SubUsed] = []
        self.usedsize = 0

    # iterator over every hole between the used blocks
    # should probably be some optimized data structure instead
    def free_list(self) -> Iterator[SubFree]:
        prev = 0
        for i, u in enumerate(self.used):
            free = u.offset - prev
            yield SubFree(i, prev, free)
            prev = u.offset + u.size
        free = self.size - prev
        yield SubFree(len(self.used), prev, free)

    def insert(self, index: int, offset: int, size: int) -> None:
        self.used.insert(index, SubUsed(offset, size))


class Allocator:
    def __init__(self, alignment: int):
        self.alignment = alignment

        # self.allocations = []
        self.suballocators: list[SubAllocator] = []

    def free_list(self) -> Iterator[Free]:
        for i, s in enumerate(self.suballocators):
            for f in s.free_list():
                yield Free(i, f.index, f.offset, f.size)

    def suballocate(self, size: int) -> Optional[Used]:
        # round up to alignment
        size = (size + self.alignment - 1) // self.alignment * self.alignment

        best_fit = None
        for free in self.free_list():
            if free.size > size:
                if best_fit is None or best_fit.size > free.size:
                    best_fit = free

        if best_fit is None:
            return None

        self.suballocators[best_fit.allocation].insert(
            best_fit.index, best_fit.offset, size
        )

        return Used(best_fit.allocation, best_fit.index, best_fit.offset, size)
