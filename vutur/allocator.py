from typing import Iterator, Optional, Callable, Any

from dataclasses import dataclass


class NeedsFragmentation(Exception):
    # todo: size hint?
    pass


class OutOfMemory(Exception):
    pass


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
    chunk: Any
    allocation: int
    offset: int
    size: int


class SubAllocator:
    def __init__(self, size: int, chunk: Any):
        self.size = size
        self.chunk = Any

        self.used: list[SubUsed] = []

    # iterator over every hole between the used blocks and the hole at the end
    def free_list(self) -> Iterator[SubFree]:
        prev = 0
        for i, u in enumerate(self.used):
            free = u.offset - prev
            yield SubFree(i, prev, free)
            prev = u.offset + u.size
        free = self.size - prev
        yield SubFree(len(self.used), prev, free)

    def calculate_currently_allocated(self) -> int:
        return sum(u.size for u in self.used)

    def insert(self, index: int, offset: int, size: int) -> None:
        self.used.insert(index, SubUsed(offset, size))

    def remove(self, offset: int, size: int) -> None:
        raise NotImplementedError

    def sanity_check(self) -> None:
        prev = 0
        for u in self.used:
            assert u.offset >= prev
            prev = u.offset + u.size
            assert prev <= self.size


class Allocator:
    def __init__(
        self,
        alignment: int,
        max_memory: int,
        max_contiguous_size: int,
        default_chunk_size: int,
    ):
        self.alignment = alignment
        self.max_memory = max_memory
        self.max_contiguous_size = max_contiguous_size
        self.default_chunk_size = default_chunk_size

        self.suballocators: list[SubAllocator] = []

    def free_list(self) -> Iterator[Free]:
        for i, s in enumerate(self.suballocators):
            for f in s.free_list():
                yield Free(i, f.index, f.offset, f.size)

    # todo: smarter data structure?
    def best_fit(self, size: int) -> Optional[Free]:
        best_fit = None
        for free in self.free_list():
            if free.size < size:
                continue

            if best_fit is None or best_fit.size > free.size:
                best_fit = free

        return best_fit

    def calculate_currently_allocated(self) -> int:
        return sum(s.calculate_currently_allocated() for s in self.suballocators)

    def calculate_total_chunk_size(self) -> int:
        return sum(s.size for s in self.suballocators)

    def allocate(self, size: int, allocate_chunk: Callable[[int], Any]) -> Used:
        # round up to alignment
        size = (size + self.alignment - 1) // self.alignment * self.alignment

        currently_allocated = (
            self.calculate_currently_allocated()
        )  # todo: don't recalculate every time
        if currently_allocated + size > self.max_memory:
            raise OutOfMemory(
                f"tried to allocate {size} but already allocated {currently_allocated} out of {self.max_memory}"
            )

        if size > self.max_contiguous_size:
            raise NeedsFragmentation(
                f"tried to allocate {size} which is bigger than max {self.max_contiguous_size}"
            )

        def new_chunk(chunk_size: int) -> Optional[int]:
            try:
                chunk = allocate_chunk(chunk_size)
            except OutOfMemory:
                return None
            s = SubAllocator(size, chunk)
            self.suballocators.append(s)
            return len(self.suballocators) - 1

        fit = self.best_fit(size)
        if fit is None:
            chunk_size = self.default_chunk_size
            while size <= chunk_size:
                s = new_chunk(chunk_size)
                if s is not None:
                    break
                chunk_size //= 2
            else:
                s = new_chunk(size)
                if s is None:
                    if self.calculate_total_chunk_size() == currently_allocated:
                        raise OutOfMemory
                    else:
                        raise NeedsFragmentation

        fit = self.best_fit(size)  # todo: only check new chunk?
        assert fit is not None, "We just allocated a chunk so it should fit"

        self.suballocators[fit.allocation].insert(fit.index, fit.offset, size)

        return Used(
            self.suballocators[fit.allocation].chunk, fit.allocation, fit.offset, size
        )

    def subfree(self, used: Used) -> None:
        self.suballocators[used.allocation].remove(used.offset, used.size)

        pass

    def sanity_check(self) -> None:
        for s in self.suballocators:
            s.sanity_check()
