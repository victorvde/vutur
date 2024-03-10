from typing import Iterator, Optional, Callable, Any

from dataclasses import dataclass


@dataclass
class NeedsFragmentation(Exception):
    msg: str
    size_hint: int


class OutOfMemory(Exception):
    msg: str


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
    chunk: object
    allocation: int
    offset: int
    size: int


@dataclass
class SplitUsed:
    offset: int
    size: int
    used: Used


class SubAllocator:
    def __init__(self, size: int, chunk: object):
        self.size = size
        self.chunk = chunk

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
        self.used.remove(SubUsed(offset, size))

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

        self.suballocators: dict[int, SubAllocator] = {}

    def free_list(self) -> Iterator[Free]:
        for i, s in self.suballocators.items():
            for f in s.free_list():
                yield Free(i, f.index, f.offset, f.size)

    # returns the smallest free area bigger than size, or the largest free aeea if there isn't a big enough free area
    # todo: smarter data structure?
    def best_fit(self, size: int) -> Free:
        best_fit = Free(-1, 0, 0, 0)  # empty
        for free in self.free_list():
            if best_fit.size < size:
                if free.size > best_fit.size:
                    best_fit = free
            else:
                if free.size < best_fit.size and free.size >= size:
                    best_fit = free

        return best_fit

    def calculate_currently_allocated(self) -> int:
        return sum(
            s.calculate_currently_allocated() for s in self.suballocators.values()
        )

    def calculate_total_chunk_size(self) -> int:
        return sum(s.size for s in self.suballocators.values())

    def allocate(self, size: int, allocate_chunk: Callable[[int], object]) -> Used:
        assert size > 0, "Can't allocate nothing"

        # round up to alignment
        size = (size + self.alignment - 1) // self.alignment * self.alignment

        currently_allocated = (
            self.calculate_currently_allocated()
        )  # todo: don't recalculate every time
        if currently_allocated + size > self.max_memory:
            raise OutOfMemory(
                f"tried to allocate {size} but already allocated {currently_allocated} out of maximum {self.max_memory}"
            )

        if size > self.max_contiguous_size:
            raise NeedsFragmentation(
                f"tried to allocate {size} which is bigger than max {self.max_contiguous_size}",
                self.max_contiguous_size,
            )

        def new_chunk(chunk_size: int) -> Optional[int]:
            try:
                chunk = allocate_chunk(chunk_size)
            except OutOfMemory:
                return None
            s = SubAllocator(chunk_size, chunk)
            i = 0
            while i in self.suballocators:
                i += 1
            self.suballocators[i] = s
            return i

        fit = self.best_fit(size)
        if fit.size < size:
            chunk_size = self.default_chunk_size
            while size <= chunk_size:
                s = new_chunk(chunk_size)
                if s is not None:
                    break
                chunk_size //= 2
            else:
                s = new_chunk(size)
                if s is None:
                    if fit.size == 0:
                        raise OutOfMemory(
                            f"tried to allocate {size} but already allocated {currently_allocated} and allocation failed"
                        )
                    else:
                        raise NeedsFragmentation(
                            f"Could not allocate new chunk of {size}, try splitting and usgin {fit.size} first",
                            fit.size,
                        )

            fit = self.best_fit(size)  # todo: only check new chunk?
            assert fit.size >= size, "We just allocated a chunk so it should fit"

        self.suballocators[fit.allocation].insert(fit.index, fit.offset, size)

        return Used(
            self.suballocators[fit.allocation].chunk, fit.allocation, fit.offset, size
        )

    def allocate_split(
        self,
        size: int,
        allocate_chunk: Callable[[int], object],
        free_chunk: Callable[[Any], None],
    ) -> list[SplitUsed]:
        splits: list[SplitUsed] = []
        offset = 0
        split_size = size
        while size > 0:
            try:
                u = self.allocate(split_size, allocate_chunk)
            except NeedsFragmentation as e:
                split_size = e.size_hint
            except OutOfMemory:
                for s in splits:
                    self.free(s.used, free_chunk)
                raise
            else:
                size -= split_size
                offset += split_size
                splits.append(SplitUsed(offset, split_size, u))
                split_size = size
        return splits

    def free(self, used: Used, free_chunk: Callable[[Any], None]) -> None:
        s = self.suballocators[used.allocation]
        s.remove(used.offset, used.size)
        if len(s.used) == 0:
            free_chunk(s.chunk)
            del self.suballocators[used.allocation]

    def free_split(
        self, used: list[SplitUsed], free_chunk: Callable[[Any], None]
    ) -> None:
        for u in used:
            self.free(u.used, free_chunk)

    def sanity_check(self) -> None:
        for s in self.suballocators.values():
            s.sanity_check()
