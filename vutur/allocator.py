"""
Turns big chunks of memory into smaller allocations.
Tailored to Vulkan, for example by being able to split allocations bigger than the maximum buffer size into multiple allocations.
"""

from typing import Iterator, Optional, Callable, Any

from dataclasses import dataclass

__all__ = [
    "NeedsFragmentation",
    "OutOfMemory",
    "Allocation",
    "Allocator",
]


@dataclass
class NeedsFragmentation(Exception):
    """Cannot allocate a single allocation of the requested size."""

    msg: str
    size_hint: int
    """Try using this size instead."""


class OutOfMemory(Exception):
    "There is no way to allocate the requested size."

    msg: str


@dataclass
class SubFree:
    index: int
    offset: int
    size: int


@dataclass
class SubAllocation:
    offset: int
    size: int


@dataclass
class Free:
    chunk_idx: int
    index: int
    offset: int
    size: int


@dataclass
class Allocation:
    """A single allocation within one of the chunks"""

    chunk: object
    """The chunk object as returned by the user specified chunk allocation function."""
    chunk_idx: int
    """The index of the chunk in `Allocator.chunks`."""
    offset: int
    """Offset within the chunk."""
    size: int
    """Size of the allocation."""


class SubAllocator:
    def __init__(self, size: int, chunk: object) -> None:
        self.size = size
        self.chunk = chunk

        self.allocations: list[SubAllocation] = []

    def free_list(self) -> Iterator[SubFree]:
        prev = 0
        for i, a in enumerate(self.allocations):
            free = a.offset - prev
            yield SubFree(i, prev, free)
            prev = a.offset + a.size
        free = self.size - prev
        yield SubFree(len(self.allocations), prev, free)

    def calculate_currently_allocated(self) -> int:
        return sum(a.size for a in self.allocations)

    def insert(self, index: int, offset: int, size: int) -> None:
        self.allocations.insert(index, SubAllocation(offset, size))

    def remove(self, offset: int, size: int) -> None:
        self.allocations.remove(SubAllocation(offset, size))

    def sanity_check(self) -> None:
        prev = 0
        for a in self.allocations:
            assert a.offset >= prev
            prev = a.offset + a.size
            assert prev <= self.size


class Allocator:
    def __init__(
        self,
        alignment: int,
        max_memory: int,
        max_contiguous_size: int,
        default_chunk_size: int,
    ) -> None:
        self.alignment = alignment
        """Minimum alignment of each allocation (e.g. `VkMemoryRequirements.alignment`), sizes are rounded up to this."""
        self.max_memory = max_memory
        """
        Absolute maximum amount of memory that can be allocated (e.g. `VkPhysicalDeviceVulkan11Properties.maxMemoryAllocationSize`).
        Only used as a quick check, the user defined allocation function is allowed to fail before this value.
        """
        self.max_contiguous_size = max_contiguous_size
        """Maximum contiguous size (e.g. `VkPhysicalDeviceLimits.maxStorageBufferRange`)."""
        self.default_chunk_size = default_chunk_size
        """
        Try to allocate at least this size per chunk.
        Too small creates too many chunks, too big wastes memory that other programs could use.
        Something like `max_memory // 16` is a decent trade-off.
        """
        self.suballocators: dict[int, SubAllocator] = {}
        """@private"""

    def free_list(self) -> Iterator[Free]:
        """
        @private
        iterator over every hole between the allocated blocks and the hole at the end
        """
        for i, s in self.suballocators.items():
            for f in s.free_list():
                yield Free(i, f.index, f.offset, f.size)

    def best_fit(self, size: int) -> Free:
        """
        @private
        returns the smallest free area bigger than size, or the largest free aeea if there isn't a big enough free area
        """
        # todo: smarter data structure?
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
        """Returns total size of all allocations."""
        return sum(
            s.calculate_currently_allocated() for s in self.suballocators.values()
        )

    def allocate(
        self, size: int, allocate_chunk: Callable[[int], object]
    ) -> Allocation:
        """
        Create a single allocation.
        * `size`: requested size for the allocation
        * `allocate_chunk`: user-defined chunk allocation function (e.g. `vkAllocateMemory`/`vkCreateBuffer`).
          May raise `OutOfMemory`.

        Can raise `NeedsFragmentation` or `OutOfMemory`.
        """
        assert size > 0, "Can't allocate nothing"

        # round up to alignment
        size = (size + self.alignment - 1) // self.alignment * self.alignment

        # this check is not strictly necessary, but probably a good idea especially for allocate_split
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
            """create a chunk and its suballocator"""
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

        self.suballocators[fit.chunk_idx].insert(fit.index, fit.offset, size)

        return Allocation(
            self.suballocators[fit.chunk_idx].chunk,
            fit.chunk_idx,
            fit.offset,
            size,
        )

    def allocate_split(
        self,
        size: int,
        allocate_chunk: Callable[[int], object],
        free_chunk: Callable[[Any], None],
    ) -> list[Allocation]:
        """
        Create an allocation that can be split into smaller pieces.
        * `size`: requested size for the allocation.
        * `allocate_chunk`: user-defined chunk allocation function (e.g. `vkAllocateMemory`/`vkCreateBuffer`).
          May raise `OutOfMemory`.
        * `free_chunk`: user-defined chunk free function (e.g. `vkFreeMemory`/`vkDestroyBuffer`).

        Can raise `OutOfMemory`.
        """
        splits: list[Allocation] = []
        offset = 0
        split_size = size
        while size > 0:
            try:
                a = self.allocate(split_size, allocate_chunk)
            except NeedsFragmentation as e:
                split_size = e.size_hint
            except OutOfMemory:
                for a in splits:
                    self.free(a, free_chunk)
                raise
            else:
                size -= split_size
                offset += split_size
                splits.append(a)
                split_size = size
        return splits

    def free(self, allocation: Allocation, free_chunk: Callable[[Any], None]) -> None:
        """
        Free an allocation.
        * `allocation`: what to free.
        * `free_chunk`: user-defined chunk free function (e.g. `vkFreeMemory`/`vkDestroyBuffer`).
        """
        s = self.suballocators[allocation.chunk_idx]
        s.remove(allocation.offset, allocation.size)
        if len(s.allocations) == 0:
            free_chunk(s.chunk)
            del self.suballocators[allocation.chunk_idx]

    def free_split(
        self, splitallocation: list[Allocation], free_chunk: Callable[[Any], None]
    ) -> None:
        """
        Free a split allocation.
        * `splitallocation`: what to free.
        * `free_chunk`: user-defined chunk free function (e.g. `vkFreeMemory`/`vkDestroyBuffer`).
        """
        for a in splitallocation:
            self.free(a, free_chunk)

    def sanity_check(self) -> None:
        """
        Run internal consistency checks, for testing/debugging purposes.
        """
        for s in self.suballocators.values():
            s.sanity_check()

    def chunks(self) -> dict[int, object]:
        """
        Get a `dict` of chunk index to chunk objects.
        """
        return {i: s.chunk for i, s in self.suballocators.items()}
