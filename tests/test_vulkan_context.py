from vutur.vulkan_context import VulkanContext

import pytest

def test_init() -> None:
    with pytest.raises(ValueError):
        c = VulkanContext("this device does not exist")
    c = VulkanContext()
    assert c.instance is not None
    del c


def test_alloc() -> None:
    c = VulkanContext()
    d = c.suballocate_device(100)
    d.destroy()
    c.destroy()


def test_upload() -> None:
    c = VulkanContext()
    d = c.suballocate_device(100)

    data = bytearray(100)
    c.upload(d, memoryview(data))

    del c
    del d
