from vutur.vulkan import VulkanContext

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
    for seperate in [False, True]:
        c = VulkanContext(prefer_separate_memory=seperate)
        d = c.suballocate_device(100)

        data = bytearray(100)
        c.upload(d, data)

        d.destroy()
        c.destroy()


def test_download() -> None:
    for seperate in [False, True]:
        c = VulkanContext(prefer_separate_memory=seperate)
        d = c.suballocate_device(100)

        data = bytearray(100)
        for i in range(100):
            data[i] = (i * 7) % 256

        c.upload(d, data)

        data2 = c.download(d)

        assert data == data2

        d.destroy()
        c.destroy()
