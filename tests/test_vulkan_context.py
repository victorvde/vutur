from vutur.vulkan_context import VulkanContext


def test_init() -> None:
    c = VulkanContext()
    assert c.instance is not None
    del c

def test_alloc() -> None:
    c = VulkanContext()
    h = c.suballocate_host(100)
    d = c.suballocate_device(100)
    del h
    del d
    del c

def test_alloc_ooo() -> None:
    c = VulkanContext()
    h = c.suballocate_host(100)
    d = c.suballocate_device(100)
    del c
    del h
    del d
