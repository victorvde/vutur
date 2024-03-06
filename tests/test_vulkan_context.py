from vutur.vulkan_context import VulkanContext


def test_init() -> None:
    c = VulkanContext()
    assert c.instance is not None
    del c
