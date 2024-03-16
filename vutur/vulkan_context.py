# adapted from https://github.com/realitix/vulkan/blob/master/example/contribs/example_mandelbrot_compute.py
# which is a port from https://github.com/Erkaman/vulkan_minimal_compute

import os
import logging
from typing import Any, Optional

from vutur.allocator import Allocator

import vulkan as vk

DEBUG_LAYER = "VK_LAYER_KHRONOS_validation"
DEBUG_EXTENSION = "VK_EXT_debug_utils"


# utils
def cs(c: object) -> str:
    """Convert vulkan ffi string to str"""
    return vk.ffi.string(c).decode()


def filter_set(available: set[str], optional: set[str], required: set[str]) -> set[str]:
    """Get extensions/layers to activate given optional and required ones."""
    missing = required - available
    if len(missing) > 0:
        raise ValueError(f"Missing required Vulkan: {missing}")
    return available & (optional | required)


def debug_callback(severity: int, messagetype: int, data: Any, _userdata: Any) -> bool:
    message = f"VK [{cs(data.pMessageIdName)}] [{cs(data.pMessage)}]"
    if severity & vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        logging.error(message)
    elif severity & vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        logging.warning(message)
    elif severity & vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        logging.info(message)
    elif severity & vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        logging.debug(message)
    else:
        assert False, severity
    return False


class VulkanContext:
    instance: object  # vk.VkInstance
    layers: set[str]
    extensions: set[str]
    debug_callback: object  # vk.VkDebugUtilsMessengerEXT
    physicaldevice: object  # vk.VkPhysicalDevice
    physicaldevice_properties: vk.VkPhysicalDeviceProperties2
    queuefamily: int
    device: object  # vk.VkDevice
    queue: object  # vk.VkQueue
    commandpool: object  # vk.vkCommandPool
    memory_properties: vk.VkPhysicalDeviceMemoryProperties2
    buffer_create_info: vk.VkBufferCreateInfo
    buffer_requirements: vk.VkMemoryRequirements
    host_memory: int
    device_memory: int
    host_allocator: Allocator
    device_allocator: Allocator

    def __init__(self, device_filter: Optional[str] = None) -> None:
        if device_filter is None:
            device_filter = os.getenv("VUTUR_DEVICE", "")

        self.create_instance(
            version=vk.VK_MAKE_VERSION(1, 1, 0),
            opt_layers={DEBUG_LAYER},
            req_layers=set(),
            opt_extensions={vk.VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
            req_extensions=set(),
        )
        self.create_debug_callback()
        self.create_physical_device(device_filter)
        self.create_device()
        self.create_commandpool()
        self.create_memories()
        self.host_allocator = self.create_allocator(self.host_memory)
        if self.host_memory == self.device_memory:
            self.device_allocator = self.host_allocator
        else:
            self.device_allocator = self.create_allocator(self.device_memory)

    def __del__(self) -> None:
        if hasattr(self, "commandpool"):
            vk.vkDestroyCommandPool(self.device, self.commandpool, None)
        if hasattr(self, "debug_callback"):
            func = vk.vkGetInstanceProcAddr(
                self.instance, "vkDestroyDebugUtilsMessengerEXT"
            )
            assert func
            func(self.instance, self.debug_callback, None)
        if hasattr(self, "device"):
            vk.vkDestroyDevice(self.device, None)
        if hasattr(self, "instance"):
            vk.vkDestroyInstance(self.instance, None)

    def create_instance(
        self,
        version: int,
        opt_layers: set[str],
        req_layers: set[str],
        opt_extensions: set[str],
        req_extensions: set[str],
    ) -> None:
        available_layers = {
            prop.layerName for prop in vk.vkEnumerateInstanceLayerProperties()
        }
        self.layers = filter_set(available_layers, opt_layers, set())

        available_extensions = {
            prop.extensionName
            for prop in vk.vkEnumerateInstanceExtensionProperties(None)
        }
        self.extensions = filter_set(
            available_extensions, opt_extensions, req_extensions
        )

        logging.debug(f"{available_layers=}")
        logging.debug(f"{available_extensions=}")
        logging.debug(f"{self.layers=}")
        logging.debug(f"{self.extensions=}")

        applicationInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=None,
            applicationVersion=0,
            pEngineName="vutur",
            engineVersion=0,
            apiVersion=version,
        )

        flags = 0
        if vk.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME in self.extensions:
            flags |= vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT

        createInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            flags=flags,
            pApplicationInfo=applicationInfo,
            enabledLayerCount=len(self.layers),
            ppEnabledLayerNames=self.layers,
            enabledExtensionCount=len(self.extensions),
            ppEnabledExtensionNames=self.extensions,
        )

        self.instance = vk.vkCreateInstance(createInfo, None)

    def create_debug_callback(self) -> None:
        if DEBUG_EXTENSION in self.extensions:
            func = vk.vkGetInstanceProcAddr(
                self.instance, "vkCreateDebugUtilsMessengerEXT"
            )
            cb = vk.VkDebugUtilsMessengerCreateInfoEXT(
                sType=vk.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                pNext=None,
                flags=0,
                messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                pfnUserCallback=debug_callback,
                pUserData=None,
            )
            self.debug_callback = func(self.instance, cb, None)

    def create_physical_device(self, device_filter: str) -> None:
        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        physical_devices_with = []
        for pd in physical_devices:
            props = vk.VkPhysicalDeviceProperties2(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            )
            vk.vkGetPhysicalDeviceProperties2(pd, props)
            name = cs(props.properties.deviceName)
            physical_devices_with.append((pd, props, name))
        # pick one
        for dtype in [
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU,
            vk.VK_PHYSICAL_DEVICE_TYPE_OTHER,
        ]:
            filtered = []
            for i, (pd, props, name) in enumerate(physical_devices_with):
                if device_filter not in name:
                    continue
                if props.properties.deviceType != dtype:
                    continue
                filtered.append(i)
            if len(filtered) > 1:
                raise ValueError(
                    f"Cannot choose between devices: {[physical_devices_with[i][2] for i in filtered]}"
                )
            elif len(filtered) == 1:
                break
            else:
                continue
        else:
            raise ValueError(f'No devices matching "{device_filter}" found')
        (
            self.physicaldevice,
            self.physicaldevice_properties,
            selected_name,
        ) = physical_devices_with[filtered[0]]
        logging.info(f"Selected device {selected_name}")

    def create_device(self) -> None:
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(
            self.physicaldevice
        )
        for i, props in enumerate(queue_families):
            if props.queueCount > 0 and props.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                break
        else:
            raise ValueError("No availabe compute queues on selected device")
        self.queuefamily = i

        qci = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queuefamily,
            queueCount=1,
            pQueuePriorities=[
                1.0
            ],  # we only have one queue, so this is not that imporant.
        )

        deviceFeatures = vk.VkPhysicalDeviceFeatures()
        dci = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            enabledLayerCount=len(self.layers),
            ppEnabledLayerNames=self.layers,
            pQueueCreateInfos=qci,
            queueCreateInfoCount=1,
            pEnabledFeatures=deviceFeatures,
        )

        self.device = vk.vkCreateDevice(self.physicaldevice, dci, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queuefamily, 0)

    def create_commandpool(self) -> None:
        cpci = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=0,
            queueFamilyIndex=self.queuefamily,
        )

        self.commandpool = vk.vkCreateCommandPool(self.device, cpci, None)

    def create_memories(self) -> None:
        self.memory_properties = vk.VkPhysicalDeviceMemoryProperties2(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        )
        vk.vkGetPhysicalDeviceMemoryProperties2(
            self.physicaldevice, self.memory_properties
        )

        self.buffer_create_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=1,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        temp_buffer = vk.vkCreateBuffer(self.device, self.buffer_create_info, None)
        self.buffer_requirements = vk.vkGetBufferMemoryRequirements(
            self.device, temp_buffer
        )
        vk.vkDestroyBuffer(self.device, temp_buffer, None)

        # see the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
        def find_memory_type(bits: int, properties: int) -> int:
            for i, mt in enumerate(self.memory_properties.memoryProperties.memoryTypes):
                if bits & (1 << i) and (mt.propertyFlags & properties) == properties:
                    return i
            raise ValueError(f"Can't find memory type for {bits=} {properties}")

        self.device_memory = find_memory_type(
            self.buffer_requirements.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.host_memory = find_memory_type(
            self.buffer_requirements.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def create_allocator(self, memtype: int) -> Allocator:
        props = self.memory_properties.memoryProperties.memoryTypes[memtype]
        heap = self.memory_properties.memoryProperties.memoryHeaps[props.heapIndex]
        return Allocator(
            alignment=self.buffer_requirements.alignment,
            max_memory=heap.size,
            max_contiguous_size=self.physicaldevice_properties.properties.limits.maxStorageBufferRange,
            default_chunk_size=2**29,  # 512 MiB
        )


# class VulkanArray:
#     def __init__(self, vulkan_context: VulkanContext, shape: tuple[int]):
#         self.shaps = shape
#         self.vulkan_context = vulkan_context
#         self.allocation = vulkan_context.allocate

#     def __del__(self):
#         # free or something

# class VulkanContext:
#     def __init__(self):
#         # In order to use Vulkan, you must create an instance
#         self.instance = None
#         self.debugReportCallback = None

#         # The physical device is some device on the system that supports usage of Vulkan.
#         # Often, it is simply a graphics card that supports Vulkan.
#         self.physicalDevice = None

#         # Then we have the logical device VkDevice, which basically allows
#         # us to interact with the physical device.
#         self.device = None

#         # The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.
#         # We will be creating a simple compute pipeline in this application.
#         self.pipeline = None
#         self.pipelineLayout = None
#         self.computeShaderModule = None

#         # The command buffer is used to record commands, that will be submitted to a queue.
#         # To allocate such command buffers, we use a command pool.
#         self.commandPool = None
#         self.commandBuffer = None

#         # Descriptors represent resources in shaders. They allow us to use things like
#         # uniform buffers, storage buffers and images in GLSL.
#         # A single descriptor represents a single resource, and several descriptors are organized
#         # into descriptor sets, which are basically just collections of descriptors.
#         self.descriptorPool = None
#         self.descriptorSet = None
#         self.descriptorSetLayout = None

#         # The mandelbrot set will be rendered to this buffer.
#         # The memory that backs the buffer is bufferMemory.
#         self.buffer = None
#         self.bufferMemory = None

#         # size of `buffer` in bytes.
#         self.bufferSize = 0

#         self.enabledLayers = []

#         # In order to execute commands on a device(GPU), the commands must be submitted
#         # to a queue. The commands are stored in a command buffer, and this command buffer
#         # is given to the queue.
#         # There will be different kinds of queues on the device. Not all queues support
#         # graphics operations, for instance. For this application, we at least want a queue
#         # that supports compute operations.

#         # a queue supporting compute operations.
#         self.queue = None

#         # Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
#         # are grouped into queue families.

#         # When submitting a command buffer, you must specify to which queue in the family you are submitting to.
#         # This variable keeps track of the index of that queue in its family.
#         self.queueFamilyIndex = -1

#         self.pixel = array.array('f', [0, 0, 0, 0])

#         self.saveImageTime = 0
#         self.cpuDataConverTime = 0

#     def del(self):
#         # Clean up all Vulkan Resources.

#         if enableValidationLayers:
#             # destroy callback.
#             func = vkGetInstanceProcAddr(self.instance, 'vkDestroyDebugReportCallbackEXT')
#             if func == ffi.NULL:
#                 raise Exception("Could not load vkDestroyDebugReportCallbackEXT")
#             if self.debugReportCallback:
#                 func(self.instance, self.debugReportCallback, None)

#         if self.bufferMemory:
#             vkFreeMemory(self.device, self.bufferMemory, None)
#         if self.buffer:
#             vkDestroyBuffer(self.device, self.buffer, None)
#         if self.computeShaderModule:
#             vkDestroyShaderModule(self.device, self.computeShaderModule, None)
#         if self.descriptorPool:
#             vkDestroyDescriptorPool(self.device, self.descriptorPool, None)
#         if self.descriptorSetLayout:
#             vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, None)
#         if self.pipelineLayout:
#             vkDestroyPipelineLayout(self.device, self.pipelineLayout, None)
#         if self.pipeline:
#             vkDestroyPipeline(self.device, self.pipeline, None)
#         if self.commandPool:
#             vkDestroyCommandPool(self.device, self.commandPool, None)
#         if self.device:
#             vkDestroyDevice(self.device, None)
#         if self.instance:
#             vkDestroyInstance(self.instance, None)

#     def run(self):
#         # Buffer size of the storage buffer that will contain the rendered mandelbrot set.
#         self.bufferSize = self.pixel.buffer_info()[1] * self.pixel.itemsize * WIDTH * HEIGHT

#         # Initialize vulkan
#         self.createInstance()
#         self.findPhysicalDevice()
#         self.createDevice()
#         self.createBuffer()
#         self.createDescriptorSetLayout()
#         self.createDescriptorSet()
#         self.createComputePipeline()
#         self.createCommandBuffer()

#         # Finally, run the recorded command buffer.
#         self.runCommandBuffer()

#         # The former command rendered a mandelbrot set to a buffer.
#         # Save that buffer as a png on disk.
#         st = time.time()

#         self.saveRenderedImage()

#         self.saveImageTime = time.time() - st

#     def saveRenderedImage(self):
#         # Map the buffer memory, so that we can read from it on the CPU.
#         pmappedMemory = vkMapMemory(self.device, self.bufferMemory, 0, self.bufferSize, 0)

#         # Get the color data from the buffer, and cast it to bytes.
#         # We save the data to a vector.
#         st = time.time()

#         pa = np.frombuffer(pmappedMemory, np.float32)
#         pa = pa.reshape((HEIGHT, WIDTH, 4))
#         pa *= 255

#         self.cpuDataConverTime = time.time() - st

#         # Done reading, so unmap.
#         vkUnmapMemory(self.device, self.bufferMemory)

#         # Now we save the acquired color data to a .png.
#         image = Image.fromarray(pa.astype(np.uint8))
#         image.save('mandelbrot.png')

#     @staticmethod
#     def debugReportCallbackFn(*args):
#         print('Debug Report: {} {}'.format(args[5], args[6]))
#         return 0

#     def createInstance(self):
#         enabledExtensions = []
#         # By enabling validation layers, Vulkan will emit warnings if the API
#         # is used incorrectly. We shall enable the layer VK_LAYER_LUNARG_standard_validation,
#         # which is basically a collection of several useful validation layers.
#         if enableValidationLayers:
#             # We get all supported layers with vkEnumerateInstanceLayerProperties.
#             layerProperties = vkEnumerateInstanceLayerProperties()

#             # And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
#             supportLayerNames = [prop.layerName for prop in layerProperties]
#             if "VK_LAYER_LUNARG_standard_validation" not in supportLayerNames:
#                 raise Exception('Layer VK_LAYER_LUNARG_standard_validation not supported')
#             self.enabledLayers.append("VK_LAYER_LUNARG_standard_validation")

#             # We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
#             # in order to be able to print the warnings emitted by the validation layer.
#             # So again, we just check if the extension is among the supported extensions.
#             extensionProperties = vkEnumerateInstanceExtensionProperties(None)

#             supportExtensions = [prop.extensionName for prop in extensionProperties]
#             if VK_EXT_DEBUG_REPORT_EXTENSION_NAME not in supportExtensions:
#                 raise Exception('Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported')
#             enabledExtensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

#         # Next, we actually create the instance.

#         # Contains application info. This is actually not that important.
#         # The only real important field is apiVersion.
#         applicationInfo = VkApplicationInfo(
#             sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
#             pApplicationName='Hello world app',
#             applicationVersion=0,
#             pEngineName='awesomeengine',
#             engineVersion=0,
#             apiVersion=VK_API_VERSION_1_0
#         )

#         createInfo = VkInstanceCreateInfo(
#             sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#             flags=0,
#             pApplicationInfo=applicationInfo,
#             # Give our desired layers and extensions to vulkan.
#             enabledLayerCount=len(self.enabledLayers),
#             ppEnabledLayerNames=self.enabledLayers,
#             enabledExtensionCount=len(enabledExtensions),
#             ppEnabledExtensionNames=enabledExtensions
#         )

#         # Actually create the instance.
#         # Having created the instance, we can actually start using vulkan.
#         self.instance = vkCreateInstance(createInfo, None)

#         # Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings
#         # emitted from the validation layer are actually printed.
#         if enableValidationLayers:
#             createInfo = VkDebugReportCallbackCreateInfoEXT(
#                 sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
#                 flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
#                 pfnCallback=self.debugReportCallbackFn
#             )

#             # We have to explicitly load this function.
#             vkCreateDebugReportCallbackEXT = vkGetInstanceProcAddr(self.instance, 'vkCreateDebugReportCallbackEXT')
#             if vkCreateDebugReportCallbackEXT == ffi.NULL:
#                 raise Exception('Could not load vkCreateDebugReportCallbackEXT')

#             # Create and register callback.
#             self.debugReportCallback = vkCreateDebugReportCallbackEXT(self.instance, createInfo, None)

#     def findPhysicalDevice(self):
#         # In this function, we find a physical device that can be used with Vulkan.
#         # So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices.
#         devices = vkEnumeratePhysicalDevices(self.instance)

#         # Next, we choose a device that can be used for our purposes.
#         # With VkPhysicalDeviceFeatures(), we can retrieve a fine-grained list of physical features supported by the device.
#         # However, in this demo, we are simply launching a simple compute shader, and there are no
#         # special physical features demanded for this task.
#         # With VkPhysicalDeviceProperties(), we can obtain a list of physical device properties. Most importantly,
#         # we obtain a list of physical device limitations. For this application, we launch a compute shader,
#         # and the maximum size of the workgroups and total number of compute shader invocations is limited by the physical device,
#         # and we should ensure that the limitations named maxComputeWorkGroupCount, maxComputeWorkGroupInvocations and
#         # maxComputeWorkGroupSize are not exceeded by our application.  Moreover, we are using a storage buffer in the compute shader,
#         # and we should ensure that it is not larger than the device can handle, by checking the limitation maxStorageBufferRange.
#         # However, in our application, the workgroup size and total number of shader invocations is relatively small, and the storage buffer is
#         # not that large, and thus a vast majority of devices will be able to handle it. This can be verified by looking at some devices at_
#         # http://vulkan.gpuinfo.org/
#         # Therefore, to keep things simple and clean, we will not perform any such checks here, and just pick the first physical
#         # device in the list. But in a real and serious application, those limitations should certainly be taken into account.

#         # just use the first one
#         self.physicalDevice = devices[0]

#     # Returns the index of a queue family that supports compute operations.
#     def getComputeQueueFamilyIndex(self):
#         # Retrieve all queue families.
#         queueFamilies = vkGetPhysicalDeviceQueueFamilyProperties(self.physicalDevice)

#         # Now find a family that supports compute.
#         for i, props in enumerate(queueFamilies):
#             if props.queueCount > 0 and props.queueFlags & VK_QUEUE_COMPUTE_BIT:
#                 # found a queue with compute. We're done!
#                 return i

#         return -1

#     def createDevice(self):
#         # We create the logical device in this function.

#         self.queueFamilyIndex = self.getComputeQueueFamilyIndex()
#         # When creating the device, we also specify what queues it has.
#         queueCreateInfo = VkDeviceQueueCreateInfo(
#             sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
#             queueFamilyIndex=self.queueFamilyIndex,  # find queue family with compute capability.
#             queueCount=1,  # create one queue in this family. We don't need more.
#             pQueuePriorities=[1.0]  # we only have one queue, so this is not that imporant.
#         )

#         # Now we create the logical device. The logical device allows us to interact with the physical device.
#         # Specify any desired device features here. We do not need any for this application, though.
#         deviceFeatures = VkPhysicalDeviceFeatures()
#         deviceCreateInfo = VkDeviceCreateInfo(
#             sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
#             enabledLayerCount=len(self.enabledLayers),
#             ppEnabledLayerNames=self.enabledLayers,
#             pQueueCreateInfos=queueCreateInfo,
#             queueCreateInfoCount=1,
#             pEnabledFeatures=deviceFeatures
#         )

#         self.device = vkCreateDevice(self.physicalDevice, deviceCreateInfo, None)
#         self.queue = vkGetDeviceQueue(self.device, self.queueFamilyIndex, 0)

#     # find memory type with desired properties.
#     def findMemoryType(self, memoryTypeBits, properties):
#         memoryProperties = vkGetPhysicalDeviceMemoryProperties(self.physicalDevice)

#         # How does this search work?
#         # See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
#         for i, mt in enumerate(memoryProperties.memoryTypes):
#             if memoryTypeBits & (1 << i) and (mt.propertyFlags & properties) == properties:
#                 return i

#         return -1

#     def createBuffer(self):
#         # We will now create a buffer. We will render the mandelbrot set into this buffer
#         # in a computer shade later.
#         bufferCreateInfo = VkBufferCreateInfo(
#             sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
#             size=self.bufferSize,  # buffer size in bytes.
#             usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # buffer is used as a storage buffer.
#             sharingMode=VK_SHARING_MODE_EXCLUSIVE  # buffer is exclusive to a single queue family at a time.
#         )

#         self.buffer = vkCreateBuffer(self.device, bufferCreateInfo, None)

#         # But the buffer doesn't allocate memory for itself, so we must do that manually.

#         # First, we find the memory requirements for the buffer.
#         memoryRequirements = vkGetBufferMemoryRequirements(self.device, self.buffer)

#         # There are several types of memory that can be allocated, and we must choose a memory type that:
#         # 1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
#         # 2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
#         #    with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
#         # Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
#         # visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
#         # this flag.
#         index = self.findMemoryType(memoryRequirements.memoryTypeBits,
#                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
#         # Now use obtained memory requirements info to allocate the memory for the buffer.
#         allocateInfo = VkMemoryAllocateInfo(
#             sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
#             allocationSize=memoryRequirements.size,  # specify required memory.
#             memoryTypeIndex=index
#         )

#         # allocate memory on device.
#         self.bufferMemory = vkAllocateMemory(self.device, allocateInfo, None)

#         # Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
#         vkBindBufferMemory(self.device, self.buffer, self.bufferMemory, 0)

#     def createDescriptorSetLayout(self):
#         # Here we specify a descriptor set layout. This allows us to bind our descriptors to
#         # resources in the shader.

#         # Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
#         # 0. This binds to
#         #   layout(std140, binding = 0) buffer buf
#         # in the compute shader.

#         descriptorSetLayoutBinding = VkDescriptorSetLayoutBinding(
#             binding=0,
#             descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
#             descriptorCount=1,
#             stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
#         )

#         descriptorSetLayoutCreateInfo = VkDescriptorSetLayoutCreateInfo(
#             sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
#             bindingCount=1,  # only a single binding in this descriptor set layout.
#             pBindings=descriptorSetLayoutBinding
#         )

#         # Create the descriptor set layout.
#         self.descriptorSetLayout = vkCreateDescriptorSetLayout(self.device, descriptorSetLayoutCreateInfo, None)

#     def createDescriptorSet(self):
#         # So we will allocate a descriptor set here.
#         # But we need to first create a descriptor pool to do that.

#         # Our descriptor pool can only allocate a single storage buffer.
#         descriptorPoolSize = VkDescriptorPoolSize(
#             type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
#             descriptorCount=1
#         )

#         descriptorPoolCreateInfo = VkDescriptorPoolCreateInfo(
#             sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
#             maxSets=1,  # we only need to allocate one descriptor set from the pool.
#             poolSizeCount=1,
#             pPoolSizes=descriptorPoolSize
#         )

#         # create descriptor pool.
#         self.descriptorPool = vkCreateDescriptorPool(self.device, descriptorPoolCreateInfo, None)

#         # With the pool allocated, we can now allocate the descriptor set.
#         descriptorSetAllocateInfo = VkDescriptorSetAllocateInfo(
#             sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
#             descriptorPool=self.descriptorPool,
#             descriptorSetCount=1,
#             pSetLayouts=[self.descriptorSetLayout]
#         )

#         # allocate descriptor set.
#         self.descriptorSet = vkAllocateDescriptorSets(self.device, descriptorSetAllocateInfo)[0]

#         # Next, we need to connect our actual storage buffer with the descrptor.
#         # We use vkUpdateDescriptorSets() to update the descriptor set.

#         # Specify the buffer to bind to the descriptor.
#         descriptorBufferInfo = VkDescriptorBufferInfo(
#             buffer=self.buffer,
#             offset=0,
#             range=self.bufferSize
#         )

#         writeDescriptorSet = VkWriteDescriptorSet(
#             sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
#             dstSet=self.descriptorSet,
#             dstBinding=0,  # write to the first, and only binding.
#             descriptorCount=1,
#             descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
#             pBufferInfo=descriptorBufferInfo
#         )

#         # perform the update of the descriptor set.
#         vkUpdateDescriptorSets(self.device, 1, [writeDescriptorSet], 0, None)

#     def createComputePipeline(self):
#         # We create a compute pipeline here.

#         # Create a shader module. A shader module basically just encapsulates some shader code.
#         with open('mandelbrot_compute.spv', 'rb') as comp:
#             code = comp.read()

#             createInfo = VkShaderModuleCreateInfo(
#                 sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
#                 codeSize=len(code),
#                 pCode=code
#             )

#             self.computeShaderModule = vkCreateShaderModule(self.device, createInfo, None)

#         # Now let us actually create the compute pipeline.
#         # A compute pipeline is very simple compared to a graphics pipeline.
#         # It only consists of a single stage with a compute shader.
#         # So first we specify the compute shader stage, and it's entry point(main).
#         shaderStageCreateInfo = VkPipelineShaderStageCreateInfo(
#             sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
#             stage=VK_SHADER_STAGE_COMPUTE_BIT,
#             module=self.computeShaderModule,
#             pName='main'
#         )

#         # The pipeline layout allows the pipeline to access descriptor sets.
#         # So we just specify the descriptor set layout we created earlier.
#         pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
#             sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
#             setLayoutCount=1,
#             pSetLayouts=[self.descriptorSetLayout]
#         )
#         self.pipelineLayout = vkCreatePipelineLayout(self.device, pipelineLayoutCreateInfo, None)

#         pipelineCreateInfo = VkComputePipelineCreateInfo(
#             sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
#             stage=shaderStageCreateInfo,
#             layout=self.pipelineLayout
#         )

#         # Now, we finally create the compute pipeline.
#         pipelines = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, pipelineCreateInfo, None)
#         if len(pipelines) == 1:
#             self.pipeline = pipelines[0]
#         else:
#             raise Exception("Could not create compute pipeline")

#     def createCommandBuffer(self):
#         # We are getting closer to the end. In order to send commands to the device(GPU),
#         # we must first record commands into a command buffer.
#         # To allocate a command buffer, we must first create a command pool. So let us do that.
#         commandPoolCreateInfo = VkCommandPoolCreateInfo(
#             sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
#             flags=0,
#             # the queue family of this command pool. All command buffers allocated from this command pool,
#             # must be submitted to queues of this family ONLY.
#             queueFamilyIndex=self.queueFamilyIndex
#         )

#         self.commandPool = vkCreateCommandPool(self.device, commandPoolCreateInfo, None)

#         # Now allocate a command buffer from the command pool.
#         commandBufferAllocateInfo = VkCommandBufferAllocateInfo(
#             sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
#             commandPool=self.commandPool,
#             # if the command buffer is primary, it can be directly submitted to queues.
#             # A secondary buffer has to be called from some primary command buffer, and cannot be directly
#             # submitted to a queue. To keep things simple, we use a primary command buffer.
#             level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
#             commandBufferCount=1
#         )

#         self.commandBuffer = vkAllocateCommandBuffers(self.device, commandBufferAllocateInfo)[0]

#         # Now we shall start recording commands into the newly allocated command buffer.
#         beginInfo = VkCommandBufferBeginInfo(
#             sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
#             # the buffer is only submitted and used once in this application.
#             flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
#         )
#         vkBeginCommandBuffer(self.commandBuffer, beginInfo)

#         # We need to bind a pipeline, AND a descriptor set before we dispatch.
#         # The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
#         vkCmdBindPipeline(self.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
#         vkCmdBindDescriptorSets(self.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelineLayout,
#                                 0, 1, [self.descriptorSet], 0, None)

#         # Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
#         # The number of workgroups is specified in the arguments.
#         # If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
#         vkCmdDispatch(self.commandBuffer,
#                       int(math.ceil(WIDTH / float(WORKGROUP_SIZE))),  # int for py2 compatible
#                       int(math.ceil(HEIGHT / float(WORKGROUP_SIZE))),  # int for py2 compatible
#                       1)

#         vkEndCommandBuffer(self.commandBuffer)

#     def runCommandBuffer(self):
#         # Now we shall finally submit the recorded command buffer to a queue.
#         submitInfo = VkSubmitInfo(
#             sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
#             commandBufferCount=1,  # submit a single command buffer
#             pCommandBuffers=[self.commandBuffer]  # the command buffer to submit.
#         )

#         # We create a fence.
#         fenceCreateInfo = VkFenceCreateInfo(
#             sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
#             flags=0
#         )
#         fence = vkCreateFence(self.device, fenceCreateInfo, None)

#         # We submit the command buffer on the queue, at the same time giving a fence.
#         vkQueueSubmit(self.queue, 1, submitInfo, fence)

#         # The command will not have finished executing until the fence is signalled.
#         # So we wait here.
#         # We will directly after this read our buffer from the GPU,
#         # and we will not be sure that the command has finished executing unless we wait for the fence.
#         # Hence, we use a fence here.
#         vkWaitForFences(self.device, 1, [fence], VK_TRUE, 100000000000)

#         vkDestroyFence(self.device, fence, None)

# if name == 'main':
#     startTime = time.time()

#     app = ComputeApplication()
#     app.run()

#     endTime = time.time()
#     if enableValidationLayers:
#         print('raw image data (CPU) convert time: {} seconds'.format(app.cpuDataConverTime))
#         print('Vulkan setup and compute time: {} seconds'.format(endTime-startTime-app.saveImageTime))
#         print('save image time: {} seconds'.format(app.saveImageTime))
#         print('total time used: {} seconds'.format(endTime-startTime))

#     del app
