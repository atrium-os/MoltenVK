/*
 * test_mvk_import.m - reproduce venus MoltenVK bug standalone.
 *
 * 1. Create MTLBuffer via newBufferWithLength: with StorageModeShared.
 * 2. Create VkDeviceMemory importing that MTLBuffer via
 *    VkImportMemoryMetalHandleInfoEXT.
 * 3. Bind a VkBuffer to the imported memory.
 * 4. Run vkCmdFillBuffer(0xCAFEBABE) and check.
 *
 * This matches the path vkr_metal_helpers.m + vkr_device_memory.c
 * take in the venus host worker. If THIS fails (and the non-import
 * path passes), the bug is reproduced standalone — perfect material
 * for a MoltenVK PR.
 *
 * Build: clang -fobjc-arc -framework Metal -framework Foundation \
 *        -I/opt/homebrew/include -L/opt/homebrew/lib -lvulkan \
 *        -o /tmp/test_mvk_import /tmp/test_mvk_import.m
 */

#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

#define CHECK(expr) do { VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #expr, _r); return 1; } \
} while (0)

#define N 1024
#define SIZE (N * sizeof(uint32_t))

int main(void) {
    /* 1. Allocate Metal buffer ourselves */
    id<MTLDevice> mtl_dev = MTLCreateSystemDefaultDevice();
    if (!mtl_dev) { fprintf(stderr, "no MTLDevice\n"); return 1; }
    id<MTLBuffer> mtl_buf = [mtl_dev newBufferWithLength:SIZE
                                                 options:MTLResourceStorageModeShared];
    if (!mtl_buf) { fprintf(stderr, "no MTLBuffer\n"); return 1; }
    uint32_t *cpu = (uint32_t *)[mtl_buf contents];
    for (int i = 0; i < N; i++) cpu[i] = 0xAAAAAAAAu;

    /* 2. Vulkan instance with portability + import_metal */
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2,
    };
    const char *ext_inst[] = { "VK_KHR_portability_enumeration",
                               "VK_KHR_get_physical_device_properties2",
                               "VK_KHR_external_memory_capabilities" };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo = &app,
        .enabledExtensionCount = 3, .ppEnabledExtensionNames = ext_inst,
    };
    VkInstance instance;
    CHECK(vkCreateInstance(&ici, NULL, &instance));

    uint32_t pd_count = 1;
    VkPhysicalDevice pd;
    CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, &pd));

    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &pri,
    };
    const char *ext_dev[] = { "VK_KHR_portability_subset",
                              "VK_EXT_external_memory_metal" };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 2, .ppEnabledExtensionNames = ext_dev,
    };
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    /* 3. Create VkBuffer of the right size */
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = SIZE,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer vk_buf;
    CHECK(vkCreateBuffer(dev, &bci, NULL, &vk_buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, vk_buf, &mr);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            mt = i; break;
        }
    }
    if (mt == ~0u) { fprintf(stderr, "no host-visible mt\n"); return 1; }

    /* 4. Allocate VkDeviceMemory IMPORTING our MTLBuffer */
    VkImportMemoryMetalHandleInfoEXT import = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_METAL_HANDLE_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT,
        .handle = (__bridge void *)mtl_buf,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import,
        .allocationSize = mr.size, .memoryTypeIndex = mt,
    };
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
    CHECK(vkBindBufferMemory(dev, vk_buf, mem, 0));

    /* 5. Submit fillBuffer */
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0,
    };
    VkCommandPool cp;
    CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK(vkBeginCommandBuffer(cb, &cbbi));
    vkCmdFillBuffer(cb, vk_buf, 0, VK_WHOLE_SIZE, 0xCAFEBABEu);
    VkBufferMemoryBarrier bmb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = vk_buf, .offset = 0, .size = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, NULL, 1, &bmb, 0, NULL);
    CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cb,
    };
    fprintf(stderr, "submitting fillBuffer(0xCAFEBABE) on imported MTLBuffer\n");
    CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));
    fprintf(stderr, "queue idle\n");

    /* 6. Read back via the original MTLBuffer's [contents] */
    int ok_count = 0;
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "buf[%d] = 0x%08x\n", i, cpu[i]);
        if (cpu[i] == 0xCAFEBABEu) ok_count++;
    }
    if (cpu[N-1] == 0xCAFEBABEu) ok_count++;
    fprintf(stderr, "%s: %d/9 slots show 0xCAFEBABE (via [mtl_buf contents])\n",
            ok_count == 9 ? "PASS" : "FAIL", ok_count);

    return ok_count == 9 ? 0 : 1;
}
