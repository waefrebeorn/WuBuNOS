/*
 * vk_probe.c -- enumerate EVERY Vulkan physical device reachable through
 * /dev/dxg and print hardware-aware codegen profiles, one block per device.
 *
 * This is the borg door: same binary reaches RTX 4050 (NVIDIA layers via
 * D3D12), Radeon 740M (Mesa RADV via D3D12), and llvmpipe (CPU fallback).
 * The compiler parses this output to size codegen per-device — no
 * hardcoded limits anywhere.
 *
 * C11, libvulkan only.
 *   cc -O2 -std=c11 -o vk_probe vk_probe.c -lvulkan
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "wubu-vk-probe";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance inst;
    VkResult rc = vkCreateInstance(&ici, NULL, &inst);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", rc);
        return 1;
    }

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, NULL);
    if (ndev == 0) { printf("devices=0\n"); return 0; }
    printf("devices=%u\n", ndev);

    VkPhysicalDevice devs[16];
    vkEnumeratePhysicalDevices(inst, &ndev, devs);

    for (uint32_t i = 0; i < ndev && i < 16; i++) {
        VkPhysicalDeviceProperties props;
        VkPhysicalDeviceLimits lim;
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        VkPhysicalDeviceFeatures feat;
        vkGetPhysicalDeviceFeatures(devs[i], &feat);
        lim = props.limits;
        vkGetPhysicalDeviceMemoryProperties(devs[i], &mem);

        uint64_t vram = 0, sysmem = 0;
        for (uint32_t h = 0; h < mem.memoryHeapCount; h++) {
            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram += mem.memoryHeaps[h].size;
            else
                sysmem += mem.memoryHeaps[h].size;
        }

        int ok_compute = feat.shaderInt64;
        printf("[dev%u]\n", i);
        printf("name=%s\n", props.deviceName);
        printf("vendor=0x%x\n", props.vendorID);
        printf("device_id=0x%x\n", props.deviceID);
        printf("driver=(see vendor)\n");
        printf("api=%u.%u.%u\n",
               VK_API_VERSION_MAJOR(props.apiVersion),
               VK_API_VERSION_MINOR(props.apiVersion),
               VK_API_VERSION_PATCH(props.apiVersion));
        printf("type=%d\n", (int)props.deviceType); /* 0=other 1=integrated 2=discrete 3=cpu */
        printf("compute_queues=%u\n", props.limits.maxComputeWorkGroupCount[0]);
        printf("max_workgroup_invocations=%u\n", lim.maxComputeWorkGroupInvocations);
        printf("max_workgroup_size=%u %u %u\n",
               lim.maxComputeWorkGroupSize[0], lim.maxComputeWorkGroupSize[1],
               lim.maxComputeWorkGroupSize[2]);
        printf("max_dispatch=%u %u %u\n",
               lim.maxComputeWorkGroupCount[0], lim.maxComputeWorkGroupCount[1],
               lim.maxComputeWorkGroupCount[2]);
        printf("shared_mem_per_block=%u\n", lim.maxComputeSharedMemorySize);
        printf("push_constants=%u\n", lim.maxPushConstantsSize);
        printf("uniform_buffer_max=%llu\n", (unsigned long long)lim.maxUniformBufferRange);
        printf("storage_buffer_max=%llu\n", (unsigned long long)lim.maxStorageBufferRange);
        printf("timestamp_period_ns=%.1f\n", (double)lim.timestampPeriod);
        printf("shader_int64=%d\n", (int)feat.shaderInt64);
        printf("shader_float64=%d\n", (int)feat.shaderFloat64);
        printf("vram_bytes=%llu\n", (unsigned long long)vram);
        printf("sysmem_bytes=%llu\n", (unsigned long long)sysmem);
        printf("compute_capable=%d\n", ok_compute ? 1 : 1 /* int64 optional but emulate; still compute */);
        printf("\n");
    }

    vkDestroyInstance(inst, NULL);
    return 0;
}
