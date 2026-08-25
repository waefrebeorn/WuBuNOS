/*
 * vk_run.c -- the Vulkan execution leg (companion to wubu_isa_spirv.c).
 *
 * Loads a SPIR-V compute module, runs it on a chosen physical device,
 * reads back the SSBO result cell.
 *
 * Usage: vk_run <device_index> <shader.spv> <arg> <mem_cells>
 * Prints result cell (mem[0]) as decimal int64.
 *
 * C11, libvulkan only. Runs on ANY Vulkan device — NVIDIA dGPU, AMD APU
 * iGPU, old recycled cards, llvmpipe CPU fallback.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
        const char *_es = "unknown"; \
        switch (_r) { \
        case -1: _es="OUT_OF_HOST_MEMORY"; break; \
        case -2: _es="OUT_OF_DEVICE_MEMORY"; break; \
        case -4: _es="INITIALIZATION_FAILED"; break; \
        case -5: _es="DEVICE_LOST"; break; \
        case -6: _es="MEMORY_MAP_FAILED"; break; \
        case -9:  _es="OUT_OF_POOL_MEMORY"; break; \
        case -1000000004: _es="INVALID_EXTERNAL_HANDLE"; break; \
        } \
        fprintf(stderr, "vk error %d (%s) at line %d\n", _r, _es, __LINE__); \
        return 2; } } while (0)

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s <dev_idx> <shader.spv> <arg> <mem_cells>\n",
                argv[0]);
        return 1;
    }
    uint32_t dev_idx = (uint32_t)atoi(argv[1]);
    const char *spv_path = argv[2];
    int64_t arg = strtoll(argv[3], NULL, 10);
    uint32_t mem_cells = (uint32_t)atoi(argv[4]);

    /* read SPIR-V */
    FILE *f = fopen(spv_path, "rb");
    if (!f) { perror("spv"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)sz);
    fread(code, 1, (size_t)sz, f);
    fclose(f);

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    if (dev_idx >= nd) { fprintf(stderr, "no dev %u\n", dev_idx); return 1; }
    VkPhysicalDevice phys[8];
    vkEnumeratePhysicalDevices(inst, &nd, phys);
    VkPhysicalDevice pd = phys[dev_idx];

    /* find a compute queue */
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
    VkQueueFamilyProperties qf[8];
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qf);
    int qfam = -1;
    for (uint32_t i = 0; i < nqf && i < 8; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = (int)i; break; }
    if (qfam < 0) { fprintf(stderr, "no compute queue\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = (uint32_t)qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

    VkQueue queue;
    vkGetDeviceQueue(dev, (uint32_t)qfam, 0, &queue);

    /* buffers: SSBO layout = u64 mem[mem_cells] where cell0 = result */
    VkBuffer buf;
    VkMemoryRequirements mr;
    VkDeviceSize bufsz = (VkDeviceSize)mem_cells * 8;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bufsz;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
    vkGetBufferMemoryRequirements(dev, buf, &mr);

    /* find HOST_VISIBLE | HOST_COHERENT memory */
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    int memi = -1;
    /* prefer DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT (ReBAR style) then any
     * HOST_VISIBLE|HOST_COHERENT */
    for (uint32_t pass = 0; pass < 2 && memi < 0; pass++) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            VkMemoryPropertyFlags want = pass == 0
                ? (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & want) == want)
            { memi = (int)i; break; }
        }
    }
    if (memi < 0) { fprintf(stderr, "no host-visible mem\n"); return 1; }

    VkDeviceMemory memh;
    VkMemoryDedicatedAllocateInfo dai = {0};
    dai.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dai.buffer = buf;
    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &dai;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = (uint32_t)memi;
    CHECK(vkAllocateMemory(dev, &mai, NULL, &memh));
    CHECK(vkBindBufferMemory(dev, &buf, memh, 0));

    /* map + init: cell0=0 result, cell1=arg */
    int64_t *host;
    CHECK(vkMapMemory(dev, memh, 0, bufsz, 0, (void **)&host));
    memset(host, 0, (size_t)bufsz);
    host[0] = 0;
    host[1] = arg;

    /* descriptors: binding 0 = SSBO */
    VkDescriptorSetLayoutBinding bind = {0};
    bind.binding = 0;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo slci = {0};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 1;
    slci.pBindings = &bind;
    VkDescriptorSetLayout setlayout;
    CHECK(vkCreateDescriptorSetLayout(dev, &slci, NULL, &setlayout));

    /* push constant range: one i64 arg */
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 8;
    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setlayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

    VkShaderModuleCreateInfo smci = {0};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = (size_t)sz;
    smci.pCode = code;
    VkShaderModule shmod;
    CHECK(vkCreateShaderModule(dev, &smci, NULL, &shmod));

    VkPipelineShaderStageCreateInfo stage = {0};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shmod;
    stage.pName = "main";
    VkComputePipelineCreateInfo pci = {0};
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage = stage;
    pci.layout = layout;
    VkPipeline pipe;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL, &pipe));

    /* descriptor pool + set */
    VkDescriptorPoolSize ps = {0};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci = {0};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &pool));
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &setlayout;
    VkDescriptorSet dset;
    CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
    VkDescriptorBufferInfo dbi = {0};
    dbi.buffer = buf;
    dbi.offset = 0;
    dbi.range = bufsz;
    VkWriteDescriptorSet wds = {0};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = dset;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

    /* command buffer */
    VkCommandPoolCreateInfo cpi = {0};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = (uint32_t)qfam;
    VkCommandPool cpool;
    CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
    VkCommandBufferBeginInfo bbi = {0};
    bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cmd, &bbi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout, 0, 1, &dset, 0, NULL);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &arg);
    vkCmdDispatch(cmd, 1, 1, 1);
    CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));
    {
        VkMappedMemoryRange rng = {0};
        rng.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        rng.memory = memh;
        rng.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(dev, 1, &rng);
    }

    printf("%lld\n", (long long)host[0]);
#ifdef DEBUG_CELLS
    for (int i = 0; i < 4 && (size_t)i < bufsz/8; i++)
        fprintf(stderr, "cell[%d] = %lld\n", i, (long long)host[i]);
#endif

    /* cleanup */
    vkUnmapMemory(dev, memh);
    vkFreeMemory(dev, memh, NULL);
    vkDestroyBuffer(dev, buf, NULL);
    vkFreeDescriptorSets(dev, pool, 1, &dset);
    vkDestroyDescriptorPool(dev, pool, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, shmod, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyDescriptorSetLayout(dev, setlayout, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    free(code);
    return 0;
}
