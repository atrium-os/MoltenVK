// Minimal VK_KHR_ray_query end-to-end validation host for MoltenVK.
// Builds a 1-triangle BLAS + 1-instance TLAS, dispatches ray_query.comp.spv,
// reads back the SSBO. Expect hit==1, primitiveIndex==0, t~1.0.
//
// Build:  cc host.c -o rqtest -I <Vulkan-Headers>/include -L/opt/homebrew/lib -lvulkan
// Run:    VK_DRIVER_FILES=<our MoltenVK_icd.json> ./rqtest ray_query.comp.spv
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "VK_CHECK failed %d at %s:%d\n", _r, __FILE__, __LINE__); exit(2);} } while(0)

static VkInstance       inst;
static VkPhysicalDevice phys;
static VkDevice         dev;
static VkQueue          queue;
static uint32_t         qfam;

// device-level KHR fps
static PFN_vkGetAccelerationStructureBuildSizesKHR    pGetBuildSizes;
static PFN_vkCreateAccelerationStructureKHR           pCreateAS;
static PFN_vkCmdBuildAccelerationStructuresKHR        pCmdBuildAS;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pGetASAddr;
static PFN_vkGetBufferDeviceAddressKHR                pGetBufAddr;

static uint32_t findMem(uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type\n"); exit(2);
}

// host-visible|coherent buffer; adds device-address alloc flag when requested.
// device-local & NOT host-visible -> Metal MTLStorageModePrivate (required for
// acceleration-structure backing heaps).
static uint32_t findMemPrivate(uint32_t typeBits) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((typeBits & (1u<<i)) && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return i;
    }
    fprintf(stderr, "no private (device-local) memory type\n"); exit(2);
}
// `priv` -> device-local Private memory (AS storage must be Private); else
// host-visible|coherent (mappable, for vertices/instances/readback).
static void mkBuffer(VkDeviceSize sz, VkBufferUsageFlags usage, int deviceAddr, int priv,
                     VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = sz; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bci, NULL, buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, *buf, &req);
    VkMemoryAllocateFlagsInfo fi = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = deviceAddr ? &fi : NULL;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = priv ? findMemPrivate(req.memoryTypeBits)
        : findMem(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(dev, &ai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
}
static VkDeviceAddress bufAddr(VkBuffer b) {
    VkBufferDeviceAddressInfo i = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; i.buffer = b;
    return pGetBufAddr(dev, &i);
}
static void *mapAll(VkDeviceMemory m, VkDeviceSize sz) { void *p; VK_CHECK(vkMapMemory(dev,m,0,sz,0,&p)); return p; }

static VkCommandPool cpool;
static VkCommandBuffer beginCmd() {
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = cpool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi)); return cb;
}
static void endCmd(VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, cpool, 1, &cb);
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "ray_query.comp.spv";

    // ---- instance ---- (linked directly against MoltenVK, no loader, so no
    // loader-side extension filtering and no portability-enumeration dance.)
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    const char *iext[] = { "VK_KHR_portability_enumeration" };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = iext;
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t n = 0; VK_CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    if (!n) { fprintf(stderr, "no physical devices (MoltenVK not found via VK_DRIVER_FILES?)\n"); return 2; }
    VkPhysicalDevice devs[8]; if (n>8) n=8; VK_CHECK(vkEnumeratePhysicalDevices(inst, &n, devs)); phys = devs[0];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys, &pp);
    printf("device: %s\n", pp.deviceName);

    // ---- queue family (compute) ----
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, NULL);
    VkQueueFamilyProperties qfp[16]; if (qn>16) qn=16; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qfp);
    qfam = ~0u; for (uint32_t i=0;i<qn;i++) if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam=i; break; }
    if (qfam==~0u){fprintf(stderr,"no compute queue\n");return 2;}

    // ---- device + RT features ----
    // Enable only device extensions actually present (buffer_device_address /
    // descriptor_indexing / spirv_1_4 are core 1.2, not separate exts here).
    uint32_t aec = 0; vkEnumerateDeviceExtensionProperties(phys, NULL, &aec, NULL);
    VkExtensionProperties *aep = malloc(aec * sizeof(*aep));
    vkEnumerateDeviceExtensionProperties(phys, NULL, &aec, aep);
    const char *want[] = { "VK_KHR_portability_subset", "VK_KHR_acceleration_structure",
                           "VK_KHR_ray_query", "VK_KHR_deferred_host_operations" };
    const char *dexts[8]; uint32_t dn = 0; (void)aec; (void)aep;
    // Request unconditionally: through the Vulkan loader, AS/ray_query are
    // OMITTED from device-extension ENUMERATION (a cosmetic loader filter),
    // but MoltenVK still accepts them at vkCreateDevice and exposes working
    // proc addrs. So we do NOT gate on enumeration presence.
    for (int i = 0; i < 4; i++) dexts[dn++] = want[i];
    VkPhysicalDeviceRayQueryFeaturesKHR rqf = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
    rqf.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
    asf.accelerationStructure = VK_TRUE; asf.pNext = &rqf;
    VkPhysicalDeviceVulkan12Features bda = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    bda.bufferDeviceAddress = VK_TRUE; bda.descriptorIndexing = VK_TRUE; bda.pNext = &asf;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &bda; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = dn; dci.ppEnabledExtensionNames = dexts;
    VK_CHECK(vkCreateDevice(phys, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    #define LOAD(var, fn) var = (PFN_vk##fn)vkGetDeviceProcAddr(dev, "vk" #fn); if(!var){fprintf(stderr,"missing vk" #fn "\n");return 2;}
    LOAD(pGetBuildSizes, GetAccelerationStructureBuildSizesKHR);
    LOAD(pCreateAS, CreateAccelerationStructureKHR);
    LOAD(pCmdBuildAS, CmdBuildAccelerationStructuresKHR);
    LOAD(pGetASAddr, GetAccelerationStructureDeviceAddressKHR);
    LOAD(pGetBufAddr, GetBufferDeviceAddress);  // core 1.2 (KHR alias absent w/o the ext)
    #undef LOAD

    VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = qfam; pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(dev, &pci, NULL, &cpool));

    // scratch alignment
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asp = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
    VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 }; p2.pNext = &asp;
    vkGetPhysicalDeviceProperties2(phys, &p2);
    VkDeviceSize scratchAlign = asp.minAccelerationStructureScratchOffsetAlignment ? asp.minAccelerationStructureScratchOffsetAlignment : 256;

    const VkBufferUsageFlags ASIN = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // ---- vertices (one triangle in z=0) ----
    float verts[9] = { -0.5f,-0.5f,0.0f,  0.5f,-0.5f,0.0f,  0.0f,0.5f,0.0f };
    VkBuffer vbuf; VkDeviceMemory vmem; mkBuffer(sizeof(verts), ASIN, 1, 0, &vbuf, &vmem);
    memcpy(mapAll(vmem, sizeof(verts)), verts, sizeof(verts)); vkUnmapMemory(dev, vmem);

    // ---- BLAS ----
    VkAccelerationStructureGeometryKHR bgeo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    bgeo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; bgeo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    bgeo.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    bgeo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    bgeo.geometry.triangles.vertexData.deviceAddress = bufAddr(vbuf);
    bgeo.geometry.triangles.vertexStride = 12; bgeo.geometry.triangles.maxVertex = 2;
    bgeo.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR bbi = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    bbi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bbi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bbi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bbi.geometryCount = 1; bbi.pGeometries = &bgeo;
    uint32_t one = 1;
    VkAccelerationStructureBuildSizesInfoKHR bsz = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    pGetBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bbi, &one, &bsz);

    VkBuffer blasBuf; VkDeviceMemory blasMem;
    mkBuffer(bsz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 1, 1, &blasBuf, &blasMem);
    VkBuffer bscratch; VkDeviceMemory bscratchMem;
    mkBuffer(bsz.buildScratchSize + scratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 1, 0, &bscratch, &bscratchMem);
    VkAccelerationStructureCreateInfoKHR bci = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    bci.buffer = blasBuf; bci.size = bsz.accelerationStructureSize; bci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkAccelerationStructureKHR blas; VK_CHECK(pCreateAS(dev, &bci, NULL, &blas));
    bbi.dstAccelerationStructure = blas;
    VkDeviceAddress sa = bufAddr(bscratch); sa = (sa + scratchAlign - 1) & ~(scratchAlign - 1);
    bbi.scratchData.deviceAddress = sa;
    VkAccelerationStructureBuildRangeInfoKHR brange = { .primitiveCount = 1 };
    const VkAccelerationStructureBuildRangeInfoKHR *pbr = &brange;

    // ---- instance + TLAS ----
    VkAccelerationStructureDeviceAddressInfoKHR dai = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
    dai.accelerationStructure = blas;
    VkAccelerationStructureInstanceKHR instData; memset(&instData, 0, sizeof(instData));
    instData.transform.matrix[0][0] = instData.transform.matrix[1][1] = instData.transform.matrix[2][2] = 1.0f;
    instData.mask = 0xFF; instData.instanceCustomIndex = 0;
    instData.accelerationStructureReference = pGetASAddr(dev, &dai);
    VkBuffer ibuf; VkDeviceMemory imem; mkBuffer(sizeof(instData), ASIN, 1, 0, &ibuf, &imem);
    memcpy(mapAll(imem, sizeof(instData)), &instData, sizeof(instData)); vkUnmapMemory(dev, imem);

    VkAccelerationStructureGeometryKHR tgeo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
    tgeo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR; tgeo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tgeo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tgeo.geometry.instances.data.deviceAddress = bufAddr(ibuf);
    VkAccelerationStructureBuildGeometryInfoKHR tbi = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
    tbi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tbi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tbi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tbi.geometryCount = 1; tbi.pGeometries = &tgeo;
    VkAccelerationStructureBuildSizesInfoKHR tsz = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
    pGetBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tbi, &one, &tsz);
    VkBuffer tlasBuf; VkDeviceMemory tlasMem;
    mkBuffer(tsz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 1, 1, &tlasBuf, &tlasMem);
    VkBuffer tscratch; VkDeviceMemory tscratchMem;
    mkBuffer(tsz.buildScratchSize + scratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 1, 0, &tscratch, &tscratchMem);
    VkAccelerationStructureCreateInfoKHR tci = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
    tci.buffer = tlasBuf; tci.size = tsz.accelerationStructureSize; tci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VkAccelerationStructureKHR tlas; VK_CHECK(pCreateAS(dev, &tci, NULL, &tlas));
    tbi.dstAccelerationStructure = tlas;
    VkDeviceAddress ta = bufAddr(tscratch); ta = (ta + scratchAlign - 1) & ~(scratchAlign - 1);
    tbi.scratchData.deviceAddress = ta;
    VkAccelerationStructureBuildRangeInfoKHR trange = { .primitiveCount = 1 };
    const VkAccelerationStructureBuildRangeInfoKHR *ptr = &trange;

    // ---- record both builds (BLAS, barrier, TLAS) ----
    VkCommandBuffer cb = beginCmd();
    pCmdBuildAS(cb, 1, &bbi, &pbr);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &mb, 0, NULL, 0, NULL);
    pCmdBuildAS(cb, 1, &tbi, &ptr);
    endCmd(cb);

    // ---- result SSBO ----
    VkBuffer rbuf; VkDeviceMemory rmem; mkBuffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, 0, &rbuf, &rmem);
    memset(mapAll(rmem, 16), 0, 16); vkUnmapMemory(dev, rmem);

    // ---- descriptor set: binding0 AS, binding1 SSBO ----
    VkDescriptorSetLayoutBinding lb[2] = {
        { 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 2; lci.pBindings = lb;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, NULL, &dsl));
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 } };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
    VkDescriptorPool dpool; VK_CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset; VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

    VkWriteDescriptorSetAccelerationStructureKHR was = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    was.accelerationStructureCount = 1; was.pAccelerationStructures = &tlas;
    VkDescriptorBufferInfo rbi = { rbuf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &was, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, NULL, NULL, NULL },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, dset, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &rbi, NULL },
    };
    vkUpdateDescriptorSets(dev, 2, w, 0, NULL);

    // ---- compute pipeline from spv ----
    FILE *f = fopen(spv, "rb"); if(!f){fprintf(stderr,"cannot open %s\n",spv);return 2;}
    fseek(f,0,SEEK_END); long sz = ftell(f); fseek(f,0,SEEK_SET);
    uint32_t *code = malloc(sz); fread(code,1,sz,f); fclose(f);
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO }; smci.codeSize = sz; smci.pCode = code;
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO }; plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout playout; VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &playout));
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = playout;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    // ---- dispatch ----
    VkCommandBuffer cb2 = beginCmd();
    vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, NULL);
    vkCmdDispatch(cb2, 1, 1, 1);
    endCmd(cb2);

    // ---- read back ----
    struct { uint32_t hit; float t; uint32_t prim; uint32_t inst; } *res = mapAll(rmem, 16);
    printf("RESULT: hit=%u t=%.4f primitiveIndex=%u instanceId=%u\n", res->hit, res->t, res->prim, res->inst);
    int ok = (res->hit == 1u && res->prim == 0u && res->t > 0.5f && res->t < 1.5f);
    printf(ok ? "==> VK_KHR_ray_query WORKS on MoltenVK \xe2\x9c\x93\n" : "==> FAIL (no committed hit)\n");
    return ok ? 0 : 1;
}
