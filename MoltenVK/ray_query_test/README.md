# VK_KHR_ray_query validation

A minimal end-to-end check that a Vulkan compute shader using `ray_query`
(`rayQueryEXT` / SPIR-V `OpRayQuery*`) intersects a built acceleration
structure on Apple GPUs through MoltenVK.

`ray_query.comp` is the GLSL compute shader. It fires one ray at a one-triangle
TLAS and writes the committed-hit result to an SSBO.

## 1. Compile the shader to SPIR-V

```sh
glslangValidator --target-env vulkan1.2 ray_query.comp -o ray_query.comp.spv
# or:
glslc --target-env=vulkan1.2 -fshader-stage=comp ray_query.comp -o ray_query.comp.spv
```

(`VK_KHR_ray_query` needs SPIR-V 1.4+, hence `vulkan1.2`.)

## 2. Host program outline

Use a tiny Vulkan host (vkcube-style, or a scratch `main.cpp`). Key steps:

1. **Instance + physical device.** Pick the MoltenVK device.

2. **Enable extensions + features** at device creation, chained into
   `VkDeviceCreateInfo.pNext`:
   - Extensions: `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`,
     `VK_KHR_buffer_device_address`, `VK_KHR_deferred_host_operations`,
     `VK_EXT_descriptor_indices`/`VK_KHR_spirv_1_4` as required by the loader.
   - `VkPhysicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE`
   - `VkPhysicalDeviceAccelerationStructureFeaturesKHR.accelerationStructure = VK_TRUE`
   - `VkPhysicalDeviceRayQueryFeaturesKHR.rayQuery = VK_TRUE`

3. **Geometry buffers** (all created with
   `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` and
   `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR`,
   allocated from memory with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`):
   - Vertices: a single triangle around the origin in the z=0 plane, e.g.
     `(-0.5,-0.5,0), (0.5,-0.5,0), (0.0,0.5,0)`.
   - (Optional) index buffer `{0,1,2}`.

4. **Build the BLAS.**
   - `VkAccelerationStructureGeometryKHR` of type `..._TRIANGLES_KHR`, with
     `vertexData.deviceAddress` = vertex buffer device address, `vertexStride`
     = 12, `maxVertex` = 2, `indexType` = `VK_INDEX_TYPE_NONE_KHR` (or `_UINT32`
     with the index buffer).
   - `vkGetAccelerationStructureBuildSizesKHR` → allocate AS-backing buffer
     (usage `..._ACCELERATION_STRUCTURE_STORAGE_BIT_KHR`) and scratch buffer.
   - `vkCreateAccelerationStructureKHR` (type `..._BOTTOM_LEVEL_KHR`).
   - Record `vkCmdBuildAccelerationStructuresKHR` with primitiveCount = 1.

5. **Build the TLAS.**
   - One `VkAccelerationStructureInstanceKHR`: identity transform,
     `instanceCustomIndex` = e.g. 0, `mask` = 0xFF,
     `accelerationStructureReference` =
     `vkGetAccelerationStructureDeviceAddressKHR(BLAS)`. Upload to an instance
     buffer (device-address-enabled).
   - Geometry of type `..._INSTANCES_KHR`, `data.deviceAddress` = instance
     buffer address. Build as `..._TOP_LEVEL_KHR`, primitiveCount = 1.
   - Insert a memory barrier between the BLAS and TLAS builds.

6. **Descriptor set.**
   - Layout: binding 0 = `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR`
     (stage COMPUTE), binding 1 = `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`.
   - Write binding 0 with `VkWriteDescriptorSetAccelerationStructureKHR`
     (chained into the `VkWriteDescriptorSet.pNext`, `descriptorType` =
     `..._ACCELERATION_STRUCTURE_KHR`, `pAccelerationStructures` = &TLAS).
   - Write binding 1 with the result SSBO.

7. **Pipeline + dispatch.** Compute pipeline from `ray_query.comp.spv`, bind set,
   `vkCmdDispatch(1,1,1)`, submit, wait.

8. **Read back** the SSBO and assert `hit == 1`, `primitiveIndex == 0`,
   `t ≈ 1.0`.

## 3. Expected result

`hit == 1` (committed triangle intersection) → `VK_KHR_ray_query` works on the
Apple GPU through MoltenVK. A `hit == 0` would indicate the AS was not bound /
not resident, or the ray-query capability did not survive SPIR-V → MSL.

## 4. Host programs in this directory

Two self-contained C hosts implement the outline above and both print
`RESULT: hit=1 t=1.0000 primitiveIndex=0 instanceId=0` on an M4 Max:

- **`host.c` → `rqtest`** links MoltenVK **directly** (`-lMoltenVK`, no loader).
  Build:
  ```sh
  cc host.c -o rqtest -I ../../External/Vulkan-Headers/include \
     -L<Package/Debug/.../macOS> -lMoltenVK -Wl,-rpath,<same>
  ./rqtest
  ```

- **`host_loader.c` → `rqtest_loader`** goes through the **Vulkan loader**
  (`-lvulkan`, brew libvulkan), the path a normal app / `ash::Entry::load()` uses.
  Build & run:
  ```sh
  cc host_loader.c -o rqtest_loader -I ../../External/Vulkan-Headers/include \
     -L/opt/homebrew/lib -lvulkan -Wl,-rpath,/opt/homebrew/lib
  VK_DRIVER_FILES=<Package/Debug/.../MoltenVK_icd.json> \
    DYLD_LIBRARY_PATH=/opt/homebrew/lib ./rqtest_loader
  ```

## 5. The loader enumeration quirk (important for loader-based apps)

The Vulkan **loader** (brew `vulkan-loader` 1.4.341) omits
`VK_KHR_acceleration_structure` and `VK_KHR_ray_query` — and **only** those two —
from `vkEnumerateDeviceExtensionProperties` for the MoltenVK *portability*
driver. Diagnosed in detail:

- MoltenVK advertises both correctly: its instance "supported extensions" banner
  lists 155 incl. AS/ray_query, and a **direct** (`-lMoltenVK`) device
  enumeration returns 133 incl. both.
- Through the loader the same dylib's device enumeration returns 131 — exactly
  AS + ray_query removed. Confirmed *not* a stale build, *not* the brew static
  lib, *not* an implicit layer, and *not* MoltenVK's own device-extension gating
  (all of AS's deps — `deferred_host_operations`, `buffer_device_address`,
  `descriptor_indexing` — pass through fine).
- Root cause is loader-side: AS declares a `depends` on the **instance**
  extension `VK_KHR_get_physical_device_properties2`; the loader's portability
  path checks that against the **device** list (where it never appears) and
  drops AS, then ray_query (which `depends` on AS). Enabling GPDP2 at the
  instance does not change it.

**The filter is cosmetic.** Through the loader you can still:
- `vkCreateDevice` with AS + ray_query enabled → `VK_SUCCESS`, and
- `vkGetDeviceProcAddr("vkGetAccelerationStructureBuildSizesKHR")` → non-null,

and the full BLAS/TLAS/ray-query path runs (`rqtest_loader` → `hit=1`).

**Rule for loader-based apps (incl. Orbis `aqueduct-gpu-host`):** detect
ray-query support via `vkGetPhysicalDeviceFeatures2` →
`VkPhysicalDeviceRayQueryFeaturesKHR.rayQuery` (reported correctly through the
loader), **not** via device-extension enumeration, and request
`VK_KHR_acceleration_structure` / `VK_KHR_ray_query` at device creation
unconditionally. See `host_loader.c` for the exact pattern.
