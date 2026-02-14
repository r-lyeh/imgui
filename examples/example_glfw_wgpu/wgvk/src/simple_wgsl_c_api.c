#include <simple_wgsl.h>
#include <spirv/unified1/spirv.h>
#include <string.h>

#define SUPPORT_VULKAN_BACKEND 1
#include <wgvk.h>
#include <wgvk_structs_impl.h>
#include <tint_c_api.h>

static WGPUShaderStage StageToMask(WgslStage stage) {
  switch (stage) {
    case WGSL_STAGE_VERTEX:   return WGPUShaderStage_Vertex;
    case WGSL_STAGE_FRAGMENT: return WGPUShaderStage_Fragment;
    case WGSL_STAGE_COMPUTE:  return WGPUShaderStage_Compute;
    default:                  return 0;
  }
}

static uint32_t StageToEnum(WgslStage stage) {
  switch (stage) {
    case WGSL_STAGE_VERTEX:   return WGPUShaderStageEnum_Vertex;
    case WGSL_STAGE_FRAGMENT: return WGPUShaderStageEnum_Fragment;
    case WGSL_STAGE_COMPUTE:  return WGPUShaderStageEnum_Compute;
    default:                  return 0;
  }
}

static WGPUTextureFormat SpvFormatToWGPU(uint32_t fmt) {
  switch (fmt) {
    case SpvImageFormatR32f:     return WGPUTextureFormat_R32Float;
    case SpvImageFormatR32ui:    return WGPUTextureFormat_R32Uint;
    case SpvImageFormatRgba8:    return WGPUTextureFormat_RGBA8Unorm;
    case SpvImageFormatRgba32f:  return WGPUTextureFormat_RGBA32Float;
    default:                     return WGPUTextureFormat_Undefined;
  }
}

static WGPUTextureViewDimension SsirDimToWGPU(SsirTextureDim dim) {
  switch (dim) {
    case SSIR_TEX_1D:         return WGPUTextureViewDimension_1D;
    case SSIR_TEX_2D:         return WGPUTextureViewDimension_2D;
    case SSIR_TEX_3D:         return WGPUTextureViewDimension_3D;
    case SSIR_TEX_CUBE:       return WGPUTextureViewDimension_Cube;
    case SSIR_TEX_2D_ARRAY:   return WGPUTextureViewDimension_2DArray;
    case SSIR_TEX_CUBE_ARRAY: return WGPUTextureViewDimension_CubeArray;
    default:                  return WGPUTextureViewDimension_2D;
  }
}

static WGPUStorageTextureAccess SsirAccessToWGPU(SsirAccessMode access) {
  switch (access) {
    case SSIR_ACCESS_READ:       return WGPUStorageTextureAccess_ReadOnly;
    case SSIR_ACCESS_WRITE:      return WGPUStorageTextureAccess_WriteOnly;
    case SSIR_ACCESS_READ_WRITE: return WGPUStorageTextureAccess_ReadWrite;
    default:                     return WGPUStorageTextureAccess_WriteOnly;
  }
}

static WGPUTextureSampleType SsirSampledTypeToWGPU(
    SsirModule *mod, uint32_t sampled_type) {
  SsirType *t = ssir_get_type(mod, sampled_type);
  if (!t) return WGPUTextureSampleType_Float;
  switch (t->kind) {
    case SSIR_TYPE_F32:  return WGPUTextureSampleType_Float;
    case SSIR_TYPE_U32:  return WGPUTextureSampleType_Uint;
    case SSIR_TYPE_I32:  return WGPUTextureSampleType_Sint;
    default:             return WGPUTextureSampleType_Float;
  }
}

static uint32_t SsirTypeByteSize(SsirModule *mod, uint32_t type_id) {
  SsirType *t = ssir_get_type(mod, type_id);
  if (!t) return 0;
  switch (t->kind) {
    case SSIR_TYPE_BOOL:
    case SSIR_TYPE_I32:
    case SSIR_TYPE_U32:
    case SSIR_TYPE_F32:
      return 4;
    case SSIR_TYPE_F16:
    case SSIR_TYPE_I16:
    case SSIR_TYPE_U16:
      return 2;
    case SSIR_TYPE_F64:
    case SSIR_TYPE_I64:
    case SSIR_TYPE_U64:
      return 8;
    case SSIR_TYPE_I8:
    case SSIR_TYPE_U8:
      return 1;
    case SSIR_TYPE_VEC:
      return t->vec.size * SsirTypeByteSize(mod, t->vec.elem);
    case SSIR_TYPE_MAT:
      return t->mat.cols * t->mat.rows * SsirTypeByteSize(mod, t->mat.elem);
    case SSIR_TYPE_ARRAY:
      return t->array.length * SsirTypeByteSize(mod, t->array.elem);
    case SSIR_TYPE_STRUCT: {
      if (t->struc.member_count == 0) return 0;
      uint32_t last = t->struc.member_count - 1;
      uint32_t last_offset = t->struc.offsets ? t->struc.offsets[last] : 0;
      return last_offset + SsirTypeByteSize(mod, t->struc.members[last]);
    }
    case SSIR_TYPE_PTR:
      return SsirTypeByteSize(mod, t->ptr.pointee);
    default:
      return 0;
  }
}

RGAPI tc_SpirvBlob wgslToSpirv(
    const WGPUShaderSourceWGSL *source,
    uint32_t constantCount,
    const WGPUConstantEntry *constants) {
  tc_SpirvBlob ret;
  memset(&ret, 0, sizeof(ret));

  size_t length = (source->code.length == WGPU_STRLEN)
                      ? strlen(source->code.data)
                      : source->code.length;

  char *src = (char *)RL_CALLOC(length + 1, 1);
  memcpy(src, source->code.data, length);
  src[length] = '\0';

  WgslAstNode *ast = wgsl_parse(src);
  if (!ast) {
    RL_FREE(src);
    return ret;
  }

  WgslResolver *resolver = wgsl_resolver_build(ast);
  if (!resolver) {
    wgsl_free_ast(ast);
    RL_FREE(src);
    return ret;
  }

  int ep_count = 0;
  const WgslResolverEntrypoint *eps =
      wgsl_resolver_entrypoints(resolver, &ep_count);

  WgslLowerOptions opts;
  memset(&opts, 0, sizeof(opts));
  opts.env = WGSL_LOWER_ENV_VULKAN_1_1;

  uint32_t *spirv = NULL;
  size_t word_count = 0;
  WgslLowerResult res =
      wgsl_lower_emit_spirv(ast, resolver, &opts, &spirv, &word_count);

  if (res != WGSL_LOWER_OK || !spirv) {
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    RL_FREE(src);
    return ret;
  }

  size_t byte_count = word_count * sizeof(uint32_t);
  for (int i = 0; i < ep_count && i < 16; i++) {
    uint32_t stage_idx = StageToEnum(eps[i].stage);
    if (stage_idx >= 16) continue;

    ret.entryPoints[stage_idx].codeSize = byte_count;
    ret.entryPoints[stage_idx].code =
        (uint32_t *)RL_CALLOC(word_count, sizeof(uint32_t));
    memcpy(ret.entryPoints[stage_idx].code, spirv, byte_count);

    size_t name_len = strlen(eps[i].name);
    if (name_len > 15) name_len = 15;
    memcpy(ret.entryPoints[stage_idx].entryPointName, eps[i].name, name_len);
  }

  wgsl_lower_free(spirv);
  wgsl_resolver_free(resolver);
  wgsl_free_ast(ast);
  RL_FREE(src);
  return ret;
}

RGAPI WGPUReflectionInfo reflectionInfo_wgsl_sync(WGPUStringView wgslSource) {
  WGPUReflectionInfo ret;
  memset(&ret, 0, sizeof(ret));

  size_t length = (wgslSource.length == WGPU_STRLEN)
                      ? strlen(wgslSource.data)
                      : wgslSource.length;

  char *src = (char *)RL_CALLOC(length + 1, 1);
  memcpy(src, wgslSource.data, length);
  src[length] = '\0';

  WgslAstNode *ast = wgsl_parse(src);
  if (!ast) {
    RL_FREE(src);
    return ret;
  }

  WgslResolver *resolver = wgsl_resolver_build(ast);
  if (!resolver) {
    wgsl_free_ast(ast);
    RL_FREE(src);
    return ret;
  }

  WgslLowerOptions opts;
  memset(&opts, 0, sizeof(opts));
  opts.env = WGSL_LOWER_ENV_VULKAN_1_1;
  opts.enable_debug_names = 1;

  WgslLower *lower = wgsl_lower_create(ast, resolver, &opts);
  if (!lower) {
    wgsl_resolver_free(resolver);
    wgsl_free_ast(ast);
    RL_FREE(src);
    return ret;
  }

  const SsirModule *cmod = wgsl_lower_get_ssir(lower);
  SsirModule *mod = (SsirModule *)cmod;

  int binding_count = 0;
  const WgslSymbolInfo *bindings =
      wgsl_resolver_binding_vars(resolver, &binding_count);

  int ep_count = 0;
  const WgslResolverEntrypoint *eps =
      wgsl_resolver_entrypoints(resolver, &ep_count);

  if (binding_count > 0) {
    ret.globals = (WGPUGlobalReflectionInfo *)RL_CALLOC(
        (size_t)binding_count, sizeof(WGPUGlobalReflectionInfo));
    ret.globalCount = (uint32_t)binding_count;

    for (int i = 0; i < binding_count; i++) {
      WGPUGlobalReflectionInfo *g =
          (WGPUGlobalReflectionInfo *)&ret.globals[i];
      g->bindGroup = (uint32_t)bindings[i].group_index;
      g->binding = (uint32_t)bindings[i].binding_index;
      g->name.data = bindings[i].name;
      g->name.length = strlen(bindings[i].name);

      WGPUShaderStage vis = 0;
      for (int e = 0; e < ep_count; e++) {
        int ep_bind_count = 0;
        const WgslSymbolInfo *ep_binds =
            wgsl_resolver_entrypoint_binding_vars(
                resolver, eps[e].name, &ep_bind_count);
        for (int b = 0; b < ep_bind_count; b++) {
          if (ep_binds[b].id == bindings[i].id) {
            vis |= StageToMask(eps[e].stage);
            break;
          }
        }
      }
      g->visibility = vis;

      if (!mod) continue;

      for (uint32_t gi = 0; gi < mod->global_count; gi++) {
        SsirGlobalVar *gv = &mod->globals[gi];
        if (!gv->has_group || !gv->has_binding) continue;
        if (gv->group != (uint32_t)bindings[i].group_index) continue;
        if (gv->binding != (uint32_t)bindings[i].binding_index) continue;

        SsirType *ptr_type = ssir_get_type(mod, gv->type);
        if (!ptr_type) break;

        SsirType *inner = ptr_type;
        SsirAddressSpace addr_space = SSIR_ADDR_UNIFORM_CONSTANT;
        if (ptr_type->kind == SSIR_TYPE_PTR) {
          addr_space = ptr_type->ptr.space;
          inner = ssir_get_type(mod, ptr_type->ptr.pointee);
        }

        if (!inner) break;

        switch (inner->kind) {
          case SSIR_TYPE_SAMPLER:
            g->sampler.type = WGPUSamplerBindingType_Filtering;
            break;
          case SSIR_TYPE_SAMPLER_COMPARISON:
            g->sampler.type = WGPUSamplerBindingType_Comparison;
            break;
          case SSIR_TYPE_TEXTURE:
            g->texture.sampleType =
                SsirSampledTypeToWGPU(mod, inner->texture.sampled_type);
            g->texture.viewDimension = SsirDimToWGPU(inner->texture.dim);
            break;
          case SSIR_TYPE_TEXTURE_DEPTH:
            g->texture.sampleType = WGPUTextureSampleType_Depth;
            g->texture.viewDimension = SsirDimToWGPU(inner->texture_depth.dim);
            break;
          case SSIR_TYPE_TEXTURE_STORAGE:
            g->storageTexture.format =
                SpvFormatToWGPU(inner->texture_storage.format);
            g->storageTexture.access =
                SsirAccessToWGPU(inner->texture_storage.access);
            g->storageTexture.viewDimension =
                SsirDimToWGPU(inner->texture_storage.dim);
            break;
          default:
            if (addr_space == SSIR_ADDR_UNIFORM) {
              g->buffer.type = WGPUBufferBindingType_Uniform;
            } else if (addr_space == SSIR_ADDR_STORAGE) {
              if (gv->non_writable)
                g->buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
              else
                g->buffer.type = WGPUBufferBindingType_Storage;
            }
            g->buffer.minBindingSize = SsirTypeByteSize(mod, gv->type);
            break;
        }
        break;
      }
    }
  }

  wgsl_lower_destroy(lower);
  wgsl_resolver_free(resolver);
  wgsl_free_ast(ast);
  RL_FREE(src);
  return ret;
}

RGAPI void reflectionInfo_wgsl_free(WGPUReflectionInfo *reflectionInfo) {
  RL_FREE((void *)reflectionInfo->globals);
  RL_FREE((void *)reflectionInfo->inputAttributes);
  RL_FREE((void *)reflectionInfo->outputAttributes);
}
