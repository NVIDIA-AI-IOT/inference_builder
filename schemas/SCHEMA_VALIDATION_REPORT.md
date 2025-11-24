# Schema Validation Report

**Date**: 2025-01-20
**Task**: Cross-check schema definitions against project documentation (README.md, doc/architecture.md, doc/usage.md)

## Summary

Cross-checked the JSON schemas against the Inference Builder documentation and sample configurations. Found and fixed several inconsistencies between the schema definitions and the documented behavior.

## Issues Found and Fixed

### 1. ✅ Input/Output Required vs Optional

**Issue**: Mismatch between schema and documentation

| Component | Original State | Fixed State |
|-----------|---------------|-------------|
| **Schema** | `input` and `output` marked as **required** | Changed to **optional** |
| **Documentation** | States they are optional (lines 41-42 in usage.md) | No change needed |
| **Sample Files** | ALL samples include both fields | No change needed |

**Resolution**: Updated schema to match documentation. Made `input` and `output` optional with clarified descriptions explaining when they're needed:
- Required when pipeline includes multiple models
- Required when inputs/outputs use custom types requiring pre/postprocessing

**Changed in**: `config.schema.json`

---

### 2. ✅ Top-level Preprocessors

**Issue**: Schema allowed `preprocessors` at top level, but documentation doesn't mention this

| Component | Original State | Fixed State |
|-----------|---------------|-------------|
| **Schema** | Included `preprocessors` property | Removed |
| **Documentation** | Only mentions `postprocessors` at top level | No change needed |
| **Sample Files** | None use top-level preprocessors | No change needed |

**Resolution**: Removed `preprocessors` from top-level schema. Only `postprocessors` are valid at top level (for consolidating multi-model outputs).

**Changed in**: `config.schema.json`

---

### 3. ✅ Missing Backend Types

**Issue**: Documentation mentions backends not included in schema

**Missing Backends Identified**:

| Backend | Documentation Reference | Action Taken |
|---------|------------------------|--------------|
| `deepstream/nvinferserver` | usage.md line 113 | ✅ Added to schema |
| `pytorch` | usage.md line 118 | ✅ Added schema + parameters |
| `triton/python/tensorrtllm` | usage.md line 115 | ✅ Added to triton enum |
| `triton/tensorflow` | Already in schema | ✅ Kept |

**Resolution**:
- Added `deepstream/nvinferserver` to DeepStream backend enum
- Created new `pytorch` backend schema for Huggingface Transformers support
- Created `pytorch-parameters.schema.json` with relevant parameters
- Added `triton/python/tensorrtllm` to Triton backend enum
- Updated `common/definitions.schema.json` with all backend types

**Changed in**:
- `config.schema.json`
- `backends/deepstream.schema.json`
- `backends/triton.schema.json`
- `backends/pytorch.schema.json` (new)
- `backends/parameters/pytorch-parameters.schema.json` (new)
- `common/definitions.schema.json`
- `index.json`

---

### 4. ✅ Updated Descriptions

**Issue**: Schema descriptions didn't explain conditional requirements clearly

**Updated Descriptions**:

| Field | Old Description | New Description |
|-------|----------------|-----------------|
| `input` | "Top level inference input specification" | "Top level inference input specification. Required when pipeline includes multiple models or when inputs use custom types requiring preprocessing." |
| `output` | "Top level inference output specification" | "Top level inference output specification. Required when pipeline includes multiple models or when outputs use custom types requiring postprocessing." |
| `postprocessors` | "Global postprocessors" | "Top-level postprocessors. Required when pipeline includes multiple models and outputs need to be consolidated." |

**Changed in**: `config.schema.json`

---

## Complete Backend Support Matrix

After fixes, the schema now supports all documented backends:

| Backend | Schema Support | Parameters Schema | Documentation |
|---------|---------------|-------------------|---------------|
| `dummy` | ✅ | ✅ `dummy-parameters.schema.json` | Testing |
| `vllm` | ✅ | ✅ `vllm-parameters.schema.json` | Large language models |
| `triton/python` | ✅ | ✅ `triton-parameters.schema.json` | Triton Python backend |
| `triton/tensorrt` | ✅ | ✅ `triton-parameters.schema.json` | Triton TensorRT backend |
| `triton/onnx` | ✅ | ✅ `triton-parameters.schema.json` | Triton ONNX backend |
| `triton/pytorch` | ✅ | ✅ `triton-parameters.schema.json` | Triton PyTorch backend |
| `triton/tensorflow` | ✅ | ✅ `triton-parameters.schema.json` | Triton TensorFlow backend |
| `triton/python/tensorrtllm` | ✅ | ✅ `triton-parameters.schema.json` | Triton with TensorRT-LLM |
| `tensorrtllm` | ✅ | ✅ `tensorrtllm-parameters.schema.json` | TensorRT-LLM |
| `tensorrtllm/pytorch` | ✅ | ✅ `tensorrtllm-parameters.schema.json` | TensorRT-LLM with PyTorch |
| `deepstream/nvinfer` | ✅ | ✅ `deepstream-parameters.schema.json` | DeepStream nvinfer |
| `deepstream/nvinferserver` | ✅ | ✅ `deepstream-parameters.schema.json` | DeepStream nvinferserver |
| `polygraphy` | ✅ | ✅ `polygraphy-parameters.schema.json` | TensorRT via Polygraphy |
| `pytorch` | ✅ | ✅ `pytorch-parameters.schema.json` | Huggingface Transformers |

---

## Verification

All changes were verified against:

1. **Documentation Files**:
   - `README.md`
   - `doc/architecture.md`
   - `doc/usage.md`

2. **Sample Configurations**:
   - ✅ `builder/samples/vllm/*.yaml`
   - ✅ `builder/samples/qwen/*.yaml`
   - ✅ `builder/samples/tao/*.yaml`
   - ✅ `builder/samples/nvclip/*.yaml`
   - ✅ `builder/samples/changenet/*.yaml`
   - ✅ `builder/samples/ds_app/**/*.yaml`
   - ✅ `builder/samples/dummy/*.yaml`

3. **Architecture Alignment**:
   - ✅ Processor interface (preprocessors/postprocessors)
   - ✅ ModelOperator and ModelBackend concepts
   - ✅ Data flow through routes
   - ✅ Server integration patterns

---

## Schema Structure Improvements

The validation also confirmed the improved schema architecture:

```
✅ Unified base model schema (common/base-model.schema.json)
✅ Separate parameter schemas per backend (backends/parameters/*.json)
✅ Composition via allOf for backend schemas
✅ Shared definitions in common/definitions.schema.json
✅ Consistent preprocessor/postprocessor specs
```

---

## Remaining Considerations

### 1. Server Types

The documentation mentions these server types (usage.md line 9):
- `triton`
- `fastapi`
- `nim`
- `serverless`

These are command-line options, not schema-validated. Consider adding CLI argument validation in future.

### 2. Data Types

All documented data types are supported in the schema:
- ✅ Basic types (TYPE_BOOL, TYPE_INT*, TYPE_FP*, TYPE_STRING)
- ✅ Custom image types (TYPE_CUSTOM_IMAGE_BASE64, TYPE_CUSTOM_IMAGE_ASSETS)
- ✅ Custom video types (TYPE_CUSTOM_VIDEO_ASSETS, TYPE_CUSTOM_VIDEO_CHUNK_ASSETS)
- ✅ Custom binary types (TYPE_CUSTOM_BINARY_BASE64, TYPE_CUSTOM_BINARY_URLS)
- ✅ DeepStream types (TYPE_CUSTOM_DS_*)
- ✅ VLM types (TYPE_CUSTOM_VLM_INPUT)
- ✅ Generic types (TYPE_CUSTOM_OBJECT)

### 3. Routes Format

Routes are validated as object with string values. The format `<model>:["tensor"]` is documented but not strictly validated. Consider adding pattern validation in future.

---

## Conclusion

✅ **All major inconsistencies have been resolved**
✅ **Schema now accurately reflects documented behavior**
✅ **All documented backends are supported**
✅ **Descriptions clarify conditional requirements**
✅ **Schema structure follows best practices**

The schemas are now ready for use and should provide accurate validation and IDE support for Inference Builder configurations.

