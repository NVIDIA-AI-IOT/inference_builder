from .model_config_pb2 import DataType

datatype_mapping = {
    "TYPE_INVALID": DataType.TYPE_INVALID,
    "TYPE_BOOL": DataType.TYPE_BOOL,
    "TYPE_UINT8": DataType.TYPE_UINT8,
    "TYPE_UINT16": DataType.TYPE_UINT16,
    "TYPE_UINT32": DataType.TYPE_UINT32,
    "TYPE_UINT64": DataType.TYPE_UINT64,
    "TYPE_INT8": DataType.TYPE_INT8,
    "TYPE_INT16": DataType.TYPE_INT16,
    "TYPE_INT32": DataType.TYPE_INT32,
    "TYPE_INT64": DataType.TYPE_INT64,
    "TYPE_FP16": DataType.TYPE_FP16,
    "TYPE_FP32": DataType.TYPE_FP32,
    "TYPE_FP64": DataType.TYPE_FP64,
    "TYPE_STRING": DataType.TYPE_STRING,
    "TYPE_BF16": DataType.TYPE_BF16
}