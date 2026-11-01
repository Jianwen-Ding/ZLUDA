#pragma once
#include "runtime_contract.hpp"
#include "zluda_tensorrt.h"
nvinfer1::IExecutionContext* ztrtAbiCreateContext(nvinfer1::ICudaEngine*,ZtrtEngine*,nvinfer1::ILogger*) noexcept;
