#pragma once
#include "runtime_contract.hpp"
#include "zluda_tensorrt.h"
nvinfer1::IExecutionContext* ztrtAbiCreateContext(nvinfer1::ICudaEngine*,ZtrtEngine*,nvinfer1::ILogger*,bool,nvinfer1::IRuntimeConfig* config=nullptr,nvinfer1::IErrorRecorder* recorder=nullptr) noexcept;

bool ztrtAbiContextMatches(nvinfer1::IExecutionContext const*,nvinfer1::ICudaEngine const*) noexcept;
