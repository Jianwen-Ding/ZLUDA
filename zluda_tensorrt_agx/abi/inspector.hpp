#pragma once
#include "runtime_contract.hpp"
#include "zluda_tensorrt.h"
nvinfer1::IEngineInspector* ztrtAbiCreateInspector(nvinfer1::ICudaEngine*,ZtrtEngine*,nvinfer1::ILogger*,nvinfer1::IErrorRecorder* recorder=nullptr) noexcept;
