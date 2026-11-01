#pragma once
#include "runtime_contract.hpp"
nvinfer1::IRuntimeConfig* ztrtAbiCreateConfig(nvinfer1::ICudaEngine*,nvinfer1::ILogger*,int32_t strategy=0) noexcept;
nvinfer1::IRuntimeConfig* ztrtAbiCloneConfig(nvinfer1::IRuntimeConfig*,nvinfer1::ICudaEngine*,nvinfer1::ILogger*) noexcept;
