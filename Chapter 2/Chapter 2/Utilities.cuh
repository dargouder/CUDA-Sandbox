#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <sstream>
#include <iostream>


inline void HandleError(cudaError_t p_cuda_error) {
	if (p_cuda_error != cudaSuccess) {
		std::cout << cudaGetErrorString(p_cuda_error) << "in " << __FILE__ << " at line " << __LINE__ << std::endl;
		exit(EXIT_FAILURE);
	}
}


std::stringstream DeviceParameterInformation() {
	int count;
	cudaDeviceProp properties;
	HandleError(cudaGetDeviceCount(&count));
	std::stringstream ss;
	for (int i = 0; i < count; ++i){
		HandleError(cudaGetDeviceProperties(&properties, i));
		ss << "\t---\tGeneral Information for device " << i << std::endl;
		ss << "Name: " << properties.name << std::endl;
		ss << "Compute Capability: " << properties.major << "." << properties.minor << std::endl;
		ss << "Clock Rate: " << properties.clockRate << std::endl;
		ss << "Device copy overlap: ";
		if (properties.deviceOverlap) {
			ss << "Enabled" << std::endl;
		}
		else {
			ss << "Disabled" << std::endl;
		}

		ss << "Kernel Execution timeout: ";
		if (properties.kernelExecTimeoutEnabled) {
			ss << "Enabled" << std::endl;
		}
		else {
			ss << "Disabled" << std::endl;
		}


		ss << "\t---\tMemory Information for device " << i << "---" << std::endl;
		ss << "Total Global Memory: " << properties.totalGlobalMem << std::endl;
		ss << "Total Constant Memory: " << properties.totalConstMem << std::endl;
		ss << "Max Mem Pitch: " << properties.memPitch << std::endl;
		ss << "Texture Alignment: " << properties.textureAlignment << std::endl;

		ss << "\t---\tInformation for device " << i << " ---\t" << std::endl;
		ss << "Multiprocessor count: " << properties.multiProcessorCount << std::endl;
		ss << "Shared Memory per MP: " << properties.sharedMemPerBlock << std::endl;
		ss << "Registers per MP: " << properties.regsPerBlock << std::endl;
		ss << "Threads in Warp: " << properties.warpSize << std::endl;
		ss << "Max Threads per Block: " << properties.maxThreadsPerBlock << std::endl;
		ss << "Max thread dimensions: ( " << properties.maxThreadsDim[0] << "," << properties.maxThreadsDim[1] << ","
			<< properties.maxThreadsDim[2] << ")" << std::endl;
		ss << "Max grid dimensions: ( " << properties.maxGridSize[0] << "," << properties.maxGridSize[1] << ","
			<< properties.maxGridSize[2] << ")" << std::endl;

		ss << std::endl;
		return ss;
	}
}

class Complex {
public:
	float r;
	float i;

	Complex(float p_r, float p_i) : r(p_r), i(p_i) {

	}

	float magnitude2(void) {
		return r*r + i*i;
	}
	Complex operator*(const Complex& p_a) {
		return Complex(r * p_a.r - i*p_a.i, i*p_a.r + r * p_a.i);
	}

	Complex operator+(const Complex& p_a) {
		return Complex(r + p_a.r, i + p_a.i);
	}
};

class Complex_d{
public:
	float r;
	float i;

	__device__ Complex_d(float p_r, float p_i) : r(p_r), i(p_i) {

	}

	__device__ float magnitude2(void) {
		return r*r + i*i;
	}
	__device__ Complex_d operator*(const Complex_d& p_a) {
		return Complex_d(r * p_a.r - i*p_a.i, i*p_a.r + r * p_a.i);
	}

	__device__ Complex_d operator+(const Complex_d& p_a) {
		return Complex_d(r + p_a.r, i + p_a.i);
	}
};