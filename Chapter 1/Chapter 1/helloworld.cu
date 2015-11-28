
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <sstream>


__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}
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

int main()
{
	int c;
	int* d_c;
	std::cout << DeviceParameterInformation().str() << std::endl;
	HandleError(cudaMalloc((void**)&d_c, sizeof(int)));

	add << <1, 1 >> >(2, 7, d_c);
	HandleError(cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
	
	std::cout << " result is " << c  << std::endl;
	cudaFree(d_c);
	return 0;
}