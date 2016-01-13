#pragma once

#include "Image.h"
#include "Utilities.cuh"

const int IMAGE_WIDTH = 800;
const int IMAGE_HEIGHT = 600;
int julia(int p_x, int p_y){
	const float scale = 1.5;

	float jx = scale * (float)(IMAGE_WIDTH / 2 - p_x) / (IMAGE_WIDTH / 2);
	float jy = scale * (float)(IMAGE_HEIGHT / 2 - p_y) / (IMAGE_HEIGHT / 2);

	Complex c(-0.1f, 0.651f);
	Complex a(jx, jy);

	for (int i = 0; i < 200; ++i){
		a = a * a + c;
		if (a.magnitude2() > 1000){
			return 0;
		}
	}

	return 1;

}
void julia_cpu(RGBColour* p_image){
	for (int x = 0; x < IMAGE_WIDTH; x++){
		for (int y = 0; y < IMAGE_HEIGHT; y++){
			int julia_value = julia(x, y);
			int index = x + y * IMAGE_WIDTH;
			p_image[index].r = 255 * julia_value;
		}
	}
}
void ComputeJuliaSetCPU(){
	Image* cpu_image = new Image(IMAGE_WIDTH, IMAGE_HEIGHT);
	//RGBColour* gpu_image;

	//HandleError(cudaMalloc((void**)&gpu_image, cpu_image->size));

	julia_cpu(cpu_image->image);

	cpu_image->writePPM("juliaset.ppm");
	delete cpu_image;

}

__device__ int julia_d(int p_x, int p_y){
	const float scale = 1.5;

	float jx = scale * (float)(IMAGE_WIDTH / 2 - p_x) / (IMAGE_WIDTH / 2);
	float jy = scale * (float)(IMAGE_HEIGHT / 2 - p_y) / (IMAGE_HEIGHT / 2);

	Complex_d c(-0.1f, 0.651f);
	Complex_d a(jx, jy);

	for (int i = 0; i < 200; ++i){
		a = a * a + c;
		if (a.magnitude2() > 1000){
			return 0;
		}
	}

	return 1;
}
__global__ void kernel(RGBColour* p_image){

	/* map from threadIdx/BlockIdx to pixel position */

	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	/* calculate the value at this position */
	int julia_value = julia_d(x, y);
	p_image[offset].r = 255 * julia_value;
}


void ComputeJuliaSetGPU() {
	Image* cpu_image = new Image(IMAGE_WIDTH, IMAGE_HEIGHT);
	RGBColour* gpu_image;


	size_t pitch;
	HandleError(cudaMalloc((void**)&gpu_image,cpu_image->size));

	dim3 grid(IMAGE_WIDTH, IMAGE_HEIGHT);
	kernel<<<grid,1>>>(gpu_image);

	HandleError(cudaMemcpy(cpu_image->image, gpu_image, cpu_image->size, cudaMemcpyDeviceToHost));


	cpu_image->writePPM("juliaset.ppm");
	delete cpu_image;
	cudaFree(gpu_image);
}