#pragma once

#include "Image.h"
#include "Utilities.cuh"
const int IMAGE_HEIGHT = 600;
const int IMAGE_WIDTH = 800;

int julia(int p_x, int p_y){

}
void kernel(RGBColour* gpu_image){
	for (int y = 0; y < IMAGE_HEIGHT; y++){
		for (int x = 0; x < IMAGE_WIDTH; y++){
			int offset = x + y * IMAGE_WIDTH;

			int julia_value = julia(x, y);

		}
	}
}
void ComputeJuliaSet(){
	Image* cpu_image = new Image(IMAGE_WIDTH, IMAGE_HEIGHT);
	RGBColour* gpu_image;

	HandleError(cudaMalloc((void**)&gpu_image, cpu_image->size));

	kernel(gpu_image);

	cpu_image->writePPM("juliaset.ppm");
	delete cpu_image;

}