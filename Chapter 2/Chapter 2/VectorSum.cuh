#pragma once
#include "Utilities.cuh"

const int n = 10;

__global__ void add(int *a, int *b, int* c){
	if (blockIdx.x < n) {
		c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	}
}

void VectorSum() {
	int a[n], b[n], c[n];
	int *d_a, *d_b, *d_c;

	/* Allocate the GPU memory */
	HandleError(cudaMalloc((void**)&d_a, n*sizeof(int)));
	HandleError(cudaMalloc((void**)&d_b, n*sizeof(int)));
	HandleError(cudaMalloc((void**)&d_c, n*sizeof(int)));

	for (int i = 0; i < n; i++){
		a[i] = -i;
		b[i] = i*i;
	}

	/* Copy arrays a and b to the GPU, to the respective arrays located on the GPU. */
	HandleError(cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice));
	HandleError(cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice));

	/* Launch kernel to perform vector sum */
	add << <n, 1 >> >(d_a, d_b, d_c);

	/* Copy the array c from the GPU back to the CPU */
	HandleError(cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost));

	/* Display the results */
	for (int i = 0; i < n; i++){
		std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
	}

	/* Free the memory allocated on the GPU */

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}