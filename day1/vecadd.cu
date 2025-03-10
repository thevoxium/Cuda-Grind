#include <iostream>
#include <cuda_runtime.h>

#define threads_per_block 256

__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *a_h, *b_h, *c_h;
    float *a_d, *b_d, *c_d;
    int n = 1000000;

    a_h = (float*)malloc(n * sizeof(float));
    b_h = (float*)malloc(n * sizeof(float));
    c_h = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a_h[i] = i;
        b_h[i] = 2 * i;
    } 

    cudaMalloc((void**)&a_d, n * sizeof(float));
    cudaMalloc((void**)&b_d, n * sizeof(float));
    cudaMalloc((void**)&c_d, n * sizeof(float));

    cudaMemcpy(a_d, a_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(float), cudaMemcpyHostToDevice);
    
    vecAdd<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c_h, c_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f %f %f\n", a_h[i], b_h[i], c_h[i]);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}
