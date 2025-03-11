#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

void printMatrix(float* matrix, int Width, const char* name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%.1f ", matrix[i * Width + j]);
        }
        printf("\n");
    }
}

int main() {
    int Width = 10;
    size_t size = Width * Width * sizeof(float);

    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);

    for (int i = 0; i < Width * Width; i++) {
        h_M[i] = 1.0f;
        h_N[i] = 2.0f;
    }

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_M, d_N, d_P, Width);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Print a subset of the matrices
    printMatrix(h_M, Width, "Input Matrix M");
    printMatrix(h_N, Width, "Input Matrix N");
    printMatrix(h_P, Width, "Output Matrix P");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
