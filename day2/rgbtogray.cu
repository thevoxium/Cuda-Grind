#include <stdio.h>
#include <stdlib.h> 

#define width 1200
#define height 1200
#define channels 3


__global__ void convertToGray(unsigned char* pin, unsigned char* pout){
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.y;

    if (row < height && col < width){
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * channels;
        unsigned char r = pin[rgbOffset];
        unsigned char g = pin[rgbOffset + 1];
        unsigned char b = pin[rgbOffset + 2];
        pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}


int main(){
    unsigned char* pin_h = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));
    unsigned char* pout_h = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));

    unsigned char* pin_d, *pout_d;

    cudaMalloc((void**)&pin_d, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&pout_d, width * height * channels * sizeof(unsigned char));

    for (int i=0; i<width * height * channels; i++){
        pin_h[i] = rand()%256;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    cudaMemcpy(pin_d, pin_h, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    convertToGray<<<blocksPerGrid, threadsPerBlock>>>(pin_d, pout_d);

    cudaMemcpy(pout_h, pout_d, width*height*channels*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++) {
        printf("%d ", pout_h[i]);
        if ((i + 1) % width == 0) printf("\n");
    }
    free(pin_h);
    free(pout_h);
    cudaFree(pin_d);
    cudaFree(pout_d);
}
