#include <stdio.h>
#include <stdlib.h> 

#define width 4
#define height 4
#define channels 1
#define BLUR_SIZE 1

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        for(int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1; ++blurRow) {
            for(int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if(curRow>=0 && curRow<h && curCol>=0 && curCol<w) {
                    pixVal += in[curRow*w + curCol];
                    ++pixels;
                }
            }
        }
        out[row*w + col] = (unsigned char)(pixVal/pixels);
    }
}

int main(){
    unsigned char* pin_h = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char* pout_h = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    unsigned char* pin_d, *pout_d;

    cudaMalloc((void**)&pin_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&pout_d, width * height * sizeof(unsigned char));

    for (int i=0; i<width * height; i++){
        pin_h[i] = rand()%256;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    cudaMemcpy(pin_d, pin_h, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(pin_d, pout_d, width, height);

    cudaMemcpy(pout_h, pout_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++) {
        printf("%d ", pout_h[i]);
        if ((i + 1) % width == 0) printf("\n");
    }
    free(pin_h);
    free(pout_h);
    cudaFree(pin_d);
    cudaFree(pout_d);
}
