# Cuda-Grind

Following the kernels in the PMPP book for the Cuda Challenge by Umar Jamil

#### Day 1
File : vecadd.cu \
Details: Implemented the simple vector addition kernel shown in the first chapter of PMPP book. \
Learned:
- Function Declarations (global, device, host)
- Kernel Call & Grid Launch (<<<>>>)
- Built In varibles (blockDim, blockIdx, threadIdx)
- Functions provided by the CUDA API (eg. cudaMalloc, cudaFree, cudaMemcpy)

#### Day 2 
File : rgbtogray.cu \
Detail: Implemented the rgb to gray conversion kernel shown in the third chapter of the PMPP book. \

File blurkernel_1channel.cu \
Detail: Implemented the blur kernel example shown in the third chapter of the PMPP book. \
Learned:
- multidimensional grids and blocks in Cuda
- how to map threads to multidimensional data
- read half of chapter 3 of PMPP book

#### Day 3
File : mat_mul.cu \
Details: Implemented the matrix multiplication kernel mentioned in the third chapter of PMPP book. \
Learned:
- Almost same learning as day2 continued as I was still finishing the remaining 3rd chapter.


