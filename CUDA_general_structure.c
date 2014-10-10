// general cuda code structure - 2014-10-09

// includes in the -
#include <stdio.h>


__global__
// global memory. as opposed to __shared__ and __local__.

void kernel_name(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  // blockIdx.x , blockDim.x and threadIdx.x are three internal indexers that
  // can help find the thread absolut number.
  // blockIdx = the number of block.
  // threadIdx = the number of thread in a specific block.
  // blockDim = the size of a block.
  
  // following formulae from this helpful thread at Udacity.com forums - 
  // http://forums.udacity.com/questions/100027222/an-intuition-for-finding-the-index-for-a-given-grid-size-and-block-size-hw1

    // threadsPerBlock = blockDim.x * blockDim.y
    // blockId = blockIdx.x + (blockIdx.y * gridDim.x)
    // threadId = threadIdx.x + (threadIdx.y * blockDim.x)

    // globalIdx = (blockIdx * threadsPerBlock) + threadId

    // globalIdx = (blockId * threadsPerBlock) + threadIdx.x + (threadIdx.y * blockDim.x)


  // In this example every thread is calculates the grey intensity of a pixel 
  // from the RGB values, disregarding the alpha.
  int abs_loc_x = blockIdx.x + threadIdx.x;
  int abs_loc_y = blockIdx.y + threadIdx.y;
  
  uchar4 rgba = rgbaImage[abs_loc_x * numCols+ abs_loc_y];
  float i = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
  greyImage[abs_loc_x * numCols+ abs_loc_y] = i;
}



void main_program_name(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
    
// dim3 is a struct of 3 intergers that defines an array of up to 3 dimentions,
// and is used to specify the size of grids or blocks.
// dim3 grid(64); defines a grid of 64 x 1 x 1 blocks.
// dim3 block(8,8); defines a block of 8 x 8 x 1 threads.

{
  const dim3 blockSize(numRows, 1, 1);  
  const dim3 gridSize(1, numCols, 1);  
  kernel_name<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  // the main_program_name<<<gridSize, blockSize>>>(...) header takes the number of grids and blocks
  // in the <<<>>> area and otherwise the same as the declaration at the top.
  // This will run the function at the top block x threads times.
  // The blockIdx and threadIdx assist in finding the thread absolut number.



  cudaDeviceSynchronize();
  // Acts a a barrier. Blocks until the device has completed all preceding requested tasks.


}