// general cuda code structure - 2014-10-09

// includes -
#include <stdio.h>


// 

__global__
void main_function_name(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  // blockIdx.x , blockDim.x and threadIdx.x are three internal indexers that
  // can help 
  int abs_loc_x = blockIdx.x + threadIdx.x;
  int abs_loc_y = blockIdx.y + threadIdx.y;
  
  uchar4 rgba = rgbaImage[abs_loc_x * numCols+ abs_loc_y];
  float i = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
  greyImage[abs_loc_x * numCols+ abs_loc_y] = i;
}



void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
    
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(numRows, 1, 1);  //TODO
  const dim3 gridSize(1, numCols, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
    
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}