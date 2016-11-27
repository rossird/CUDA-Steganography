#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>


/**
 
 We are encoding 4 bits per pixel (one bit per channel).
 Wwe will have one thread working on a pixel.
 So we will need to have 2*dataSize number of threads.
 
 */
void encode_parallel(const uchar4* const h_sourceImg,
                     uchar4* const h_destImg,
                     const char* const h_binData,
                     int numBytesData,
                     const size_t numRowsSource, const size_t numColsSource)
{

  //Allocate device memory
  uchar4* d_sourceImg;
  uchar4* d_destImg;
  char* d_binData;
  cudaMalloc(&d_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource);
  cudaMalloc(&d_destImg, sizeof(uchar4) * numRowsSource * numColsSource);
  cudaMalloc(&d_binData, sizeof(char) * numBytesData);

  //Execute 1 thread per pixel of output image.
  //This means 1 thread per 4 bits of data.
  int numThreads = ceil(numBytesData / 4);
  
  //Free memory
  cudaFree(d_sourceImg);
  cudaFree(d_destImg);
  cudaFree(d_binData);
                  
}
