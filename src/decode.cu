#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

using namespace std;



// 1 byte is stored in 2 pixels
// extract 1 byte per thread
__global__ void decode_per_byte(uchar4* const d_encodedImage, unsigned char* d_encodedData, int numBytes) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int curr_pixel = 2*idx;

  if (curr_pixel+1 >= numBytes) {
    // We don't have a complete byte, return
    return;
  }

  bool bits[8];

  // Let's bring the pixels to local memory
  uchar4 pixel1 = d_encodedImage[curr_pixel];
  uchar4 pixel2 = d_encodedImage[curr_pixel + 1];

  bits[0] = pixel1.x & 1;
  bits[1] = pixel1.y & 1;
  bits[2] = pixel1.z & 1;
  bits[3] = pixel1.w & 1;
  bits[4] = pixel2.x & 1;
  bits[5] = pixel2.y & 1;
  bits[6] = pixel2.z & 1;
  bits[7] = pixel2.w & 1;

  unsigned char byte = 0;
  for(int i = 0; i < 8; ++i) byte |= ((unsigned char)bits[i] << i);

  // 0,1 = byte 0
  // 2,3 = byte 1
  // 4,5 = byte 2
  d_encodedData[idx] = (unsigned char)byte;
}


/**
| 11 11 12 16 ; 11 0  13 0  |
| 15 11 14 6  ; 15 14 19 80 | Encoded image (each set of 4 is 1 pixel)
| 13 14 16 21 ; 14 19 10 17 |
| 10 11 10 10 ; 11 11 10 10 |

=

[ 1100 1010 1100 1010 1001 0101 0100 1100]  Data file

Taking the last bit from each channel
 */
void decode_parallel(const uchar4* const h_encodedImage,
                     unsigned char* h_encodedData,
                     const size_t numRowsSource, const size_t numColsSource)
{


  int numBytes = numRowsSource * numColsSource / 2;
  unsigned char* d_encodedData;
  cudaMalloc(&d_encodedData, (sizeof(unsigned char) * numBytes));

  uchar4* d_encodedImage;
  cudaMalloc(&d_encodedImage, sizeof(uchar4) * numRowsSource * numColsSource);
  cudaMemcpy(d_encodedImage, h_encodedImage, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice);


  int threadsPerBlock = 1024;
  int totalNumThreads = numBytes;
  int numBlocks = ceil((float)totalNumThreads / threadsPerBlock);

  decode_per_byte<<<numBlocks, threadsPerBlock>>>(d_encodedImage, d_encodedData, numBytes);

  cudaMemcpy(h_encodedData, d_encodedData, sizeof(unsigned char) * numBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_encodedData);
}