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

  bool bits[8];

  bits[0] = d_encodedImg[curr_pixel].x & 1;
  bits[1] = d_encodedImg[curr_pixel].y & 1;
  bits[2] = d_encodedImg[curr_pixel].z & 1;
  bits[3] = d_encodedImg[curr_pixel].w & 1;
  bits[4] = d_encodedImg[curr_pixel + 1].x & 1;
  bits[5] = d_encodedImg[curr_pixel + 1].y & 1;
  bits[6] = d_encodedImg[curr_pixel + 1].z & 1;
  bits[7] = d_encodedImg[curr_pixel + 1].w & 1;

  unsigned char byte = 0;
  for(int i = 0; i < 8; ++i) byte |= ((unsigned char)bits[i] << i);

  // 0,1 = 0
  // 2,3 = 1
  // 4,5 = 2
  // To figure out what byte we're writing to
  d_encodedData[idx] = (unsigned char)byte;
}


/**
| 11 11 12 16 ; 11 0  13 0  |
| 15 11 14 6  ; 15 14 19 80 | Encoded image (each set of 4 is 1 pixel)
| 13 14 16 21 ; 14 19 10 17 |
| 10 11 10 10 ; 11 11 10 10 |

-

| 10 11 12 15 ; 11 255 12 0 |
| 15 10 13 5  ; 15 14 19 80 | Original image 
| 12 14 16 21 ; 14 18 10 16 |
| 10 10 10 10 ; 10 10 10 10 |

=

[ 1001 0110 1111 0000 1010 0101 0100 1100]  Data file


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

  decode_per_byte<<<numBlocks, threadsPerBlock>>>(d_encodedImage, d_encodedData);

  cudaMemcpy(h_encodedData, d_encodedData, sizeof(unsigned char) * numBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_encodedData);
}