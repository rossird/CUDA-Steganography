#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

using namespace std;

//Execute 1 thread per pixel of output image.
//Each thread handles all four channels of the output pixels
__global__ void encode_per_pixel_kernel(uchar4* const d_destImg,
                                        const char* const d_binData,
                                        int numBytesData)
{
  //Get pixel index
  //Theres two pixels per byte of data
  //Thread 2 would be pixel 2 and working on byte 1 nibble 0
  //Thread 3 would be pixel 3 and working on byte 1 nibble 1
  //Thread 4 would be pixel 4 and working on byte 2 nibble 0
  //Thread 5 would be pixel 5 and working on byte 2 nibble 1
  int pixel = threadIdx.x + blockDim.x * blockIdx.x;
  if(pixel >= 2 * numBytesData)
    return;
  
  //Calculate which nibble (0 or 1) in the byte
  //and which byte (0 to numBytesData)
  int byteIndex = pixel / 2;
  int nibble = pixel % 2;
  char dataByte = d_binData[byteIndex];
 
  //Let's work with a local copy. We only need two global accesses this way.
  uchar4 outputPixel = d_destImg[pixel];
  
  //Channel 0 (first bit in the nibble)
  int offset = (7 - 4 * nibble);
  bool bit = (dataByte >> offset) & 1;
  outputPixel.x = outputPixel.x & ~1 | bit;
  
  //Channel 1 (2nd bit)
  offset -= 1;
  bit = (dataByte >> offset) & 1;
  outputPixel.y = outputPixel.y & ~1 | bit;
  
  //Channel 2 (3rd bit)
  offset -= 1;
  bit = (dataByte >> offset) & 1;
  outputPixel.z = outputPixel.z & ~1 | bit;
  
  //Channel 3 (4th bit) This is the alpha channel
  offset -= 1;
  bit = (dataByte >> offset) & 1;
  outputPixel.w = outputPixel.w & ~1 | bit;
  
  d_destImg[pixel] = outputPixel;
  
}


//1 channel per bit of data
//8 channels per byte of data
//This calls requires two global memory accesses
__global__ void encode_per_channel_kernel(uchar4* const d_destImg,
                              const char* const d_binData,
                              int numBytesData)
{
  //1 thread per bit of data
  //Thread 0 works on pixel 0 channel 0 byte 0 nibble 0 bit 0
  //Thread 1 works on pixel 0 channel 1 byte 0 nibble 0 bit 1
  //Thread 2 works on pixel 0 channel 2 byte 0 nibble 0 bit 2
  //Thread 3 works on pixel 0 channel 3 byte 0 nibble 0 bit 3
  //Thread 4 works on pixel 1 channel 0 byte 0 nubble 1 bit 0
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx >= 8 * numBytesData)
    return;
    
  //Calculate channel (0-4) and pixel (0 - 2*numBytes - 1)
  int channel = idx % 4;
  int pixel = idx / 4;
  
  //Calculate which nibble (0 or 1) in the byte
  //and which byte (0 to numBytesData - 1)
  int byteIndex = pixel / 2;
  int nibble = pixel % 2;
  char dataByte = d_binData[byteIndex];
  
  //Let's work with a local copy. 
  uchar4 outputPixel = d_destImg[pixel];
  
  //Get the bit
  //Offset should be 7 for channel 0, nibble 0
  //Offset should be 0 for channel 3, nibble 1
  int offset = (7 - 4 * nibble) - channel;
  bool bit = (dataByte >> offset) & 1;
  
  if(channel == 0) {
    outputPixel.x = outputPixel.x & ~1 | bit;
  } else if(channel == 1){ 
    outputPixel.y = outputPixel.y & ~1 | bit;
  } else if(channel == 2){
    outputPixel.z = outputPixel.z & ~1 | bit;
  } else if(channel == 3){
    outputPixel.w = outputPixel.w & ~1 | bit;
  }
  
  d_destImg[pixel] = outputPixel;
 
}

/**

    | 10 11 12 15 ; 11 255 12 0 |
    | 15 10 13 5  ; 15 14 19 80 | Original image (each set of 4 is 1 pixel).
    | 12 14 16 21 ; 14 18 10 16 |
    | 11 11 11 11 ; 10 10 10 10 |

    and

    [ 1001 0110 1111 0000 1010 0101 0100 1100]  Data file

    = 

    | 11 10 12 15 ; 10 255 13 0  |
    | 15 11 13 5  ; 14 14  18 80 | Encoded image
    | 13 14 17 20 ; 14 19  10 17 |
    | 11 10 11 11 ; 11 11  10 10 |
    
    To encode the data, we will use the least significant bit approach by
    modifying the LSB of each channel of each pixel of th input image. The
    LSB will match the corresponding bit of the input data. The data can be
    decoded by reading the LSB from the encoded image.
    
    For example, if the channel byte is 0001 1001 (value of 25) and we want to
    encode a 1, the byte would remain the same. If we want to encode a 0, the
    byte would become 0001 1000 (value of 24).
    
    If the channel byte is 0010 1110 (value of 46), and we want to encode a 1,
    then the byte would become 0010 1111 (value of 47). If we want to encode a
    0, then the byte would remain the same.
    
 */
void encode_parallel(const uchar4* const h_sourceImg,
                     uchar4* const h_destImg,
                     const char* const h_binData,
                     int numBytesData,
                     const size_t numRowsSource, const size_t numColsSource)
{

  //Allocate device memory
  uchar4* d_destImg;
  char* d_binData;
  cudaMalloc(&d_destImg, sizeof(uchar4) * numRowsSource * numColsSource);
  cudaMalloc(&d_binData, sizeof(char) * numBytesData);
  
  cudaMemcpy(d_destImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_binData, h_binData, numBytesData, cudaMemcpyHostToDevice);

  //Each thread handles 1 pixel
  //This means 1 thread per 4 bits of data (2 threads per byte)
  //int numThreads = numBytesData * 2.0;
  //int threadsPerBlock = 1024;
  //int numBlocks = ceil((float)numThreads / threadsPerBlock);
  
  //encode_per_pixel_kernel<<<numBlocks, threadsPerBlock>>>(d_destImg, d_binData, numBytesData);
  
  //Each thread handles 1 channel of 1 pixel
  //This means 1 thread per bit of data (8 threads per byte)
  int numThreads = numBytesData * 8;
  int threadsPerBlock = 1024;
  int numBlocks = ceil((float)numThreads / threadsPerBlock);
  
  encode_per_channel_kernel<<<numBlocks, threadsPerBlock>>>(d_destImg, d_binData, numBytesData);
  
  cudaMemcpy(h_destImg, d_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyDeviceToHost);
  
  //Free memory
  cudaFree(d_destImg);
  cudaFree(d_binData);
                  
}
