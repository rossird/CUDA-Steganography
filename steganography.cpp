#include <string>
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/timer.h>

#include "loadSaveImage.h"
#include "timer.h"
#include "steganography.h"

using namespace std;

/**
 *
 */
void encode(string imageFilePath, string dataFilePath, 
            string outputFilePath, ImplementationType iType)
{
  //Print some useful information
  cout << "Encoding\n";
  if (iType == PARALLEL) {
    cout << "Using parallel implementation.\n";
  } else if (iType == SERIAL) {
    cout << "Using serial implementation.\n";
  } else {
    cout << "Uknown implementation.\n";
    return;
  }
  
  //Open file stream
  fstream imageFile(imageFilePath.c_str(), fstream::in | fstream::binary);
  fstream dataFile(dataFilePath.c_str(), fstream::in | fstream::binary);
  fstream outputFile(outputFilePath.c_str(), fstream::out | fstream::binary);
  
  //Check for valid files
  if(!imageFile.good()) {
    cout << "Bad image file path " << imageFilePath << endl;
    return;
  }
  if(!dataFile.good()) {
    imageFile.close();
    cout << "Bad data file path " << dataFilePath << endl;
    return;
  }
  if(!outputFile.good()) {
    imageFile.close();
    dataFile.close();
    cout << "Bad output file path " << outputFilePath << endl;
    return;
  }
  
  //Load image file into uchar4 array
  uchar4* imageData;
  size_t numColsImage;
  size_t numRowsImage;
  loadImageRGBA(imageFilePath, &imageData, &numRowsImage, &numColsImage);
  
  //Load data file into char* array
  char* data;
  dataFile.seekg(0, ios::end);
  streampos size = dataFile.tellg();
  data = new char[size];
  dataFile.seekg(0, ios::beg);
  dataFile.read(data, size);
  dataFile.close();
  
  cout << "Datafile: " << dataFilePath << " Size: " << size << endl;
  
  //Create array for output image
  uchar4* outputImageData = new uchar4[numRowsImage * numColsImage];
  
  GpuTimer timer;
  timer.Start();
  //Encode the data
  if(iType == PARALLEL) {
    encode_parallel(imageData, outputImageData, data, size, numRowsImage, numColsImage);
  } else if(iType == SERIAL) {
    encode_serial(imageData, outputImageData, data, size, numRowsImage, numColsImage);
  }
  timer.Stop();
  
  cout << "Elapsed time: " << timer.Elapsed() << endl;
  
  //Turn uchar4 array into char* array
  saveImageRGBA(outputImageData, numRowsImage, numColsImage, outputFilePath);
  //saveImageRGBA(imageData, numRowsImage, numColsImage, outputFilePath);
  
  //Close all the files
  imageFile.close();
  dataFile.close();
  outputFile.close();
  
  //Clean up the memory
  delete[] imageData;
  delete[] data;
  delete[] outputImageData;
  
  return;
}

/*

| 10 11 12 15 ; 11 255 12 0 |
| 15 10 13 5  ; 15 14 19 80 |   Original image
| 12 14 16 21 ; 14 18 10 16 |
| 10 10 10 10 ; 10 10 10 10 |

+

[ 1001 0110 1111 0000 1010 0101 0100 1100]  Data file

= 

| 11 11 12 16 ; 11 0  13 0  |
| 15 11 14 6  ; 15 14 19 80 | Encoded image
| 13 14 16 21 ; 14 19 10 17 |
| 10 11 10 10 ; 11 11 10 10 |
*/
void encode_serial(const uchar4* const h_sourceImg,
                   uchar4* const h_destImg,
                   char* const h_binData,
                   int numBytesData,
                   size_t numRowsSource, size_t numColsSource)
{
  
  //Copy source image into destination image. Then modify destination image contents
  memcpy(h_destImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource);
 
  //Each bit of data is encoded in 1 channel of a uchar4
  //There are 4 channels in a uchar4.
  //So 8 bits would require 2 uchar4, or 2 pixels.
  //1 byte of data to 2 pixels.
  //1 pixel is 4 uchars.
  //1 uchar is 8 bytes.
  //So 1 byte of data to 64 bytes of image.
  for(int i = 0; i < numBytesData; i++) {
    char dataByte = h_binData[i];
    
    for(int j = 0; j < 8 * sizeof(char); j++) {
    
      //Calculate current channel and pixel
      int pixel = j / 4;  //0-3 first pixel, 4-7 second pixel
      int channel = j % 4;
      
      //Get current bit (starting at msb)
      char mask = 1 << (7-j);
      char bit = (dataByte & mask) >> (7-j);
      //cout << "channel: " << channel << " pixel: " << pixel << " bit: " << int(bit);
      
      //2 * current byte index plus current pixel for this byte (0 or 1)
      int imgIndex = 2*i + pixel;
      
      //Defined the relationship between channel number and x,y,z,w as:
      // Channel 0: x
      // Channel 1: y
      // Channel 2: z
      // Channel 3: w
      if(channel == 0) {
        //cout << " old byte: " << int(h_destImg[imgIndex].x);
        h_destImg[imgIndex].x += bit;
        //cout << " new byte: " << int(h_destImg[imgIndex].x) << endl;
      } else if(channel == 1) {
        //cout << " old byte: " << int(h_destImg[imgIndex].y);
        h_destImg[imgIndex].y += bit;
        //cout << " new byte: " << int(h_destImg[imgIndex].y) << endl;
      } else if(channel == 2) {
        //cout << " old byte: " << int(h_destImg[imgIndex].z);
        h_destImg[imgIndex].z += bit;
        //cout << " new byte: " << int(h_destImg[imgIndex].z) << endl;
      } else if(channel == 3) {
        //cout << " old byte: " << int(h_destImg[imgIndex].w);
        h_destImg[imgIndex].w += bit;
        //cout << " new byte: " << int(h_destImg[imgIndex].w) << endl;
      }
    }

  }

}

/**
 *
 */
void decode(string imageFilePath, string encodedImagePath, string outputFilePath, ImplementationType iType) {

  //Print some useful information
  cout << "Decoing\n";
  if (iType == PARALLEL) {
    cout << "Using parallel implementation.\n";
  } else if (iType == SERIAL) {
    cout << "Using serial implementation.\n";
  } else {
    cout << "Uknown implementation.\n";
    return;
  }
  
  //Open file stream
  fstream origImageFile(imageFilePath.c_str(), fstream::in | fstream::binary);
  fstream encodedImageFile(encodedImagePath.c_str(), fstream::in | fstream::binary);
  fstream outputFile(outputFilePath.c_str(), fstream::out | fstream::binary);
  
  //Check for valid files
  if(!origImageFile.good()) {
    cout << "Bad image file path " << imageFilePath << endl;
    return;
  }
  if(!encodedImageFile.good()) {
    origImageFile.close();
    cout << "Bad data file path " << encodedImagePath << endl;
    return;
  }
  if(!outputFile.good()) {
    origImageFile.close();
    encodedImageFile.close();
    cout << "Bad output file path " << outputFilePath << endl;
    return;
  }
  
  char* data;
  
  
  return;
}

/**
 *  Print the help menu to instruct users of input parameters.
 */
void print_help()
{
  cout << "Usage: stega <-encode | -decode> <OPTIONS> ... [-serial | -parallel]\n";

  cout << "To encode:\n";
  cout << "stega -encode <image file> <data file> <output file name> [-serial | -parallel] \n";
  cout << "Default implementation is parallel\n";
  cout << endl;
  cout << "To decode:\n";
  cout << "stega -decode <image file> <encoded image> <output file name> [-serial | -parallel] \n";
  cout << "Default implementation is parallel\n";
  return;
}
