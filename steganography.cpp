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
  
  //Encode the data
  if(iType == PARALLEL) {
    encode_parallel(imageData, outputImageData, data, size, numRowsImage, numColsImage);
  } else if(iType == SERIAL) {
    encode_serial(imageData, outputImageData, data, size, numRowsImage, numColsImage);
  }
  
  //Turn uchar4 array into char* array
  saveImageRGBA(outputImageData, numRowsImage, numColsImage, outputFilePath);
  
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

void encode_serial(const uchar4* const h_sourceImg,
                   uchar4* const h_destImg,
                   char* const h_binData,
                   int numBytesData,
                   size_t numRowsSource, size_t numColsSource)
{

  cout << "Running encode serial. NumBytes: " << numBytesData << endl;
  
  //Copy source image into destination image. Then modify destination image contents
  memcpy(h_destImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource);
  
  //Each byte of data is encoded in two bytes of the image
  for(int i = 0; i < numBytesData; i++) {
    char dataByte = h_binData[i];
    
    for(int j = 0; j < 8; j++) {
      int channel = j % 4;
      int pixel = j / 4;
      char mask = (1 << j);
      char bit = dataByte & mask;
      cout << "channel: " << channel << " pixel: " << pixel << "bit: " << bit << endl;
      
      int imgIndex = i + pixel;
      if(channel == 0)
        h_destImg[imgIndex].x += bit;
      else if(channel == 1)
        h_destImg[imgIndex].y += bit;
      else if(channel == 2)
        h_destImg[imgIndex].z += bit;
      else if(channel == 3)
        h_destImg[imgIndex].w += bit;
      
    }

  }

}

/**
 *
 */
void decode(string imagFilePath, string encodedImagePath, string outputFilePath, ImplementationType iType) {
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
