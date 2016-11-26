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
  uchar4** imageData;
  size_t numColsImage;
  size_t numRowsImage;
  loadImageRGBA(imageFilePath, imageData, &numRowsImage, &numColsImage);
  
  //Load data file into char* array
  char* data;
  streampos size = dataFile.tellg();
  data = new char[size];
  dataFile.seekg(0, ios::beg);
  dataFile.read(data, size);
  dataFile.close();
  
  //Encode the data
  //encode();
  
  //Close all the files
  imageFile.close();
  dataFile.close();
  outputFile.close();
  
  //Clean up the memory
  delete[] *imageData;
  delete[] data;
  
  return;
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
