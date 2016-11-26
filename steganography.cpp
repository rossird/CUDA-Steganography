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

using namespace std;

enum ImplementationType {
  SERIAL,
  PARALLEL,
};

/* Checks for valid files. Calls the appropriate encode implementation. */
void encode(string imageFilePath, string dataFilePath,
            string outputFilePath, ImplementationType iType);
            
void encode_parallel(const uchar4* const h_sourceImg,
                     const uchar4* const h_destImg,
                     const char* const h_binData,
                     const size_t numRowsSource, const size_t numColsSource);

/* Checks for valid files. Calls the appropriate decode implementation. */
void decode(string imagFilePath, string encodedImagePath,
            string outputFilePath, ImplementationType iType);


void loadImageRGBA(const std::string &filename,
                   uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols);
                    
/* Print the help menu to instruct users of input parameters. */
void print_help();

int main(int argc, char* argv[]) 
{
  // Parse command line arguments
  if (argc < 2) {
    cout << "Not enough input parameters!\n";
    print_help();
  }
  else
  {
    string input1(argv[1]);
    
    //Help
    if (input1.compare("--help") == 0 ||
        input1.compare("-help") == 0 ||
        input1.compare("help") == 0 ||
        input1.compare("h") == 0) {
      print_help();
      return 0;
    }
  
    //Check number of input arguments
    if(argc < 5) {
      cout << "Not enough input arguments\n";
      print_help();
      return 0;
    }
    
    ImplementationType implementation = PARALLEL; //Default
    
    //Collect input args
    if (argc > 5) {
      string iTypeString(argv[5]);
      if(iTypeString.compare("-parallel") == 0) {
        implementation = PARALLEL;
      } else if(iTypeString.compare("-serial") == 0) {
        implementation = SERIAL;
      } else {
        implementation = PARALLEL;
      }
    }
   
    //Encode or decode
    if (input1.compare("-encode") == 0) {
 
      string imageFilePath(argv[2]);
      string dataFilePath(argv[3]);
      string outputFilePath(argv[4]);
      
      encode(imageFilePath, dataFilePath, outputFilePath, implementation);
      
    } else if(input1.compare("-decode") == 0) {

      string imageFilePath(argv[2]);
      string encodedImagePath(argv[3]);
      string outputFilePath(argv[4]);
      
      decode(imageFilePath, encodedImagePath, outputFilePath, implementation);
      
    } else {
      cout << "Invalid option: " << input1 << endl;
      print_help();
      return 0;
    }
  }
  
  return 0;
}

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
  
  imageFile.close();
  dataFile.close();
  outputFile.close();
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
