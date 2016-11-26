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
