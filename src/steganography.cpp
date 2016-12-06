#include <string>
#include<bitset>
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
  streampos dataFileSize = dataFile.tellg();
  data = new char[dataFileSize];
  dataFile.seekg(0, ios::beg);
  dataFile.read(data, dataFileSize);
  dataFile.close();
  
  //Check if file sizes work
  //Data file size * 8 must be <= size of image
  int imageFileSize = numRowsImage * numColsImage * 4;
  if(dataFileSize * 8 > imageFileSize) {
    cout << "Data file is too large for the input image.\n";
    cout << "Data file must be less than or equal 1/8 of image size\n";
    cout << "Data file size: " << dataFileSize << endl;
    cout << "Image file size: " << imageFileSize << endl;
    
    delete[] imageData;
    delete[] data;
    
    return;
  } 
  
  //Create array for output image
  uchar4* outputImageData = new uchar4[numRowsImage * numColsImage];
  
  GpuTimer timer;
  timer.Start();
  //Encode the data
  if(iType == PARALLEL) {
    encode_parallel(imageData, outputImageData, data, dataFileSize, numRowsImage, numColsImage);
  } else if(iType == SERIAL) {
    encode_serial(imageData, outputImageData, data, dataFileSize, numRowsImage, numColsImage);
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
void encode_serial(const uchar4* const h_sourceImg,
                   uchar4* const h_destImg,
                   char* const h_binData,
                   int numBytesData,
                   size_t numRowsSource, size_t numColsSource)
{
  
  //Copy source image into destination image. Then modify destination image contents
  memcpy(h_destImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource);
 
  //Each bit of data is encoded in 1 channel of a uchar4
  //So 1 bit corresponds to 1 byte
  for(int i = 0; i < numBytesData; i++) {
    char dataByte = h_binData[i];
    
    for(int j = 0; j < 8 * sizeof(char); j++) {
    
      //Calculate current channel and pixel
      int pixel = j / 4;  //0-3 first pixel, 4-7 second pixel
      int channel = j % 4;
      
      //Get the bit (start at MSB)
      //Offset should be 7 for channel 0, nibble 0 (pixel 0)
      //Offset should be 0 for channel 3, nibble 1 (pixel 1)
      int offset = (7 - 4 * pixel) - channel;
      bool bit = (dataByte >> offset) & 1;

      cout << "Bit is " << bit << endl;
      
      //2 * current byte index plus current pixel for this byte (0 or 1)
      int imgIndex = 2*i + pixel;
      
      //Defined the relationship between channel number and x,y,z,w as:
      // Channel 0: x
      // Channel 1: y
      // Channel 2: z
      // Channel 3: w
      if(channel == 0) {
        h_destImg[imgIndex].x = h_destImg[imgIndex].x & ~1 | bit;
      } else if(channel == 1) {
        h_destImg[imgIndex].y = h_destImg[imgIndex].y & ~1 | bit;
      } else if(channel == 2) {
        h_destImg[imgIndex].z = h_destImg[imgIndex].z & ~1 | bit;
      } else if(channel == 3) {
        h_destImg[imgIndex].w = h_destImg[imgIndex].w & ~1 | bit;
      }
    }

  }

}

/**
 *
 */
void decode(string encodedImagePath, string outputFilePath, ImplementationType iType) {

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
  fstream encodedImageFile(encodedImagePath.c_str(), fstream::in | fstream::binary);
  fstream outputFile(outputFilePath.c_str(), fstream::out);
  
  //Check for valid files
  if(!encodedImageFile.good()) {
    cout << "Bad data file path " << encodedImagePath << endl;
    return;
  }
  if(!outputFile.good()) {
    encodedImageFile.close();
    cout << "Bad output file path " << outputFilePath << endl;
    return;
  }
  
  uchar4* encodedImage;
  size_t numColsImage;
  size_t numRowsImage;
  loadImageRGBA(encodedImagePath, &encodedImage, &numRowsImage, &numColsImage);
  
  unsigned long long numPixels = numColsImage * numRowsImage;
  unsigned long long numBits = numPixels/4;
  unsigned long long numBytes = 1;
  unsigned char* encodedData = new unsigned char[numBytes];
  GpuTimer timer;
  timer.Start();

  //Extract the encoded data
  if (iType == PARALLEL) {
    //decode parallel
    cout << "Parallel Implementation not yet implmented" << endl; 
  } else if (iType == SERIAL) {
    //decode serial
    decode_serial(encodedImage, encodedData, numRowsImage, numColsImage);
  }

  timer.Stop();
  
  cout << "Elapsed time: " << timer.Elapsed() << endl;

  // save data file
  outputFile.write((char *)encodedData, numBytes);

  // Close the file streams
  encodedImageFile.close();
  outputFile.close();

  // clean up memory
  delete[] encodedImage;
  delete[] encodedData;
  
  return;
}

/**
* Decode serial
*/
void decode_serial(const uchar4* const h_encodedImg,
                   unsigned char* h_encodedData,
                   size_t numRowsSource, size_t numColsSource)
{
  
  unsigned long long numPixels = numRowsSource*numColsSource;

  unsigned long long numBits = numPixels/4;
  unsigned long long numBytes = numBits/8;
  // We're jumping 2 pixels at a time to gather a byte of data
  // If we can't find a full byte at the end, we will drop the incomplete byte
  // as this is certainly not part of the original data
  for (unsigned long long curr_pixel = 0; curr_pixel < 2; curr_pixel += 2) {
    if (curr_pixel + 1 >= numPixels) {
      // If we don't have 8 bits, break
      break;
    }

    bitset<8> x;
    // bits[0] = h_encodedImg[curr_pixel].x & 1;
    // bits[1] = h_encodedImg[curr_pixel].y & 1;
    // bits[2] = h_encodedImg[curr_pixel].z & 1;
    // bits[3] = h_encodedImg[curr_pixel].w & 1;
    // bits[4] = h_encodedImg[curr_pixel + 1].x & 1;
    // bits[5] = h_encodedImg[curr_pixel + 1].y & 1;
    // bits[6] = h_encodedImg[curr_pixel + 1].z & 1;
    // bits[7] = h_encodedImg[curr_pixel + 1].w & 1;
    x.set(0, (h_encodedImg[curr_pixel].x & 1));
    x.set(1, (h_encodedImg[curr_pixel].y & 1));
    x.set(2, (h_encodedImg[curr_pixel].z & 1));
    x.set(3, (h_encodedImg[curr_pixel].w & 1));
    x.set(4, (h_encodedImg[curr_pixel + 1].x & 1));
    x.set(5, (h_encodedImg[curr_pixel + 1].y & 1));
    x.set(6, (h_encodedImg[curr_pixel + 1].z & 1));
    x.set(7, (h_encodedImg[curr_pixel + 1].w & 1));

    // debug
    x.set(0, 0);
    x.set(1, 1);
    x.set(2, 1);
    x.set(3, 0);
    x.set(4, 0);
    x.set(5, 0);
    x.set(6, 0);
    x.set(7, 1);

    cout << "----Printing individual channels----" << endl;
    cout << h_encodedImg[curr_pixel].x << endl;
    cout << h_encodedImg[curr_pixel].y << endl;
    cout << h_encodedImg[curr_pixel].z << endl;
    cout << h_encodedImg[curr_pixel].w << endl;
    cout << h_encodedImg[curr_pixel+1].x << endl;
    cout << h_encodedImg[curr_pixel+1].y << endl;
    cout << h_encodedImg[curr_pixel+1].z << endl;
    cout << h_encodedImg[curr_pixel+1].w << endl;
    cout << "----Printing individual channels ended----" << endl;

    cout << "----Printing individual bits----" << endl;
    for (int i = 0; i < 8; ++i) {
      cout << x.get(i) << endl;
    }
    cout << "----Printing individual bits ended----" << endl;
    


    cout << "Resulting char is " << ((unsigned char)x.to_ulong()) << endl;

    // 0,1 = 0
    // 2,3 = 1
    // 4,5 = 2
    // To figure out what byte we're writing to
    h_encodedData[curr_pixel/2] = ((unsigned char)x.to_ulong()); 
  }
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
  cout << "stega -decode <encoded image> <output file name> [-serial | -parallel] \n";
  cout << "Default implementation is parallel\n";
  return;
}