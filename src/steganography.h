#ifndef STEGANOGRAPHY_H
#define STEGANOGRAPHY_H
#include <string>
#include <cuda_runtime.h>

enum ImplementationType {
  SERIAL,
  PARALLEL,
};

/* Checks for valid files. Calls the appropriate encode implementation. */
void encode(std::string imageFilePath, std::string dataFilePath,
            std::string outputFilePath, ImplementationType iType);
            
void encode_parallel(const uchar4* const h_sourceImg,
                     uchar4* const h_destImg,
                     const char* const h_binData,
                     int numBytesData,
                     size_t numRowsSource, size_t numColsSource);
                     
void encode_serial(const uchar4* const h_sourceImg,
                   uchar4* const h_destImg,
                   char* const h_binData,
                   int numBytesData,
                   size_t numRowsSource, size_t numColsSource);

/* Checks for valid files. Calls the appropriate decode implementation. */
void decode(std::string encodedImagePath, std::string outputFilePath,
            ImplementationType iType);


void decode_serial(const uchar4* const h_encodedImg,
                   unsigned char* h_encodedData,
                   size_t numRowsSource, size_t numColsSource);

void decode_parallel(const uchar4* const h_encodedImage,
                     unsigned char* h_encodedData,
                     const size_t numRowsSource, const size_t numColsSource);
                    
/* Print the help menu to instruct users of input parameters. */
void print_help();

#endif
