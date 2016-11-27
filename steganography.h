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
void decode(std::string imagFilePath, std::string encodedImagePath,
            std::string outputFilePath, ImplementationType iType);


void loadImageRGBA(const std::string &filename,
                   uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols);
                    
/* Print the help menu to instruct users of input parameters. */
void print_help();

#endif
