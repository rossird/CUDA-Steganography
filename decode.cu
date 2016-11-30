#include <fstream>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

using namespace std;

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
                     const uchar4* const h_originalImage,
                     char* const h_binData,
                     int numBytesData,
                     const size_t numRowsSource, const size_t numColsSource)
{

                  
}
