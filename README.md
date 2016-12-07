# CUDA-Steganography
Parallel Computing Final project
Richard Rossi, Tejas Deshpande


## How to use
Make sure you have CUDA and OpenCV installed. We've tested this on unix, and cannot confirm
if this'll work on Windows.

1. Clone this repo
2. Load the cuda module: 'module load cuda'
3. Run `sh build.sh` in the root of the folder
4. The executable will be available in `{REPO}/bin`
5. To encode, use it as `./stego -encode image_path data_path output_image [-serial | -parallel (default)]`
6. To decode use it as `./stego -decode steg_image_path output_file_path [-serial | -parallel(default)]`
