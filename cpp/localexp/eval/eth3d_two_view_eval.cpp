// Copyright 2017 Silvano Galliani, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//******************
//added by CCJ.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>
#include <string>
//added by CCJ.
using namespace boost::python;
using namespace std;
//******************
//******************

#define Is_Display 0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <boost/filesystem.hpp>
#define PNG_SKIP_SETJMP_CHECK
#include <libpng/png.h>

// Return codes of the program:
// 0: Success.
// 1: System failure (e.g., due to wrong parameters given).
// 2: Reconstruction file input failure (PFM file cannot be found or read).
enum class ReturnCodes {
  kSuccess = 0,
  kSystemFailure = 1,
  kReconstructionFileInputFailure = 2
};

// Evaluation result for a pixel for the "bad pixel" metrics:
// 0: The pixel is accurate within the metric's evaluation threshold.
// 1: The pixel is not accurate, but masked out by the non-occluded mask.
// 2: The pixel is not accurate and not masked out.
// 3: There is no ground truth for the pixel.
enum class BadPixelState {
  kGood = 0,
  kBadButMasked = 1,
  kBad = 2,
  kNoGroundTruth = 3
};

struct Metrics {
  double coverage;
  double bad_0_5;
  double bad_1_0;
  double bad_2_0;
  double bad_4_0;
  double avgerr;
  double rms;
  double a50;
  double a90;
  double a95;
  double a99;
	long int valid_pix_num;
};

float ConvertEndianness(float big_float) {
  uint8_t* big_float_char_ptr = reinterpret_cast<uint8_t*>(&big_float);
  std::swap(big_float_char_ptr[0], big_float_char_ptr[3]);
  std::swap(big_float_char_ptr[1], big_float_char_ptr[2]);
  return big_float;
}

bool ReadPFMFile(const char* pfm_filename, int* width, int* height,
                 std::vector<float>* buffer, std::string* error_string) {
  FILE* pfm_file = fopen(pfm_filename, "rb");
  if (!pfm_file) {
    *error_string = "File cannot be opened.";
    return false;
  }
  
  const int kRowBufferSize = 4096;
  char row[kRowBufferSize];
  
  // Accept only grayscale PFM files with header Pf (not PF).
  if (std::fgets(row, kRowBufferSize, pfm_file) == nullptr) {
    *error_string = "Cannot read start of file content (1).";
    return false;
  }
  if (row[0] != 'P' || row[1] != 'f') {
    *error_string = "Format code is not Pf.";
    return false;
  }
  
  // Skip possible comments.
  if (std::fgets(row, kRowBufferSize, pfm_file) == nullptr) {
    *error_string = "Cannot read start of file content (2).";
    return false;
  }
  while(row[0] == '#' || row[0] == '\n') {
    if (std::fgets(row, kRowBufferSize, pfm_file) == nullptr) {
      *error_string = "Cannot read start of file content (3).";
      return false;
    }
  }
  
  // Read image size.
  if (sscanf(row, "%d %d", width, height) != 2) {
    *error_string = "Cannot read width and height.";
    return false;
  }

  // Read scale.
  if (std::fgets(row, kRowBufferSize, pfm_file) == nullptr) {
    *error_string = "Cannot read start of file content (4).";
    return false;
  }
  float scale;
  if (sscanf(row, "%f", &scale) != 1) {
    *error_string = "Cannot read scale.";
    return false;
  }
  
  // Read data.
  buffer->resize((*width) * (*height));
  if (fread(buffer->data(), sizeof(float), (*width) * (*height), pfm_file) !=
      (*width) * (*height)) {
    *error_string = "Cannot read image content.";
    return false;
  }
  
  // If scale is positive, convert to little endian.
  if (scale > 0) {
    for (int i = 0; i < (*width) * (*height); ++ i) {
      buffer->at(i) = ConvertEndianness(buffer->at(i));
    }
  }
  
  // Since the images are expected to start with the topmost row (and not with
  // the bottommost row as in the PFM format), mirror the image vertically.
  for (int y = 0; y < (*height) / 2; ++ y) {
    for (int x = 0; x < (*width); ++ x) {
      std::swap(buffer->at(x + (*width) * y),
                buffer->at(x + (*width) * ((*height) - 1 - y)));
    }
  }
  
  fclose(pfm_file);
  return true;
}

bool ReadMaskPNGFile(const char* path, std::vector<uint8_t>* buffer) {
  // Open the file.
  FILE* file = fopen(path, "rb");
  if (!file) {
    return false;
  }
  
  // Check the first bytes of the file against the PNG header signature to
  // verify that the file is a PNG file. 8 bytes is the maximum according to the
  // libpng documentation.
  const int kBytesToCheck = 8;
  uint8_t header[8];
  if (fread(header, 1, kBytesToCheck, file) != kBytesToCheck) {
    return false;
  }
  bool is_png = !png_sig_cmp(header, 0, kBytesToCheck);
  if (!is_png) {
    return false;
  }
  
  // Create PNG read and info structs.
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    return false;
  }
  
  // Set the error handler.
  if (setjmp(png_jmpbuf(png_ptr))) {
    // This is executed if the error handler is triggered.
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(file);
    return false;
  }
  
  // Initialize I/O.
  png_init_io(png_ptr, file);
  png_set_sig_bytes(png_ptr, kBytesToCheck);
  
  // Read info.
  png_read_info(png_ptr, info_ptr);
  uint32_t width = png_get_image_width(png_ptr, info_ptr);
  uint32_t height = png_get_image_height(png_ptr, info_ptr);
  uint8_t bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  uint8_t color_type = png_get_color_type(png_ptr, info_ptr);
  uint8_t channel_count = png_get_channels(png_ptr, info_ptr);
  
  // Let libpng convert paletted data to RGB.
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png_ptr);
  }

  // Let libpng convert images with a transparent color to use an alpha channel.
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png_ptr);
  }

  // Let libpng expand grayscale images with less than 8 bits per pixel.
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  }
  
  // Let libpng scale 16-bit images down to 8 bit.
  if (bit_depth == 16) {
#if PNG_LIBPNG_VER >= 10504
       png_set_scale_16(png_ptr);
#else
       png_set_strip_16(png_ptr);
#endif
  }
  
  // Let libpng remove an alpha channel if necessary, and convert to grayscale
  // if necessary.
  if (channel_count == 2 && color_type & PNG_COLOR_MASK_ALPHA) {
    png_set_strip_alpha(png_ptr);
  } else if (channel_count == 4 && color_type & PNG_COLOR_MASK_ALPHA) {
    png_set_strip_alpha(png_ptr);
    // error_action == 1: No warning if the image is not actually grayscale.
    // Negative weights result in default weights being used.
    png_set_rgb_to_gray(png_ptr, /*error_action*/ 1, -1, -1);
  } else if (channel_count == 3) {
    // error_action == 1: No warning if the image is not actually grayscale.
    // Negative weights result in default weights being used.
    png_set_rgb_to_gray(png_ptr, /*error_action*/ 1, -1, -1);
  } else if (channel_count == 1) {
    // Ok.
  } else {
    return false;
  }
  
  // Update the info struct with the transforms set above.
  png_read_update_info(png_ptr, info_ptr);
  
  // Read the image.
  buffer->resize(width * height);
  png_bytep* row_pointers = new png_bytep[height];
  for (uint32_t y = 0; y < height; ++ y) {
    row_pointers[y] = reinterpret_cast<png_bytep>(buffer->data() + y * width);
  }
  png_read_image(png_ptr, row_pointers);
  delete[] row_pointers;
  
  // Clean up.
  // NOTE: png_read_end is unnecessary as long as we don't intend to
  // read anything after the PNG data.
  // png_read_end(png_ptr, nullptr);
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  fclose(file);
  return true;
}

bool WritePNGFile(const char* path, uint32_t width, uint32_t height,
                  bool colored, const std::vector<uint8_t>& buffer) {
  
	const int output_bit_depth = 8;
  const int output_channels = colored ? 3 : 1;
  
  // Open the file.
  FILE* file = fopen(path, "wb");
  if (!file) {
    return false;
  }
  
  // Create PNG write and info structs.
  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    return false;
  }
  
  // Set the error handler.
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(file);
    return false;
  }
  
  // Initialize I/O.
  png_init_io(png_ptr, file);
  
  // Write info.
  int color_type;
  if (output_channels == 1) {
    color_type = PNG_COLOR_TYPE_GRAY;
  } else if (output_channels == 3) {
    color_type = PNG_COLOR_TYPE_RGB;
  } else if (output_channels == 4) {
    color_type = PNG_COLOR_TYPE_RGB_ALPHA;
  } else {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(file);
    return false;
  }
  png_set_IHDR(
    png_ptr,
    info_ptr,
    width,
    height,
    output_bit_depth,
    color_type,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png_ptr, info_ptr);
  
  // Write image.
  const png_byte** row_pointers = new const png_byte*[height];
  for (uint32_t y = 0; y < height; ++ y) {
    row_pointers[y] =
        reinterpret_cast<const png_byte*>(
            buffer.data() + output_channels * width * y);
  }
  png_write_image(png_ptr, const_cast<png_byte**>(row_pointers));
  delete[] row_pointers;
  
  png_write_end(png_ptr, nullptr);
  
  // Clean up.
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(file);
  return true;
}


void ComputeMetrics(bool limit_to_nocc,
                    int image_width,
                    int image_height,
                    const std::vector<float>& reconstruction,
                    const std::vector<float>& ground_truth,
                    const std::vector<uint8_t>& mask,
                    Metrics* metrics,
                    std::vector<BadPixelState>* bad_pixels,
                    std::vector<float>* errors) {
  
	std::vector<float> abs_errors;
  abs_errors.reserve(image_width * image_height);// reserve a chunk of memroy;
  
	if (bad_pixels) {
    for (int i = 0; i < 4; ++ i) {
      bad_pixels[i].clear();
			// kGood = 0 : The pixel is accurate within the metric's evaluation threshold.
      bad_pixels[i].resize(reconstruction.size(), BadPixelState::kGood);
    }
  }
  
  if (errors) {
    errors->clear();
    errors->resize(reconstruction.size(), 0.f);
  }
  
  std::size_t bad_0_5_count = 0;
  std::size_t bad_1_0_count = 0;
  std::size_t bad_2_0_count = 0;
  std::size_t bad_4_0_count = 0;
  
  double abs_error_sum = 0;
  double squared_error_sum = 0;
  
  std::size_t num_ground_truth_pixels = 0;
  
  for (std::size_t i = 0; i < reconstruction.size(); ++ i) {
    float reconstruction_disp = reconstruction[i];
    float ground_truth_disp = ground_truth[i];
    /*
		 * The `occlusion mask` for the left image is given as a file "mask0nocc.png". 
		 *   - Pixels without ground truth have the color (0, 0, 0). 
		 *   - Pixels which are only observed by the left image have the color (128, 128, 128). 
		 *   - Pixels which are observed by both images have the color (255, 255, 255). 
		 *   - For the "non-occluded" evaluation, the evaluation is limited to the pixels observed by both images.
		 *
		 * */
    if (!std::isfinite(ground_truth_disp) // if infinite; 
				|| (limit_to_nocc && mask[i] != 255)) { // limit_to_nocc, means only focusing on the non-occluded region.
      if (bad_pixels) {
        for (int k = 0; k < 4; ++ k) {
          // kNoGroundTruth = 3: There is no ground truth for the pixel.
          bad_pixels[k][i] = BadPixelState::kNoGroundTruth;
        }
      }
      continue;
    }
    ++ num_ground_truth_pixels;
    
    if (std::isnan(reconstruction_disp) ||
        !std::isfinite(reconstruction_disp) ||
        reconstruction_disp > image_width ||
        reconstruction_disp < -image_width) {
      continue;
    }
    
    const float error = reconstruction_disp - ground_truth_disp;
    if (errors) {
      errors->at(i) = error;
    }
    const float abs_error = fabsf(error);
    
    abs_errors.push_back(abs_error);
    abs_error_sum += abs_error;
    squared_error_sum += error * error;
    
    if (abs_error > 0.5f) {
      ++ bad_0_5_count;
      if (bad_pixels) {
        // kBadButMasked = 1, the pixel is not accurate, but masked out by the non-occluded mask.
        // kBad = 2: The pixel is not accurate and not masked out.
        bad_pixels[0][i] = (mask[i] != 255) ? BadPixelState::kBadButMasked :
                                              BadPixelState::kBad;
      }
      
      if (abs_error > 1.0f) {
        ++ bad_1_0_count;
        if (bad_pixels) {
          bad_pixels[1][i] = (mask[i] != 255) ? BadPixelState::kBadButMasked :
                                                BadPixelState::kBad;
        }
        
        if (abs_error > 2.0f) {
          ++ bad_2_0_count;
          if (bad_pixels) {
            bad_pixels[2][i] = (mask[i] != 255) ? BadPixelState::kBadButMasked :
                                                  BadPixelState::kBad;
          }
          
          if (abs_error > 4.0f) {
            ++ bad_4_0_count;
            if (bad_pixels) {
              bad_pixels[3][i] =
                  (mask[i] != 255) ? BadPixelState::kBadButMasked :
                                     BadPixelState::kBad;
            }
          }
        }
      }
    }
  }
  
  std::sort(abs_errors.begin(), abs_errors.end());
  std::size_t num_estimated_pixels = abs_errors.size();
	//printf("imgW * imgH = %d * %d = %d, ", image_width, image_height, image_width*image_height);
	//printf("num_estimated_pixels = %d\n", (int)num_estimated_pixels);
  
  if (num_estimated_pixels == num_ground_truth_pixels) {
    // Make sure that floating point arithmetic issues cannot happen and the
    // coverage for methods that estimate all pixels is always exactly equal to
    // one.
    metrics->coverage = 1;
  } 
	else {
    metrics->coverage =
        (1.0 * num_estimated_pixels) / (1.0 * num_ground_truth_pixels);
  }
  
  if (num_estimated_pixels > 0) {
    metrics->bad_0_5 = bad_0_5_count / (1.0 * abs_errors.size());
    metrics->bad_1_0 = bad_1_0_count / (1.0 * abs_errors.size());
    metrics->bad_2_0 = bad_2_0_count / (1.0 * abs_errors.size());
    metrics->bad_4_0 = bad_4_0_count / (1.0 * abs_errors.size());
    
    metrics->avgerr = abs_error_sum / abs_errors.size();
    metrics->rms = sqrtf(squared_error_sum / abs_errors.size());
    
    metrics->a50 = abs_errors[floor(abs_errors.size() * 0.5f)];
    metrics->a90 = abs_errors[floor(abs_errors.size() * 0.9f)];
    metrics->a95 = abs_errors[floor(abs_errors.size() * 0.95f)];
    metrics->a99 = abs_errors[floor(abs_errors.size() * 0.99f)];
		metrics->valid_pix_num = num_estimated_pixels;
  } 
	else {
    // No pixels have a depth estimate. Return the worst possible result.
    metrics->bad_0_5 = 1;
    metrics->bad_1_0 = 1;
    metrics->bad_2_0 = 1;
    metrics->bad_4_0 = 1;
    
    metrics->avgerr = std::numeric_limits<float>::infinity();
    metrics->rms = std::numeric_limits<float>::infinity();
    
    metrics->a50 = std::numeric_limits<float>::infinity();
    metrics->a90 = std::numeric_limits<float>::infinity();
    metrics->a95 = std::numeric_limits<float>::infinity();
    metrics->a99 = std::numeric_limits<float>::infinity();
		metrics->valid_pix_num = num_estimated_pixels;
  }
}

void OutputMetrics(const Metrics& metrics) {
  std::cout << "coverage: " << metrics.coverage << std::endl;
  
  std::cout << "bad_0_5: " << metrics.bad_0_5 << std::endl;
  std::cout << "bad_1_0: " << metrics.bad_1_0 << std::endl;
  std::cout << "bad_2_0: " << metrics.bad_2_0 << std::endl;
  std::cout << "bad_4_0: " << metrics.bad_4_0 << std::endl;
  
  std::cout << "avgerr: " << metrics.avgerr << std::endl;
  std::cout << "rms: " << metrics.rms << std::endl;
  
  std::cout << "a50: " << metrics.a50 << std::endl;
  std::cout << "a90: " << metrics.a90 << std::endl;
  std::cout << "a95: " << metrics.a95 << std::endl;
  std::cout << "a99: " << metrics.a99 << std::endl;
	std::cout << "valid_pix_num: " << metrics.valid_pix_num << std::endl;
}

void GetMinMaxDisparity(
    const std::vector<float>& ground_truth,
    float* min_disparity,
    float* max_disparity) {
  *min_disparity = std::numeric_limits<float>::infinity();
  *max_disparity = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < ground_truth.size(); ++ i) {
    float disp = ground_truth[i];
    if (!std::isfinite(disp)) {
      continue;
    }
    
    if (disp < *min_disparity) {
      *min_disparity = disp;
    }
    if (disp > *max_disparity) {
      *max_disparity = disp;
    }
  }
}

void CreateReconstructionVisualization(
    int width,
    int height,
    const std::vector<float>& reconstruction,
    float min_disparity,
    float max_disparity,
    //const char* visualizations_path
    const string & visualizations_path
		) {
  // Create color-coded reconstruction disparity visualization.
  // This uses Andreas Geiger's color map from the Kitti benchmark:
  float map[8][4] = {{0, 0, 0, 114}, {0, 0, 1, 185}, {1, 0, 0, 114},
                     {1, 0, 1, 174}, {0, 1, 0, 114}, {0, 1, 1, 185},
                     {1, 1, 0, 114}, {1, 1, 1, 0}};
  float sum = 0;
  for (int32_t i = 0; i < 8; ++ i) {
    sum += map[i][3];
  }
  
  float weights[8]; // relative weights
  float cumsum[8];  // cumulative weights
  cumsum[0] = 0;
  for (int32_t i = 0; i < 7; ++ i) {
    weights[i] = sum / map[i][3];
    cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
  }
  
  std::vector<uint8_t> colors(3 * reconstruction.size());
  for (std::size_t i = 0; i < reconstruction.size(); ++ i) {
    float reconstruction_disp = reconstruction[i];
    float value =
        std::max(0.f,
            std::min(1.0f, (reconstruction_disp - min_disparity) /
                           (max_disparity - min_disparity)));
    
    // Find bin.
    int32_t bin;
    for (bin = 0; bin < 7; ++ bin) {
      if (value < cumsum[bin + 1]) {
        break;
      }
    }
    
    // Compute red/green/blue values.
    float w = 1.0f - (value - cumsum[bin]) * weights[bin];
    colors[3 * i + 0] =
        static_cast<uint8_t>(
            (w * map[bin][0] + (1.0f - w) * map[bin + 1][0]) * 255.0f);
    colors[3 * i + 1] =
        static_cast<uint8_t>(
            (w * map[bin][1] + (1.0f - w) * map[bin + 1][1]) * 255.0f);
    colors[3 * i + 2] =
        static_cast<uint8_t>(
            (w * map[bin][2] + (1.0f - w) * map[bin + 1][2]) * 255.0f);
  }
  
  //std::string reconstruction_png_path = 
	//	(boost::filesystem::path(visualizations_path)/
	//	 "reconstruction.png").string();
  std::string reconstruction_png_path = visualizations_path + "_reconstruction.png";
	//std::cout << reconstruction_png_path << "\n";
  WritePNGFile(reconstruction_png_path.c_str(), width, height, /*colored*/ true, colors);

}


void CreateBadPixelVisualizations(
    int width,
    int height,
    const std::vector<BadPixelState>* bad_pixels,
    //const char* visualizations_path
    const string & visualizations_path) {
  std::vector<uint8_t> colors(3 * width * height);
  std::vector<std::string> names = {"0_5", "1_0", "2_0", "4_0"};
  for (int k = 0; k < 4; ++ k) {
    for (std::size_t i = 0; i < bad_pixels[k].size(); ++ i) {
      switch (bad_pixels[k][i]) {
      case BadPixelState::kGood:
        colors[3 * i + 0] = 255;
        colors[3 * i + 1] = 255;
        colors[3 * i + 2] = 255;
        break;
      case BadPixelState::kBadButMasked:
        colors[3 * i + 0] = 128;
        colors[3 * i + 1] = 128;
        colors[3 * i + 2] = 128;
        break;
      case BadPixelState::kBad:
        colors[3 * i + 0] = 0;
        colors[3 * i + 1] = 0;
        colors[3 * i + 2] = 0;
        break;
      case BadPixelState::kNoGroundTruth:
        colors[3 * i + 0] = 0;
        colors[3 * i + 1] = 0;
        colors[3 * i + 2] = 255;
        break;
      }
    }
    
		//std::string file_path =
    //    (boost::filesystem::path(visualizations_path) /
    //    ("bad_" + names[k] + ".png")).string();
		std::string file_path = visualizations_path + "_bad_" + names[k] + ".png";
    WritePNGFile(file_path.c_str(), width, height, /*colored*/ true, colors);
  }
}

void CreateErrorVisualizations(
    int width,
    int height,
    const std::vector<float>& errors,
    const std::vector<uint8_t>& mask,
    //const char* visualizations_path,
    const string & visualizations_path) {
  
	const float kMaxError = 5;
  
  std::vector<uint8_t> absolute_colors(3 * width * height);
  std::vector<uint8_t> signed_colors(3 * width * height);
  for (std::size_t i = 0; i < errors.size(); ++ i) {
    float error = errors[i];
    bool masked_out = (mask[i] != 255);
    bool gt_missing = (mask[i] == 0);
    
    uint8_t abs_error_colorvalue =
        std::min<float>(255.0f, 255.99f * fabs(error) / kMaxError) + 0.5f;
    absolute_colors[3 * i + 0] = abs_error_colorvalue;
    absolute_colors[3 * i + 1] = masked_out ? 0 : abs_error_colorvalue;
    absolute_colors[3 * i + 2] = masked_out ? 0 : abs_error_colorvalue;
    
    uint8_t sgn_error_colorvalue =
        std::max(0.f,
            std::min(255.0f, 255.99f * 0.5f * (1.0f + error / kMaxError)))
        + 0.5f;
    signed_colors[3 * i + 0] = gt_missing ? 128 : sgn_error_colorvalue;
    signed_colors[3 * i + 1] = gt_missing ? 128 : (masked_out ? 0 : sgn_error_colorvalue);
    signed_colors[3 * i + 2] = gt_missing ? 128 : (masked_out ? 0 : sgn_error_colorvalue);
  }
  
  //std::string absolute_file_path =
  //    (boost::filesystem::path(visualizations_path) /
  //    "abs_errors.png").string();
  
  std::string absolute_file_path = visualizations_path + "_abs_errors.png";
	
	WritePNGFile(absolute_file_path.c_str(), width, height, /*colored*/ true, absolute_colors);
  
  //std::string signed_file_path =
  //    (boost::filesystem::path(visualizations_path) /
  //    "sgn_errors.png").string();
	
  std::string signed_file_path = visualizations_path  + "_sgn_errors.png";
  WritePNGFile(signed_file_path.c_str(), width, height, /*colored*/ true, signed_colors);

}


#if 0
int main(int argc, char** argv) {
  // Parse parameters.
  if (argc < 4 || argc > 6) {
    std::cerr << "Usage example: " << argv[0]
              << " reconstruction.pfm ground-truth.pfm"
              << " nocc-mask.png [optional: visualization_directory "
              << "(create_training_visualizations ? true : false)]"
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }
  
  const char* reconstruction_path = argv[1];
  const char* ground_truth_path = argv[2];
  const char* mask_path = argv[3];
  bool create_visualizations = (argc >= 5);
  const char* visualizations_path = (create_visualizations ? argv[4] : "");
  bool create_training_visualizations =
      (argc >= 6 && argv[5] == std::string("true"));
  
  if (create_visualizations) {
    boost::filesystem::create_directories(visualizations_path);
  }
  
  std::string error_string;
  
  // Read reconstruction.
  int reconstruction_width, reconstruction_height;
  std::vector<float> reconstruction;
  if (!ReadPFMFile(reconstruction_path, &reconstruction_width,
                   &reconstruction_height, &reconstruction, &error_string)) {
    std::cerr << "Cannot open or parse the reconstruction PFM file: "
              << reconstruction_path
              << " (Error: " << error_string << ")"
              << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }
  
  // Read ground truth.
  int gt_width, gt_height;
  std::vector<float> ground_truth;
  if (!ReadPFMFile(ground_truth_path, &gt_width, &gt_height, &ground_truth,
                   &error_string)) {
    std::cerr << "Cannot open or parse the ground truth PFM file: "
              << ground_truth_path
              << " (Error: " << error_string << ")"
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }
  
  if (reconstruction_width != gt_width || reconstruction_height != gt_height) {
    std::cerr << "Reconstruction image size does not fit the ground truth "
              << "image size." << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }
  
  // Read non-occluded mask.
  std::vector<uint8_t> mask;
  if (!ReadMaskPNGFile(mask_path, &mask)) {
    std::cerr << "Cannot open or parse the non-occluded mask file: "
              << mask_path << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }
  
  // Compute and output metrics.
  Metrics metrics;
  std::vector<BadPixelState> bad_pixels[4];
  std::vector<float> errors;
  
  std::cout << "Non-occluded:" << std::endl;
  ComputeMetrics(true, gt_width, gt_height, reconstruction, ground_truth, mask,
                 &metrics, nullptr, nullptr);
  OutputMetrics(metrics);
  std::cout << std::endl;
  
  std::cout << "All:" << std::endl;
  ComputeMetrics(false, gt_width, gt_height, reconstruction, ground_truth, mask,
                 &metrics, bad_pixels, &errors);
  OutputMetrics(metrics);
  std::cout << std::endl;
  
  if (create_visualizations) {
    float min_disparity, max_disparity;
    GetMinMaxDisparity(ground_truth, &min_disparity, &max_disparity);
    
    CreateReconstructionVisualization(gt_width, gt_height, reconstruction,
                                      min_disparity, max_disparity,
                                      visualizations_path);
    if (create_training_visualizations) {
      CreateBadPixelVisualizations(gt_width, gt_height, bad_pixels,
                                   visualizations_path);
      CreateErrorVisualizations(gt_width, gt_height, errors, mask,
                                visualizations_path);
    }
  }
  
  return static_cast<int>(ReturnCodes::kSuccess);
}
#endif

//const std::string baseDir = "/home/ccj/PKLS/";
//std::string disp_suffix = "_sgm_disp0PKLS";
//std::string imgDir;

int eth3d_2_view_evaluate(
  const std::string & reconstruction_path,// reconstrucition.pfm
  const std::string & ground_truth_path, // ground-truth.pfm
  const std::string & mask_path, // nocc-mask.png 
  const std::string & visualizations_path,
  const std::string & create_training_visualizations, // == "true"
	PyObject * PyMetrics
	){
  // Parse parameters.
	//cout << "doing eth3d evaluation\n";
  
  bool create_visualizations = (visualizations_path != std::string(""));
  bool isCreate_training_visualizations = (create_training_visualizations 
			== std::string("true"));

  //if (create_visualizations) {
  //  boost::filesystem::create_directories(visualizations_path.c_str());
  //}
  
  std::string error_string;
  
  // Read reconstruction.
  int reconstruction_width, reconstruction_height;
  std::vector<float> reconstruction;
  if (!ReadPFMFile(reconstruction_path.c_str(), &reconstruction_width,
                   &reconstruction_height, &reconstruction, &error_string)) {
    std::cerr << "Cannot open or parse the reconstruction PFM file: "
              << reconstruction_path
              << " (Error: " << error_string << ")"
              << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }
  
  // Read ground truth.
  int gt_width, gt_height;
  std::vector<float> ground_truth;
  if (!ReadPFMFile(ground_truth_path.c_str(), &gt_width, &gt_height, &ground_truth,
                   &error_string)) {
    std::cerr << "Cannot open or parse the ground truth PFM file: "
              << ground_truth_path
              << " (Error: " << error_string << ")"
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }
  
  if (reconstruction_width != gt_width || reconstruction_height != gt_height) {
    std::cerr << "Reconstruction image size does not fit the ground truth "
              << "image size." << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }
  
  // Read non-occluded mask.
  std::vector<uint8_t> mask;
  if (!ReadMaskPNGFile(mask_path.c_str(), &mask)) {
    std::cerr << "Cannot open or parse the non-occluded mask file: "
              << mask_path << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }
  
  // Compute and output metrics.
  Metrics metrics_noc, metrics_all;
  std::vector<BadPixelState> bad_pixels[4];
  std::vector<float> errors;
  
  if (Is_Display){	
		std::cout << "Non-occluded:" << std::endl;
	}
  ComputeMetrics(true, gt_width, gt_height, reconstruction, ground_truth, mask,
                 &metrics_noc, nullptr, nullptr);
  
  if (Is_Display){	
		OutputMetrics(metrics_noc);
		std::cout << std::endl;
		std::cout << "Non-occluded:" << std::endl;
		std::cout << "All:" << std::endl;
	}
  	
  ComputeMetrics(false, gt_width, gt_height, reconstruction, ground_truth, mask,
                 &metrics_all, bad_pixels, &errors);
	if (Is_Display){
		OutputMetrics(metrics_all);
    std::cout << std::endl;
	}
  
  if (create_visualizations) {
    float min_disparity, max_disparity;
    GetMinMaxDisparity(ground_truth, &min_disparity, &max_disparity);
    
    CreateReconstructionVisualization(gt_width, gt_height, reconstruction,
                                      min_disparity, max_disparity,
                                      visualizations_path);

    if (isCreate_training_visualizations) {
      CreateBadPixelVisualizations(gt_width, gt_height, bad_pixels, visualizations_path);
      CreateErrorVisualizations(gt_width, gt_height, errors, mask, visualizations_path);
    }
  }
  
  PyArrayObject* PyMetricsA = reinterpret_cast<PyArrayObject*>(PyMetrics);
	double * PyMetricsP = reinterpret_cast<double*>(PyArray_DATA(PyMetricsA));
	// non_occluded
  PyMetricsP[0] =  metrics_noc.coverage;
  PyMetricsP[1] =  metrics_noc.bad_0_5;
  PyMetricsP[2] =  metrics_noc.bad_1_0;
  PyMetricsP[3] =  metrics_noc.bad_2_0;
  PyMetricsP[4] =  metrics_noc.bad_4_0;
  PyMetricsP[5] =  metrics_noc.avgerr;
  PyMetricsP[6] =  metrics_noc.rms;
  PyMetricsP[7] =  metrics_noc.a50;
  PyMetricsP[8] =  metrics_noc.a90;
  PyMetricsP[9] =  metrics_noc.a95;
  PyMetricsP[10] = metrics_noc.a99;
  PyMetricsP[11] = (double)metrics_noc.valid_pix_num;
	
	// all 
  PyMetricsP[12] =  metrics_all.coverage;
  PyMetricsP[13] =  metrics_all.bad_0_5;
  PyMetricsP[14] =  metrics_all.bad_1_0;
  PyMetricsP[15] =  metrics_all.bad_2_0;
  PyMetricsP[16] =  metrics_all.bad_4_0;
  PyMetricsP[17] =  metrics_all.avgerr;
  PyMetricsP[18] =  metrics_all.rms;
  PyMetricsP[19] =  metrics_all.a50;
  PyMetricsP[20] =  metrics_all.a90;
  PyMetricsP[21] =  metrics_all.a95;
  PyMetricsP[22] = metrics_all.a99;
  PyMetricsP[23] = (double)metrics_all.valid_pix_num;

  return static_cast<int>(ReturnCodes::kSuccess);
}


BOOST_PYTHON_MODULE(libeth3d_2_view_evaluate){
	numeric::array::set_module_and_type("numpy", "ndarray");
	def("eth3d_2_view_evaluate", eth3d_2_view_evaluate);
	import_array();
}
