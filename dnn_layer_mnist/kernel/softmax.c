// Copyright 2022 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include <math.h>
#include <string.h>

// Vector operations temporarily disabled due to missing riscv_vector.h
// #include "riscv_vector.h"

#include "../softmax/lib/exp.h"

// Forward declarations for vector softmax functions
static void softmax_vec_single(const float *input, float *output, size_t size);
static void softmax_vec_exp(int vreg, size_t vl);

// Our fdiv cannot receive any X in input
// The following macro is just a trick and should NOT be used
#define RESET_VREGS

// Scalar implmentation inspired by OpenCV softmax:
// https://github.com/opencv/opencv/blob/master/modules/dnn/src/layers/softmax_layer.cpp
void softmax_scalar(const float *i, const float *o, const float *buf,
                    uint64_t channels, uint64_t innerSize) {

  // OpenCV names
  float *srcPtr = (float *)i;
  float *bufPtr = (float *)buf;
  float *dstPtr = (float *)o;

  // Batch size == 1
  size_t outerSize = 1;

  // Steps
  size_t outerStep = channels * innerSize;
  size_t cnStep = innerSize;

  // Compute max along axis
  for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {

    size_t srcOffset = outerDim * outerStep;
    size_t bufOffset = outerDim * cnStep;

    memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

    for (size_t cnDim = 1; cnDim < channels; cnDim++) {
      for (size_t i = 0; i < innerSize; i++) {
        bufPtr[bufOffset + i] =
            fmax(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
      }
    }

    // Subtract max
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {
      size_t srcOffset = outerDim * outerStep;
      size_t bufOffset = outerDim * cnStep;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] = srcPtr[offset + i] - bufPtr[bufOffset + i];
      }
    }

    // Exponentiate
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {
      size_t srcOffset = outerDim * outerStep;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] = exp(dstPtr[offset + i]);
      }
    }

    // Sum exps and divide
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++) {
      size_t srcOffset = outerDim * outerStep;
      size_t bufOffset = outerDim * cnStep;

      // Sum exp along axis
      for (size_t i = 0; i < innerSize; i++)
        bufPtr[bufOffset + i] = 0.f;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          bufPtr[bufOffset + i] += dstPtr[offset + i];
      }

      // Divide by computed sum
      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] /= bufPtr[bufOffset + i];
      }
    }
  }
}

void softmax_vec(const float *input, float *output, uint64_t channels,
                 uint64_t innerSize) {
  
  // For MNIST: channels=10 (output classes), innerSize=1 (batch_size=1)
  // This is the most common case for neural network classification
  
  if (innerSize == 1 && channels <= 128) {
    // Optimized vector path for single sample classification
    softmax_vec_single(input, output, channels);
  } else {
    // Fallback to scalar for complex cases
    static float temp_buffer[10240];  
    softmax_scalar(input, output, temp_buffer, channels, innerSize);
  }
}

// Vector softmax for single sample (most common neural network case)
static void softmax_vec_single(const float *input, float *output, size_t size) {
    
    // Step 1: Find maximum value for numerical stability (scalar for simplicity)
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Step 2: Compute exp(x - max) using vector operations
    size_t avl = size;
    size_t vl;
    size_t i = 0;
    
    // First pass: compute exp(x - max) and store
    while (avl > 0) {
        // Set vector length
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
        
        // Load input values: v8 = input[i:i+vl]
        asm volatile("vle32.v v8, (%0)" : : "r"(input + i));
        
        // Subtract max for numerical stability: v8 = input - max_val
        asm volatile("vfsub.vf v8, v8, %0" : : "f"(max_val));
        
        // Compute exp using scalar calls for each element (simple but correct)
        // Store intermediate results  
        asm volatile("vse32.v v8, (%0)" : : "r"(output + i));
        
        // Apply exp to each element (fallback to ensure correctness)
        for (size_t j = 0; j < vl; j++) {
            output[i + j] = expf(output[i + j]);
        }
        
        avl -= vl;
        i += vl;
    }
    
    // Step 3: Compute sum of exponentials
    float sum = 0.0f;
    for (size_t j = 0; j < size; j++) {
        sum += output[j];
    }
    
    // Step 4: Normalize by dividing by sum using vector operations
    avl = size;
    i = 0;
    
    while (avl > 0) {
        // Set vector length  
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
        
        // Load exp values: v8 = exp_values[i:i+vl]
        asm volatile("vle32.v v8, (%0)" : : "r"(output + i));
        
        // Divide by sum: v8 = exp_values / sum
        asm volatile("vfdiv.vf v8, v8, %0" : : "f"(sum));
        
        // Store final softmax result
        asm volatile("vse32.v v8, (%0)" : : "r"(output + i));
        
        avl -= vl;
        i += vl;
    }
}

// Helper function: removed complex exp implementation, using expf() for correctness
static void softmax_vec_exp(int vreg, size_t vl) {
    // This function is no longer used - keeping for compatibility
    (void)vreg;
    (void)vl;
}

// Alias for backward compatibility
void softmax(const float *i, const float *o, const float *buf,
             uint64_t channels, uint64_t innerSize) {
    softmax_scalar(i, o, buf, channels, innerSize);
}
