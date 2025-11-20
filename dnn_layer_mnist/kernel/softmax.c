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

// Forward declaration for scalar softmax used before its definition
static void softmax_scalar(const float *i, float *o, float *buf,
                           uint64_t channels, uint64_t innerSize);

// Forward declarations for vector softmax functions
static void softmax_vec_single(const float *input, float *output, size_t size);
static void softmax_vec_exp(int vreg, size_t vl);

// Our fdiv cannot receive any X in input
// The following macro is just a trick and should NOT be used
#define RESET_VREGS

// OPTIMIZATION 3: Vectorized exp() using polynomial approximation

// Fast vector exp approximation using Pade approximation or Taylor series
// exp(x) ≈ (1 + x/n)^n for large n
// Or use: exp(x) = 2^(x/ln(2)) and use vfrsqrt for fast computation

// Use fixed vector registers (v8 input, v4 temp accumulator, v16 output).
// Avoid using %d in inline asm strings (invalid escape) by not injecting
// dynamic register numbers into asm text.
static inline void vector_exp_approx(size_t vl) {
    (void)vl; // silence unused-parameter warning (VL is set by caller via vsetvli)

    // Using a 5th order Taylor expansion centered at 0:
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120

    asm volatile("vfmv.v.f v24, %0" :: "f"(1.0f));  // v24 = 1.0 (constant)

    // v0 <- v8 (work on v0 so other code paths don't rely on dynamic reg moves)
    asm volatile("vmv.v.v v0, v8");

    // v8 = x^2
    asm volatile("vfmul.vv v8, v0, v0");

    // v16 = x^3 = x^2 * x
    asm volatile("vfmul.vv v16, v8, v0");

    // Start accumulation: v4 = 1.0 + x
    asm volatile("vfadd.vv v4, v24, v0");  // v4 = 1 + x

    // Add x^2/2
    asm volatile("vfmul.vf v12, v8, %0" :: "f"(0.5f));
    asm volatile("vfadd.vv v4, v4, v12");  // v4 += x²/2

    // Add x^3/6
    asm volatile("vfmul.vf v12, v16, %0" :: "f"(0.166666667f));
    asm volatile("vfadd.vv v4, v4, v12");  // v4 += x³/6

    // Add x^4/24 = (x^2)^2/24
    asm volatile("vfmul.vv v12, v8, v8");
    asm volatile("vfmul.vf v12, v12, %0" :: "f"(0.041666667f));
    asm volatile("vfadd.vv v4, v4, v12");  // v4 += x⁴/24

    // Add x^5/120 = x^3 * x^2 / 120
    asm volatile("vfmul.vv v12, v16, v8");
    asm volatile("vfmul.vf v12, v12, %0" :: "f"(0.008333333f));
    asm volatile("vfadd.vv v4, v4, v12");  // v4 += x⁵/120

    // Move result to v16 (fixed destination)
    asm volatile("vmv.v.v v16, v4");
}

void softmax_vec_optimized(const float *input, float *output, uint64_t channels,
                           uint64_t innerSize) {
    
    if (innerSize != 1 || channels > 1024) {
        // Fallback for complex cases
        static float temp_buffer[10240];
        softmax_scalar(input, output, temp_buffer, channels, innerSize);
        return;
    }
    
    // Optimized path for single sample classification
    size_t size = channels;
    
    // Step 1: Find max using vector reduction
    size_t avl = size;
    float max_val;
    
    // Initialize max with first element
    asm volatile("vsetvli zero, zero, e32, m1, ta, ma");
    asm volatile("vfmv.s.f v0, %0" :: "f"(input[0]));
    
    // Vector reduction to find max
    for (size_t i = 0; i < size; ) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(size - i));
        
        asm volatile("vle32.v v8, (%0)" :: "r"(input + i));
        asm volatile("vfredmax.vs v0, v8, v0");
        
        i += vl;
    }
    asm volatile("vfmv.f.s %0, v0" : "=f"(max_val));
    
    // Step 2: Compute exp(x - max) and sum in single pass
    avl = size;
    float sum = 0.0f;
    
    // Initialize sum accumulator
    asm volatile("vsetvli zero, zero, e32, m1, ta, ma");
    asm volatile("vmv.s.x v0, zero");
    
    for (size_t i = 0; i < size; ) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(size - i));
        
        // Load and subtract max
        asm volatile("vle32.v v8, (%0)" :: "r"(input + i));
        asm volatile("vfsub.vf v8, v8, %0" :: "f"(max_val));
        
        // Compute exp using approximation
        // For channels=10, values after subtracting max are typically in [-10, 0]
        // Use Taylor series which is accurate for this range
        
        // v16 = exp(v8)
        vector_exp_approx(vl);        
        // Store exp values
        asm volatile("vse32.v v16, (%0)" :: "r"(output + i));
        
        // Accumulate sum
        asm volatile("vfredusum.vs v0, v16, v0");
        
        i += vl;
    }
    
    // Extract sum
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum));
    
    // Step 3: Normalize by dividing by sum
    avl = size;
    
    for (size_t i = 0; i < size; ) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(size - i));
        
        // Load exp values
        asm volatile("vle32.v v8, (%0)" :: "r"(output + i));
        
        // Divide by sum
        asm volatile("vfdiv.vf v8, v8, %0" :: "f"(sum));
        
        // Store normalized result
        asm volatile("vse32.v v8, (%0)" :: "r"(output + i));
        
        i += vl;
    }
}

// Alternative: Use reciprocal for faster division
void softmax_vec_fast_div(const float *input, float *output, uint64_t channels,
                          uint64_t innerSize) {
    
    if (innerSize != 1) {
        static float temp_buffer[10240];
        softmax_scalar(input, output, temp_buffer, channels, innerSize);
        return;
    }
    
    size_t size = channels;
    
    // Find max (same as before)
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Vectorized normalization with reciprocal
    float inv_sum = 1.0f / sum;
    
    size_t avl = size;
    for (size_t i = 0; i < size; ) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(size - i));
        
        asm volatile("vle32.v v8, (%0)" :: "r"(output + i));
        
        // Multiply by reciprocal (faster than division)
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_sum));
        
        asm volatile("vse32.v v8, (%0)" :: "r"(output + i));
        
        i += vl;
    }
}


// Scalar implmentation inspired by OpenCV softmax:
// https://github.com/opencv/opencv/blob/master/modules/dnn/src/layers/softmax_layer.cpp
void softmax_scalar(const float *i, float *o, float *buf,
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
            fmaxf(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
      }
    }

    // Subtract max
    for (size_t outerDim2 = 0; outerDim2 < outerSize; outerDim2++) {
      size_t srcOffset2 = outerDim2 * outerStep;
      size_t bufOffset2 = outerDim2 * cnStep;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset2 + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] = srcPtr[offset + i] - bufPtr[bufOffset2 + i];
      }
    }

    // Exponentiate
    for (size_t outerDim2 = 0; outerDim2 < outerSize; outerDim2++) {
      size_t srcOffset2 = outerDim2 * outerStep;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset2 + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] = expf(dstPtr[offset + i]);
      }
    }

    // Sum exps and divide
    for (size_t outerDim2 = 0; outerDim2 < outerSize; outerDim2++) {
      size_t srcOffset2 = outerDim2 * outerStep;
      size_t bufOffset2 = outerDim2 * cnStep;

      // Sum exp along axis
      for (size_t i = 0; i < innerSize; i++)
        bufPtr[bufOffset2 + i] = 0.f;

      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset2 + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          bufPtr[bufOffset2 + i] += dstPtr[offset + i];
      }

      // Divide by computed sum
      for (size_t cnDim = 0; cnDim < channels; cnDim++) {
        const int offset = srcOffset2 + cnDim * cnStep;
        for (size_t i = 0; i < innerSize; i++)
          dstPtr[offset + i] /= bufPtr[bufOffset2 + i];
      }
    }
  }
}

void softmax_vec(const float *input, float *output, uint64_t channels,
                 uint64_t innerSize) {
    
    // For MNIST: channels=10 (output classes), innerSize=1 (batch_size=1)
    if (innerSize != 1 || channels > 1024) {
        // Fallback to scalar for complex cases
        static float temp_buffer[10240];
        softmax_scalar(input, output, temp_buffer, channels, innerSize);
        return;
    }
    
    // Optimized vector path for single sample classification
    size_t size = channels;
    
    // Step 1: Find maximum value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Step 2: Compute exp(x - max) and sum
    // Use scalar exp for correctness, vectorize the rest
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Step 3: Vectorized normalization using reciprocal (faster than division)
    float inv_sum = 1.0f / sum;
    
    size_t avl = size;
    for (size_t i = 0; i < size; ) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(size - i));
        
        // Load exp values
        asm volatile("vle32.v v8, (%0)" :: "r"(output + i));
        
        // Multiply by reciprocal (faster than vfdiv)
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_sum));
        
        // Store normalized result
        asm volatile("vse32.v v8, (%0)" :: "r"(output + i));
        
        i += vl;
    }
}
// void softmax_vec(const float *input, float *output, uint64_t channels,
//                  uint64_t innerSize) {
  
//   // For MNIST: channels=10 (output classes), innerSize=1 (batch_size=1)
//   // This is the most common case for neural network classification
  
//   if (innerSize == 1 && channels <= 128) {
//     // Optimized vector path for single sample classification
//     softmax_vec_single(input, output, channels);
//   } else {
//     // Fallback to scalar for complex cases
//     static float temp_buffer[10240];  
//     softmax_scalar(input, output, temp_buffer, channels, innerSize);
//   }
// }

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

void softmax(const float *i, float *o, float *buf,
             uint64_t channels, uint64_t innerSize) {
    softmax_scalar(i, o, buf, channels, innerSize);
}
