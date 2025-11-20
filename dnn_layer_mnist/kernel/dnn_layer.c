// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// Complete DNN Layer Implementation

#include "dnn_layer.h"
#include "fmatmul.h"
#include "batch_norm.h"
#include "relu.h"
#include "softmax.h"
#include <string.h>
#include <math.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// External debug buffers
extern float matmul_out_scalar[];
extern float matmul_out_vec[];
extern float bn_out_scalar[];
extern float bn_out_vec[];
extern float relu_out_scalar[];
extern float relu_out_vec[];

// Vectorized forward pass - USES OPTIMIZED FUNCTIONS
void dnn_layer_forward(float *output, const float *input,
                       const dnn_params_t *params, const dnn_config_t *config,
                       float *temp_buffer) {
    
    // Step 1: Matrix multiplication - Use optimized version with transposed weights
    printf("before matmul\n");
    
    // Check if we have batch_size=1 for GEMV optimization
    if (config->batch_size == 1) {
        // Use optimized GEMV (expects transposed weights)
        fmatmul_gemv_optimized(temp_buffer, input, params->weights,
                               config->input_size, config->output_size);
    } else {
        // Fallback to general matmul
        fmatmul(temp_buffer, input, params->weights,
                config->batch_size, config->input_size, config->output_size);
    }
    
    printf("after matmul\n");
    
    // Save for debugging
    memcpy(matmul_out_vec, temp_buffer, config->batch_size * config->output_size * sizeof(float));

    // Step 2: Batch Normalization - using optimized version
    printf("before batch norm\n");
    batch_norm_vec(output, temp_buffer,
                   params->bn_gamma, params->bn_beta,
                   params->bn_mean, params->bn_var,
                   params->bn_epsilon,
                   config->batch_size, config->output_size);
    printf("after batch norm\n");
    
    // Save for debugging
    memcpy(bn_out_vec, output, config->batch_size * config->output_size * sizeof(float));
    
    // Step 3: ReLU Activation
    printf("before relu\n");
    relu_vec(temp_buffer, output, config->batch_size * config->output_size);
    printf("after relu\n");
    
    // Save for debugging
    memcpy(relu_out_vec, temp_buffer, config->batch_size * config->output_size * sizeof(float));
    
    // Step 4: Softmax - using optimized version
    printf("before softmax\n");
    softmax_vec(temp_buffer, output, config->output_size, config->batch_size);
    printf("after softmax\n");
}


// Standard scalar implementation of fmatmul_scalar (C = A * B^T)
static void fmatmul_scalar(float *c, const float *a, const float *b,
                           unsigned long int M, unsigned long int N, unsigned long int P) {
    // M = batch_size
    // N = input_size
    // P = output_size
    
    for (unsigned long int m = 0; m < M; ++m) {        // Iterate rows of A (batch_size)
        for (unsigned long int p = 0; p < P; ++p) {    // Iterate rows of B (output_size)
            float sum = 0.0f;
            for (unsigned long int n = 0; n < N; ++n) { // Iterate cols of A and B (input_size)
                // C[m,p] += A[m,n] * B[p,n]
                sum += a[m * N + n] * b[p * N + n];
            }
            c[m * P + p] = sum;
        }
    }
}

// Corrected softmax_scalar implementation
static void softmax_scalar_impl(const float *i, float *o, float *buf,
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

// Scalar forward pass for verification
void dnn_layer_forward_scalar(float *output, const float *input,
                              const dnn_params_t *params, const dnn_config_t *config,
                              float *temp_buffer) {
    printf("starting scalar execution:\n");
    float *softmax_buf = temp_buffer + config->batch_size * config->output_size;
    printf("allocated softmax buffer at offset %p\n", (void *)softmax_buf);
    
    // Step 1: Matrix multiplication
    fmatmul_scalar(temp_buffer, input, params->weights,
                   config->batch_size, config->input_size, config->output_size);
    printf("completed matrix multiplication\n");
    
    // Save for debugging
    memcpy(matmul_out_scalar, temp_buffer, config->batch_size * config->output_size * sizeof(float));
    
    // Step 2: Batch Normalization
    batch_norm_scalar(output, temp_buffer,
                      params->bn_gamma, params->bn_beta,
                      params->bn_mean, params->bn_var,
                      params->bn_epsilon,
                      config->batch_size, config->output_size);
    printf("completed batch normalization\n");
    
    // Save for debugging
    memcpy(bn_out_scalar, output, config->batch_size * config->output_size * sizeof(float));
    
    // Step 3: ReLU Activation
    relu_scalar(temp_buffer, output, config->batch_size * config->output_size);
    printf("completed ReLU activation\n");
    
    // Save for debugging
    memcpy(relu_out_scalar, temp_buffer, config->batch_size * config->output_size * sizeof(float));
    
    // Step 4: Softmax
    softmax_scalar_impl(temp_buffer, output, softmax_buf,
                        config->output_size, config->batch_size);
    printf("completed softmax\n");
}
