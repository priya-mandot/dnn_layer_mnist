// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// Complete DNN Layer Implementation - With Detailed Timing

#include "dnn_layer.h"
#include "fmatmul.h"
#include "batch_norm.h"
#include "relu.h"
#include "softmax.h"
#include "runtime.h"  // For timing functions
#include <string.h>
#include <math.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Timing structure
// typedef struct {
//     int64_t matmul_cycles;
//     int64_t batchnorm_cycles;
//     int64_t relu_cycles;
//     int64_t softmax_cycles;
//     int64_t total_cycles;
// } layer_timing_t;

// External debug buffers
extern float matmul_out_scalar[];
extern float matmul_out_vec[];
extern float bn_out_scalar[];
extern float bn_out_vec[];
extern float relu_out_scalar[];
extern float relu_out_vec[];

// Vectorized forward pass with timing
void dnn_layer_forward_timed(float *output, const float *input,
                             const dnn_params_t *params, const dnn_config_t *config,
                             float *temp_buffer, layer_timing_t *timing) {
    
    // Step 1: Matrix multiplication - Input × Weights^T
    start_timer();
    fmatmul(temp_buffer, input, params->weights,
            config->batch_size, config->input_size, config->output_size);
    stop_timer();
    timing->matmul_cycles = get_timer();
    memcpy(matmul_out_vec, temp_buffer, config->batch_size * config->output_size * sizeof(float));

    // Step 2: Batch Normalization (using scalar for now)
    start_timer();
    batch_norm_scalar(output, temp_buffer,
                      params->bn_gamma, params->bn_beta,
                      params->bn_mean, params->bn_var,
                      params->bn_epsilon,
                      config->batch_size, config->output_size);
    stop_timer();
    timing->batchnorm_cycles = get_timer();
    memcpy(bn_out_vec, output, config->batch_size * config->output_size * sizeof(float));
    
    // Step 3: ReLU Activation
    start_timer();
    relu_vec(temp_buffer, output, config->batch_size * config->output_size);
    stop_timer();
    timing->relu_cycles = get_timer();
    memcpy(relu_out_vec, temp_buffer, config->batch_size * config->output_size * sizeof(float));
    
    // Step 4: Softmax
    start_timer();
    softmax_vec(temp_buffer, output, config->output_size, config->batch_size);
    stop_timer();
    timing->softmax_cycles = get_timer();
}

// Original vectorized forward pass (for backward compatibility)
void dnn_layer_forward(float *output, const float *input,
                       const dnn_params_t *params, const dnn_config_t *config,
                       float *temp_buffer) {
    layer_timing_t dummy_timing;
    dnn_layer_forward_timed(output, input, params, config, temp_buffer, &dummy_timing);
}

// Standard scalar implementation of fmatmul_scalar (C = A * B^T)
static void fmatmul_scalar(float *c, const float *a, const float *b,
                           unsigned long int M, unsigned long int N, unsigned long int P) {
    for (unsigned long int m = 0; m < M; ++m) {
        for (unsigned long int p = 0; p < P; ++p) {
            float sum = 0.0f;
            for (unsigned long int n = 0; n < N; ++n) {
                sum += a[m * N + n] * b[p * N + n];
            }
            c[m * P + p] = sum;
        }
    }
}

// Corrected softmax_scalar implementation
static void softmax_scalar_impl(const float *i, float *o, float *buf,
                                uint64_t channels, uint64_t innerSize) {
    float *srcPtr = (float *)i;
    float *bufPtr = (float *)buf;
    float *dstPtr = (float *)o;

    size_t outerSize = 1;
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

            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] = 0.f;

            for (size_t cnDim = 0; cnDim < channels; cnDim++) {
                const int offset = srcOffset + cnDim * cnStep;
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] += dstPtr[offset + i];
            }

            for (size_t cnDim = 0; cnDim < channels; cnDim++) {
                const int offset = srcOffset + cnDim * cnStep;
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[offset + i] /= bufPtr[bufOffset + i];
            }
        }
    }
}

// Scalar forward pass with timing
void dnn_layer_forward_scalar_timed(float *output, const float *input,
                                   const dnn_params_t *params, const dnn_config_t *config,
                                   float *temp_buffer, layer_timing_t *timing) {
    float *softmax_buf = temp_buffer + config->batch_size * config->output_size;
    
    // Step 1: Matrix multiplication
    start_timer();
    fmatmul_scalar(temp_buffer, input, params->weights,
                   config->batch_size, config->input_size, config->output_size);
    stop_timer();
    timing->matmul_cycles = get_timer();
    memcpy(matmul_out_scalar, temp_buffer, config->batch_size * config->output_size * sizeof(float));
    
    // Step 2: Batch Normalization
    start_timer();
    batch_norm_scalar(output, temp_buffer,
                      params->bn_gamma, params->bn_beta,
                      params->bn_mean, params->bn_var,
                      params->bn_epsilon,
                      config->batch_size, config->output_size);
    stop_timer();
    timing->batchnorm_cycles = get_timer();
    memcpy(bn_out_scalar, output, config->batch_size * config->output_size * sizeof(float));
    
    // Step 3: ReLU Activation
    start_timer();
    relu_scalar(temp_buffer, output, config->batch_size * config->output_size);
    stop_timer();
    timing->relu_cycles = get_timer();
    memcpy(relu_out_scalar, temp_buffer, config->batch_size * config->output_size * sizeof(float));
    
    // Step 4: Softmax
    start_timer();
    softmax_scalar_impl(temp_buffer, output, softmax_buf,
                        config->output_size, config->batch_size);
    stop_timer();
    timing->softmax_cycles = get_timer();
}

// Original scalar forward pass (for backward compatibility)
void dnn_layer_forward_scalar(float *output, const float *input,
                              const dnn_params_t *params, const dnn_config_t *config,
                              float *temp_buffer) {
    layer_timing_t dummy_timing;
    dnn_layer_forward_scalar_timed(output, input, params, config, temp_buffer, &dummy_timing);
}
