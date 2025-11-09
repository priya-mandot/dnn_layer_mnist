// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// DNN Layer Interface for CVA6-Ara2 Platform

#ifndef __DNN_LAYER_H__
#define __DNN_LAYER_H__

#include <stdint.h>
#include <stddef.h>

// DNN Layer configuration structure
typedef struct {
    size_t input_size;      // Number of input features
    size_t output_size;     // Number of output features (classes)
    size_t batch_size;      // Batch size (typically 1 for inference)
} dnn_config_t;

// DNN Layer parameters structure
typedef struct {
    float *weights;         // Weight matrix [output_size x input_size]
    float *bn_gamma;        // Batch norm scale parameter [output_size]
    float *bn_beta;         // Batch norm shift parameter [output_size]
    float *bn_mean;         // Batch norm running mean [output_size]
    float *bn_var;          // Batch norm running variance [output_size]
    float bn_epsilon;       // Small constant for numerical stability
} dnn_params_t;

// Timing structure for layer profiling
typedef struct {
    int64_t matmul_cycles;
    int64_t batchnorm_cycles;
    int64_t relu_cycles;
    int64_t softmax_cycles;
    int64_t total_cycles;
} layer_timing_t;

/**
 * @brief Execute a complete DNN feedforward layer
 * 
 * Performs: Input -> MatMul -> BatchNorm -> ReLU -> Softmax -> Output
 * 
 * @param output Output buffer [batch_size x output_size]
 * @param input Input buffer [batch_size x input_size]
 * @param params DNN layer parameters
 * @param config DNN layer configuration
 * @param temp_buffer Temporary buffer for intermediate results [batch_size x output_size]
 **/
void dnn_layer_forward(float *output, const float *input, 
                       const dnn_params_t *params, const dnn_config_t *config,
                       float *temp_buffer);

/**
 * @brief Execute DNN layer with scalar implementation (for verification)
 */
void dnn_layer_forward_scalar(float *output, const float *input,
                              const dnn_params_t *params, const dnn_config_t *config,
                              float *temp_buffer);

/**
 * @brief Execute DNN layer with detailed timing (vectorized)
 * 
 * @param timing Output parameter for timing breakdown
 */
void dnn_layer_forward_timed(float *output, const float *input,
                             const dnn_params_t *params, const dnn_config_t *config,
                             float *temp_buffer, layer_timing_t *timing);

/**
 * @brief Execute DNN layer with detailed timing (scalar)
 * 
 * @param timing Output parameter for timing breakdown
 */
void dnn_layer_forward_scalar_timed(float *output, const float *input,
                                   const dnn_params_t *params, const dnn_config_t *config,
                                   float *temp_buffer, layer_timing_t *timing);

#endif // __DNN_LAYER_H__
