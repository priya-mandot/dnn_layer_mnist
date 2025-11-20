// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// Batch Normalization for RISC-V Vector Extension

#ifndef __BATCH_NORM_H__
#define __BATCH_NORM_H__

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Vectorized batch normalization
 * 
 * Computes: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
 * 
 * @param output Output buffer [batch_size x channels]
 * @param input Input buffer [batch_size x channels]
 * @param gamma Scale parameter [channels]
 * @param beta Shift parameter [channels]
 * @param mean Running mean [channels]
 * @param var Running variance [channels]
 * @param epsilon Small constant for numerical stability
 * @param batch_size Number of samples in batch
 * @param channels Number of channels/features
 */
void batch_norm_vec(float *output, const float *input,
                    const float *gamma, const float *beta,
                    const float *mean, const float *var,
                    float epsilon, size_t batch_size, size_t channels);

/**
 * @brief Scalar batch normalization (for verification)
 */
void batch_norm_scalar(float *output, const float *input,
                       const float *gamma, const float *beta,
                       const float *mean, const float *var,
                       float epsilon, size_t batch_size, size_t channels);

void batch_norm_vec_fused(float *output, const float *input,
                          const float *gamma, const float *beta,
                          const float *mean, const float *var,
                          float epsilon, size_t batch_size, size_t channels);

void batch_norm_vec_optimized(float *output, const float *input,
                               const float *gamma, const float *beta,
                               const float *mean, const float *var,
                               float epsilon, size_t batch_size, size_t channels);

#endif // __BATCH_NORM_H__