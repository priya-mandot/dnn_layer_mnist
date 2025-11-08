// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// ReLU Activation for RISC-V Vector Extension

#ifndef __RELU_H__
#define __RELU_H__

#include <stddef.h>

/**
 * @brief Vectorized ReLU activation
 * 
 * Computes: y = max(0, x)
 * 
 * @param output Output buffer
 * @param input Input buffer
 * @param size Number of elements
 */
void relu_vec(float *output, const float *input, size_t size);

/**
 * @brief In-place vectorized ReLU activation
 * 
 * Computes: x = max(0, x)
 * 
 * @param data Input/Output buffer
 * @param size Number of elements
 */
void relu_vec_inplace(float *data, size_t size);

/**
 * @brief Scalar ReLU activation (for verification)
 */
void relu_scalar(float *output, const float *input, size_t size);

#endif // __RELU_H__