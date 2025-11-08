// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// Softmax Activation for RISC-V Vector Extension

#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Vectorized softmax activation
 * 
 * Computes softmax along the channel dimension
 * 
 * @param i Input buffer [channels x innerSize]
 * @param o Output buffer [channels x innerSize]
 * @param channels Number of channels (classes)
 * @param innerSize Inner dimension size (typically batch size)
 */
void softmax_vec(const float *i, float *o, uint64_t channels, uint64_t innerSize);

/**
 * @brief Scalar softmax activation (for verification)
 * 
 * @param i Input buffer
 * @param o Output buffer
 * @param buf Temporary buffer [innerSize]
 * @param channels Number of channels
 * @param innerSize Inner dimension size
 */
void softmax_scalar(const float *i, float *o, float *buf,
                    uint64_t channels, uint64_t innerSize);

#endif // __SOFTMAX_H__