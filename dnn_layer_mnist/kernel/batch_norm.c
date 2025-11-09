// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// Batch Normalization Implementation for RISC-V Vector Extension

#include "batch_norm.h"
// Vector operations temporarily disabled due to missing riscv_vector.h
// #include "riscv_vector.h"
#include <math.h>

// Scalar implementation for verification
void batch_norm_scalar(float *output, const float *input,
                       const float *gamma, const float *beta,
                       const float *mean, const float *var,
                       float epsilon, size_t batch_size, size_t channels) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < channels; c++) {
            size_t idx = b * channels + c;
            // Normalize: (x - mean) / sqrt(var + eps)
            float normalized = (input[idx] - mean[c]) / sqrtf(var[c] + epsilon);
            // Scale and shift: gamma * normalized + beta
            output[idx] = gamma[c] * normalized + beta[c];
        }
    }
}

// Vectorized batch normalization using inline assembly
void batch_norm_vec(float *output, const float *input,
                    const float *gamma, const float *beta,
                    const float *mean, const float *var,
                    float epsilon, size_t batch_size, size_t channels) {
    
    // Process each batch sample
    for (size_t b = 0; b < batch_size; b++) {
        const float *input_ptr = input + b * channels;
        float *output_ptr = output + b * channels;
        
        size_t avl = channels;
        size_t vl;
        size_t c = 0;
        
        // Stripmine loop over channels using vector instructions
        while (avl > 0) {
            // Set vector length
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
            
            // Load input values: v8 = input[c:c+vl]
            asm volatile("vle32.v v8, (%0)" : : "r"(input_ptr + c));
            
            // Load batch norm parameters
            asm volatile("vle32.v v16, (%0)" : : "r"(mean + c));     // v16 = mean
            asm volatile("vle32.v v24, (%0)" : : "r"(var + c));      // v24 = variance
            asm volatile("vle32.v v0, (%0)" : : "r"(gamma + c));     // v0 = gamma  
            asm volatile("vle32.v v4, (%0)" : : "r"(beta + c));      // v4 = beta
            
            // Create epsilon vector: v12 = epsilon (broadcast)
            asm volatile("vfmv.v.f v12, %0" : : "f"(epsilon));
            
            // Compute standard deviation: v24 = sqrt(var + epsilon)
            asm volatile("vfadd.vv v24, v24, v12");  // v24 = var + epsilon
            asm volatile("vfsqrt.v v24, v24");       // v24 = sqrt(var + epsilon)
            
            // Normalize: v8 = (input - mean) / std
            asm volatile("vfsub.vv v8, v8, v16");    // v8 = input - mean
            asm volatile("vfdiv.vv v8, v8, v24");    // v8 = (input - mean) / std
            
            // Scale and shift: v8 = gamma * normalized + beta
            asm volatile("vfmul.vv v8, v0, v8");     // v8 = gamma * normalized
            asm volatile("vfadd.vv v8, v8, v4");     // v8 = gamma * normalized + beta
            
            // Store result
            asm volatile("vse32.v v8, (%0)" : : "r"(output_ptr + c));
            
            avl -= vl;
            c += vl;
        }
    }
}