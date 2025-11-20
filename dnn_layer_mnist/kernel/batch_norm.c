// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// Batch Normalization Implementation for RISC-V Vector Extension

#include "batch_norm.h"
// Vector operations temporarily disabled due to missing riscv_vector.h
// #include "riscv_vector.h"
#include <math.h>

// OPTIMIZATION 2: Reduce vector instruction count in BatchNorm

void batch_norm_vec_optimized(float *output, const float *input,
                               const float *gamma, const float *beta,
                               const float *mean, const float *var,
                               float epsilon, size_t batch_size, size_t channels) {
    
    // For batch_size=1, simplify to single pass
    if (batch_size == 1) {
        size_t avl = channels;
        size_t vl;
        
        for (size_t c = 0; c < channels; c += avl) {
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(channels - c));
            avl = vl;
            
            // Load everything in one go
            const float *in_ptr = input + c;
            const float *mean_ptr = mean + c;
            const float *var_ptr = var + c;
            const float *gamma_ptr = gamma + c;
            const float *beta_ptr = beta + c;
            float *out_ptr = output + c;
            
            // Single vector operation chain:
            // output = gamma * (input - mean) / sqrt(var + eps) + beta
            
            asm volatile("vle32.v v0, (%0)" :: "r"(in_ptr));      // v0 = input
            asm volatile("vle32.v v8, (%0)" :: "r"(mean_ptr));    // v8 = mean
            asm volatile("vle32.v v16, (%0)" :: "r"(var_ptr));    // v16 = var
            asm volatile("vle32.v v24, (%0)" :: "r"(gamma_ptr));  // v24 = gamma
            
            // Compute: v0 = (input - mean)
            asm volatile("vfsub.vv v0, v0, v8");
            
            // Compute: v16 = sqrt(var + epsilon)
            asm volatile("vfadd.vf v16, v16, %0" :: "f"(epsilon));
            asm volatile("vfsqrt.v v16, v16");
            
            // Compute: v0 = (input - mean) / sqrt(var + eps)
            asm volatile("vfdiv.vv v0, v0, v16");
            
            // Compute: v0 = gamma * normalized
            asm volatile("vfmul.vv v0, v0, v24");
            
            // Load beta and add
            asm volatile("vle32.v v8, (%0)" :: "r"(beta_ptr));
            asm volatile("vfadd.vv v0, v0, v8");
            
            // Store final result
            asm volatile("vse32.v v0, (%0)" :: "r"(out_ptr));
        }
        return;
    }
    
    // General case for batch_size > 1
    for (size_t b = 0; b < batch_size; b++) {
        const float *input_ptr = input + b * channels;
        float *output_ptr = output + b * channels;
        
        size_t avl = channels;
        size_t vl;
        
        for (size_t c = 0; c < channels; c += avl) {
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(channels - c));
            avl = vl;
            
            // Same optimized computation as above
            const float *in_ptr = input_ptr + c;
            const float *mean_ptr = mean + c;
            const float *var_ptr = var + c;
            const float *gamma_ptr = gamma + c;
            const float *beta_ptr = beta + c;
            float *out_ptr = output_ptr + c;
            
            asm volatile("vle32.v v0, (%0)" :: "r"(in_ptr));
            asm volatile("vle32.v v8, (%0)" :: "r"(mean_ptr));
            asm volatile("vle32.v v16, (%0)" :: "r"(var_ptr));
            asm volatile("vle32.v v24, (%0)" :: "r"(gamma_ptr));
            
            asm volatile("vfsub.vv v0, v0, v8");
            asm volatile("vfadd.vf v16, v16, %0" :: "f"(epsilon));
            asm volatile("vfsqrt.v v16, v16");
            asm volatile("vfdiv.vv v0, v0, v16");
            asm volatile("vfmul.vv v0, v0, v24");
            
            asm volatile("vle32.v v8, (%0)" :: "r"(beta_ptr));
            asm volatile("vfadd.vv v0, v0, v8");
            
            asm volatile("vse32.v v0, (%0)" :: "r"(out_ptr));
        }
    }
}

// Alternative: Use VFMADD for fused operations where possible
void batch_norm_vec_fused(float *output, const float *input,
                          const float *gamma, const float *beta,
                          const float *mean, const float *var,
                          float epsilon, size_t batch_size, size_t channels) {
    
    for (size_t b = 0; b < batch_size; b++) {
        const float *input_ptr = input + b * channels;
        float *output_ptr = output + b * channels;
        
        size_t avl = channels;
        
        for (size_t c = 0; c < channels; c += avl) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(channels - c));
            avl = vl;
            
            // Compute 1/sqrt(var + eps) once (rsqrt would be ideal)
            asm volatile("vle32.v v16, (%0)" :: "r"(var + c));
            asm volatile("vfadd.vf v16, v16, %0" :: "f"(epsilon));
            asm volatile("vfsqrt.v v16, v16");
            
            // Load gamma and compute: gamma / sqrt(var + eps) = scale
            asm volatile("vle32.v v24, (%0)" :: "r"(gamma + c));
            asm volatile("vfdiv.vv v24, v24, v16");  // v24 = scale
            
            // Compute: beta - mean * scale = offset
            asm volatile("vle32.v v8, (%0)" :: "r"(mean + c));
            asm volatile("vle32.v v16, (%0)" :: "r"(beta + c));
            asm volatile("vfnmsac.vv v16, v8, v24");  // v16 = beta - mean*scale
            
            // Final: output = input * scale + offset
            asm volatile("vle32.v v0, (%0)" :: "r"(input_ptr + c));
            asm volatile("vfmadd.vv v0, v24, v16");  // v0 = input*scale + offset
            
            asm volatile("vse32.v v0, (%0)" :: "r"(output_ptr + c));
        }
    }
}

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

// batch_norm.c - Replace batch_norm_vec with optimized version

void batch_norm_vec(float *output, const float *input,
                    const float *gamma, const float *beta,
                    const float *mean, const float *var,
                    float epsilon, size_t batch_size, size_t channels) {
    
    // Optimized for batch_size=1 (most common case)
    if (batch_size == 1) {
        size_t avl = channels;
        
        for (size_t c = 0; c < channels; c += avl) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(channels - c));
            avl = vl;
            
            const float *in_ptr = input + c;
            const float *mean_ptr = mean + c;
            const float *var_ptr = var + c;
            const float *gamma_ptr = gamma + c;
            const float *beta_ptr = beta + c;
            float *out_ptr = output + c;
            
            // Load all inputs
            asm volatile("vle32.v v0, (%0)" :: "r"(in_ptr));      // v0 = input
            asm volatile("vle32.v v8, (%0)" :: "r"(mean_ptr));    // v8 = mean
            asm volatile("vle32.v v16, (%0)" :: "r"(var_ptr));    // v16 = var
            asm volatile("vle32.v v24, (%0)" :: "r"(gamma_ptr));  // v24 = gamma
            
            // Compute: v0 = (input - mean)
            asm volatile("vfsub.vv v0, v0, v8");
            
            // Compute: v16 = sqrt(var + epsilon)
            asm volatile("vfadd.vf v16, v16, %0" :: "f"(epsilon));
            asm volatile("vfsqrt.v v16, v16");
            
            // Compute: v0 = (input - mean) / sqrt(var + eps)
            asm volatile("vfdiv.vv v0, v0, v16");
            
            // Compute: v0 = gamma * normalized
            asm volatile("vfmul.vv v0, v0, v24");
            
            // Load beta and add
            asm volatile("vle32.v v8, (%0)" :: "r"(beta_ptr));
            asm volatile("vfadd.vv v0, v0, v8");
            
            // Store final result
            asm volatile("vse32.v v0, (%0)" :: "r"(out_ptr));
        }
        return;
    }
    
    // General case for batch_size > 1
    for (size_t b = 0; b < batch_size; b++) {
        const float *input_ptr = input + b * channels;
        float *output_ptr = output + b * channels;
        
        size_t avl = channels;
        
        for (size_t c = 0; c < channels; c += avl) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(channels - c));
            avl = vl;
            
            const float *in_ptr = input_ptr + c;
            const float *mean_ptr = mean + c;
            const float *var_ptr = var + c;
            const float *gamma_ptr = gamma + c;
            const float *beta_ptr = beta + c;
            float *out_ptr = output_ptr + c;
            
            asm volatile("vle32.v v0, (%0)" :: "r"(in_ptr));
            asm volatile("vle32.v v8, (%0)" :: "r"(mean_ptr));
            asm volatile("vle32.v v16, (%0)" :: "r"(var_ptr));
            asm volatile("vle32.v v24, (%0)" :: "r"(gamma_ptr));
            
            asm volatile("vfsub.vv v0, v0, v8");
            asm volatile("vfadd.vf v16, v16, %0" :: "f"(epsilon));
            asm volatile("vfsqrt.v v16, v16");
            asm volatile("vfdiv.vv v0, v0, v16");
            asm volatile("vfmul.vv v0, v0, v24");
            
            asm volatile("vle32.v v8, (%0)" :: "r"(beta_ptr));
            asm volatile("vfadd.vv v0, v0, v8");
            
            asm volatile("vse32.v v0, (%0)" :: "r"(out_ptr));
        }
    }
}

// Vectorized batch normalization using inline assembly
// void batch_norm_vec(float *output, const float *input,
//                     const float *gamma, const float *beta,
//                     const float *mean, const float *var,
//                     float epsilon, size_t batch_size, size_t channels) {
    
//     // Process each batch sample
//     for (size_t b = 0; b < batch_size; b++) {
//         const float *input_ptr = input + b * channels;
//         float *output_ptr = output + b * channels;
        
//         size_t avl = channels;
//         size_t vl;
//         size_t c = 0;
        
//         // Stripmine loop over channels using vector instructions
//         while (avl > 0) {
//             // Set vector length
//             asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
            
//             // Load input values: v8 = input[c:c+vl]
//             asm volatile("vle32.v v8, (%0)" : : "r"(input_ptr + c));
            
//             // Load batch norm parameters
//             asm volatile("vle32.v v16, (%0)" : : "r"(mean + c));     // v16 = mean
//             asm volatile("vle32.v v24, (%0)" : : "r"(var + c));      // v24 = variance
//             asm volatile("vle32.v v0, (%0)" : : "r"(gamma + c));     // v0 = gamma  
//             asm volatile("vle32.v v4, (%0)" : : "r"(beta + c));      // v4 = beta
            
//             // Create epsilon vector: v12 = epsilon (broadcast)
//             asm volatile("vfmv.v.f v12, %0" : : "f"(epsilon));
            
//             // Compute standard deviation: v24 = sqrt(var + epsilon)
//             asm volatile("vfadd.vv v24, v24, v12");  // v24 = var + epsilon
//             asm volatile("vfsqrt.v v24, v24");       // v24 = sqrt(var + epsilon)
            
//             // Normalize: v8 = (input - mean) / std
//             asm volatile("vfsub.vv v8, v8, v16");    // v8 = input - mean
//             asm volatile("vfdiv.vv v8, v8, v24");    // v8 = (input - mean) / std
            
//             // Scale and shift: v8 = gamma * normalized + beta
//             asm volatile("vfmul.vv v8, v0, v8");     // v8 = gamma * normalized
//             asm volatile("vfadd.vv v8, v8, v4");     // v8 = gamma * normalized + beta
            
//             // Store result
//             asm volatile("vse32.v v8, (%0)" : : "r"(output_ptr + c));
            
//             avl -= vl;
//             c += vl;
//         }
//     }
// }