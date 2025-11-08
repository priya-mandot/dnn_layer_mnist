// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// ReLU Activation Implementation for RISC-V Vector Extension

#include "relu.h"
#include <stddef.h>
// Vector operations using inline assembly based on Ara patterns

// Scalar implementation for verification
void relu_scalar(float *output, const float *input, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// Vectorized ReLU activation using proper RISC-V vector instructions
void relu_vec(float *output, const float *input, size_t size) {
    size_t avl = size;
    const float* input_ = input;
    float* output_ = output;
    
    for (; avl > 0;) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
        
        // Load input vector
        asm volatile("vle32.v v8, (%0)" :: "r"(input_));
        
        // Load zero vector for comparison
        asm volatile("vmv.v.i v16, 0");
        
        // Create mask: v8 > 0.0f using zero register
        float zero = 0.0f;
        asm volatile("vmfgt.vf v0, v8, %0" :: "f"(zero));
        
        // Select: output = (mask) ? input : 0.0f
        asm volatile("vmerge.vvm v24, v16, v8, v0");
        
        // Store result
        asm volatile("vse32.v v24, (%0)" :: "r"(output_));
        
        // Update pointers and remaining elements
        input_ += vl;
        output_ += vl;
        avl -= vl;
    }
    
    /* TODO: Implement proper vectorized version when riscv_vector.h is available
    size_t avl = size;
    size_t vl;
    
    for (size_t i = 0; i < size; i += vl) {
        avl = size - i;
        vl = __riscv_vsetvl_e32m4(avl);
        
        // Load input vector
        vfloat32m4_t input_v = __riscv_vle32_v_f32m4(input + i, vl);
        
        // Create zero vector for comparison
        vfloat32m4_t zero_v = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        
        // Compute ReLU: max(input, 0)
        vfloat32m4_t output_v = __riscv_vfmax_vv_f32m4(input_v, zero_v, vl);
        
        // Store result
        __riscv_vse32_v_f32m4(output + i, output_v, vl);
    }
    */
}

// In-place ReLU (for use in pipeline)
void relu_inplace(float *data, size_t size) {
    // Fallback to scalar implementation
    for (size_t i = 0; i < size; i++) {
        data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
    }
    
    /* TODO: Implement proper vectorized version when riscv_vector.h is available
    size_t avl = size;
    size_t vl;
    
    for (size_t i = 0; i < size; i += vl) {
        avl = size - i;
        vl = __riscv_vsetvl_e32m4(avl);
        
        vfloat32m4_t data_v = __riscv_vle32_v_f32m4(data + i, vl);
        vfloat32m4_t zero_v = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t result_v = __riscv_vfmax_vv_f32m4(data_v, zero_v, vl);
        __riscv_vse32_v_f32m4(data + i, result_v, vl);
    }
    */
}