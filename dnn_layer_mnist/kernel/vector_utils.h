#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

#include <stdint.h>
#include <stddef.h>

/*
 * RISC-V Vector Operations for CVA6-Ara2 DNN Layer
 * Based on Ara vector programming patterns from fmatmul, dotproduct, etc.
 */

// Configuration
#ifndef VLEN
#define VLEN 4096
#endif

#ifndef NR_LANES
#define NR_LANES 4
#endif

// Vector configuration helpers based on Ara apps patterns
static inline size_t ara_vsetvl_f32m8(size_t avl) {
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
    return vl;
}

static inline size_t ara_vsetvl_f32m4(size_t avl) {
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(avl));
    return vl;
}

static inline size_t ara_vsetvl_f32m2(size_t avl) {
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(vl) : "r"(avl));
    return vl;
}

static inline size_t ara_vsetvl_f32m1(size_t avl) {
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(avl));
    return vl;
}

// Vector element-wise operations based on Ara patterns
static inline void ara_vadd_f32(float* dst, const float* src1, const float* src2, size_t n) {
    size_t avl = n;
    const float* a_ = src1;
    const float* b_ = src2;
    float* c_ = dst;
    
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vectors
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));
        asm volatile("vle32.v v16, (%0)" :: "r"(b_));
        
        // Add vectors
        asm volatile("vfadd.vv v24, v8, v16");
        
        // Store result
        asm volatile("vse32.v v24, (%0)" :: "r"(c_));
        
        // Update pointers and remaining elements
        a_ += vl;
        b_ += vl;
        c_ += vl;
        avl -= vl;
    }
}

static inline void ara_vmul_f32(float* dst, const float* src1, const float* src2, size_t n) {
    size_t avl = n;
    const float* a_ = src1;
    const float* b_ = src2;
    float* c_ = dst;
    
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vectors
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));
        asm volatile("vle32.v v16, (%0)" :: "r"(b_));
        
        // Multiply vectors
        asm volatile("vfmul.vv v24, v8, v16");
        
        // Store result
        asm volatile("vse32.v v24, (%0)" :: "r"(c_));
        
        // Update pointers and remaining elements
        a_ += vl;
        b_ += vl;
        c_ += vl;
        avl -= vl;
    }
}

static inline void ara_vfmacc_f32(float* dst, const float* src1, const float* src2, size_t n) {
    // dst += src1 * src2 (fused multiply-accumulate)
    size_t avl = n;
    const float* a_ = src1;
    const float* b_ = src2;
    float* c_ = dst;
    
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vectors
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));   // src1
        asm volatile("vle32.v v16, (%0)" :: "r"(b_));  // src2
        asm volatile("vle32.v v24, (%0)" :: "r"(c_));  // dst (accumulator)
        
        // Fused multiply-accumulate: dst = dst + src1 * src2
        asm volatile("vfmacc.vv v24, v8, v16");
        
        // Store result
        asm volatile("vse32.v v24, (%0)" :: "r"(c_));
        
        // Update pointers and remaining elements
        a_ += vl;
        b_ += vl;
        c_ += vl;
        avl -= vl;
    }
}

static inline void ara_vscale_f32(float* dst, const float* src, float scale, size_t n) {
    // dst = src * scale (broadcast scalar multiplication)
    size_t avl = n;
    const float* a_ = src;
    float* c_ = dst;
    
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vector
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));
        
        // Scale by scalar: v24 = v8 * scale
        asm volatile("vfmul.vf v24, v8, %0" :: "f"(scale));
        
        // Store result
        asm volatile("vse32.v v24, (%0)" :: "r"(c_));
        
        // Update pointers and remaining elements
        a_ += vl;
        c_ += vl;
        avl -= vl;
    }
}

// Vector reduction operations based on dotproduct and vfredsum patterns
static inline float ara_vredsum_f32(const float* src, size_t n) {
    // Based on dotproduct.c pattern for reductions
    size_t avl = n;
    const float* a_ = src;
    float result;
    
    // Initialize accumulator to zero
    asm volatile("vsetvli zero, zero, e32, m1, ta, ma");
    asm volatile("vmv.s.x v0, zero");
    
    // Process elements in chunks
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vector chunk
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));
        
        // Reduce sum: accumulate into v0
        asm volatile("vfredusum.vs v0, v8, v0");
        
        // Update pointer and remaining elements
        a_ += vl;
        avl -= vl;
    }
    
    // Extract final result from v0
    asm volatile("vfmv.f.s %0, v0" : "=f"(result));
    return result;
}

static inline float ara_vredmax_f32(const float* src, size_t n) {
    if (n == 0) return 0.0f;
    
    size_t avl = n;
    const float* a_ = src;
    float result;
    
    // Initialize accumulator with first element
    asm volatile("vsetvli zero, zero, e32, m1, ta, ma");
    asm volatile("vfmv.s.f v0, %0" :: "f"(src[0]));
    
    // Process elements in chunks
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vector chunk
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));
        
        // Reduce max: find maximum and accumulate into v0
        asm volatile("vfredmax.vs v0, v8, v0");
        
        // Update pointer and remaining elements
        a_ += vl;
        avl -= vl;
    }
    
    // Extract final result from v0
    asm volatile("vfmv.f.s %0, v0" : "=f"(result));
    return result;
}

// Dot product implementation based on dotproduct.c
static inline float ara_vdot_f32(const float* a, const float* b, size_t n) {
    size_t avl = n;
    const float* a_ = a;
    const float* b_ = b;
    float result;
    
    // Initialize accumulator to zero
    asm volatile("vsetvli zero, zero, e32, m1, ta, ma");
    asm volatile("vmv.s.x v0, zero");
    
    // Process elements in chunks
    for (; avl > 0;) {
        size_t vl = ara_vsetvl_f32m8(avl);
        
        // Load vector chunks
        asm volatile("vle32.v v8, (%0)" :: "r"(a_));
        asm volatile("vle32.v v16, (%0)" :: "r"(b_));
        
        // Multiply vectors
        asm volatile("vfmul.vv v24, v8, v16");
        
        // Reduce sum: accumulate into v0
        asm volatile("vfredusum.vs v0, v24, v0");
        
        // Update pointers and remaining elements
        a_ += vl;
        b_ += vl;
        avl -= vl;
    }
    
    // Extract final result from v0
    asm volatile("vfmv.f.s %0, v0" : "=f"(result));
    return result;
}

// Matrix operations based on fmatmul.c patterns
static inline void ara_matmul_f32(
    float* output,           // [m x n] 
    const float* input,      // [m x k]
    const float* weights,    // [k x n] (row-major)
    size_t m,               // batch size 
    size_t k,               // input size
    size_t n                // output size
) {
    // Based on fmatmul patterns - vectorize along output dimension
    size_t avl_n = n;
    
    // Process output columns in vector chunks
    for (size_t n_start = 0; n_start < n; n_start += avl_n) {
        size_t vl = ara_vsetvl_f32m8(n - n_start);
        avl_n = vl;
        
        // Process each output row
        for (size_t i = 0; i < m; i++) {
            // Initialize accumulator to zero
            asm volatile("vmv.v.i v24, 0");
            
            // Compute dot product for this row
            for (size_t j = 0; j < k; j++) {
                float input_elem = input[i * k + j];
                
                // Load weight row starting at weights[j * n + n_start]
                const float* weight_ptr = weights + j * n + n_start;
                asm volatile("vle32.v v8, (%0)" :: "r"(weight_ptr));
                
                // Multiply-accumulate: v24 += input_elem * v8
                asm volatile("vfmacc.vf v24, %0, v8" :: "f"(input_elem));
            }
            
            // Store result
            float* output_ptr = output + i * n + n_start;
            asm volatile("vse32.v v24, (%0)" :: "r"(output_ptr));
        }
    }
}

// Optimized GEMV (General Matrix-Vector multiplication) for single batch
static inline void ara_gemv_f32(
    float* output,           // [n] output vector
    const float* input,      // [k] input vector  
    const float* weights,    // [k x n] weight matrix
    size_t k,               // input size
    size_t n                // output size
) {
    // Vectorize along output dimension for better efficiency
    size_t avl = n;
    
    for (size_t n_start = 0; n_start < n; n_start += avl) {
        size_t vl = ara_vsetvl_f32m8(n - n_start);
        avl = vl;
        
        // Initialize accumulator to zero
        asm volatile("vmv.v.i v24, 0");
        
        // Compute dot products
        for (size_t j = 0; j < k; j++) {
            float input_elem = input[j];
            
            // Load weight row: weights[j * n + n_start : j * n + n_start + vl]
            const float* weight_ptr = weights + j * n + n_start;
            asm volatile("vle32.v v8, (%0)" :: "r"(weight_ptr));
            
            // Multiply-accumulate: v24 += input_elem * v8
            asm volatile("vfmacc.vf v24, %0, v8" :: "f"(input_elem));
        }
        
        // Store result
        float* output_ptr = output + n_start;
        asm volatile("vse32.v v24, (%0)" :: "r"(output_ptr));
    }
}

// Fallback compatibility aliases for gradual migration
#define vector_add_f32 ara_vadd_f32
#define vector_mul_f32 ara_vmul_f32
#define vector_scale_f32 ara_vscale_f32
#define vector_reduce_sum_f32 ara_vredsum_f32
#define vector_reduce_max_f32 ara_vredmax_f32
#define vector_matmul_f32 ara_matmul_f32

#endif // VECTOR_UTILS_H
