// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// MNIST DNN Layer Test Program for CVA6-Ara2 - With Detailed Timing

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "kernel/dnn_layer.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Configuration from mnist_data.S
extern uint64_t batch_size;
extern uint64_t input_size;
extern uint64_t output_size;
extern float bn_epsilon;

// Input data and parameters
extern float input_data[] __attribute__((aligned(4 * NR_LANES)));
extern float weights[] __attribute__((aligned(4 * NR_LANES)));
extern float bn_gamma[] __attribute__((aligned(4 * NR_LANES)));
extern float bn_beta[] __attribute__((aligned(4 * NR_LANES)));
extern float bn_mean[] __attribute__((aligned(4 * NR_LANES)));
extern float bn_var[] __attribute__((aligned(4 * NR_LANES)));

// Output buffers
extern float output_vec[] __attribute__((aligned(4 * NR_LANES)));
extern float output_scalar[] __attribute__((aligned(4 * NR_LANES)));
extern float temp_buffer[] __attribute__((aligned(4 * NR_LANES)));

// Intermediate buffers for debugging
extern float matmul_out_scalar[] __attribute__((aligned(4 * NR_LANES)));
extern float matmul_out_vec[] __attribute__((aligned(4 * NR_LANES)));
extern float bn_out_scalar[] __attribute__((aligned(4 * NR_LANES)));
extern float bn_out_vec[] __attribute__((aligned(4 * NR_LANES)));
extern float relu_out_scalar[] __attribute__((aligned(4 * NR_LANES)));
extern float relu_out_vec[] __attribute__((aligned(4 * NR_LANES)));

// Gold reference
extern float gold_output[] __attribute__((aligned(4 * NR_LANES)));

#define THRESHOLD 0.01f

// Timing structure for layer profiling
// typedef struct {
//     int64_t matmul_cycles;
//     int64_t batchnorm_cycles;
//     int64_t relu_cycles;
//     int64_t softmax_cycles;
//     int64_t total_cycles;
// } layer_timing_t;

// Helper function to print first 5 values
static void print_first_5(const char *label, const float *arr) {
    printf("%s: ", label);
    for (int i = 0; i < 5 && i < output_size; i++) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}

// Print first N values of an array
static void print_array(const char *name, const float *arr, unsigned long int size, unsigned long int limit) {
    printf("%s: ", name);
    unsigned long int print_count = (size < limit) ? size : limit;
    for (unsigned long int i = 0; i < print_count; i++) {
        printf("%.6f ", arr[i]);
    }
    if (size > limit) printf("...");
    printf("\n");
}

// Verification function
static int verify_result(const float *result, const float *gold, unsigned long int size, float threshold) {
    int errors = 0;
    float max_error = 0.0f;
    float sum_result = 0.0f, sum_gold = 0.0f;
    
    for (unsigned long int i = 0; i < size; i++) {
        sum_result += result[i];
        sum_gold += gold[i];
        
        float diff = fabsf(result[i] - gold[i]);
        float rel_error = (gold[i] != 0.0f) ? (diff / fabsf(gold[i])) : diff;
        
        if (rel_error > threshold) {
            if (errors < 3) {
                printf("Error at index %zu: got %f, expected %f (rel_error: %f)\n",
                       i, result[i], gold[i], rel_error);
            }
            errors++;
            if (diff > max_error) {
                max_error = diff;
            }
        }
    }
    
    printf("Verification stats:\n");
    printf("  Result sum: %f, Gold sum: %f\n", sum_result, sum_gold);
    printf("  Max error: %f, Errors: %d/%zu\n", max_error, errors, size);
    
    return errors;
}

// Validate softmax properties
static int validate_softmax(const float *output, unsigned long int size) {
    float sum = 0.0f;
    int negative_count = 0;
    float min_val = output[0], max_val = output[0];
    
    for (unsigned long int i = 0; i < size; i++) {
        sum += output[i];
        if (output[i] < 0.0f) negative_count++;
        if (output[i] < min_val) min_val = output[i];
        if (output[i] > max_val) max_val = output[i];
    }
    
    printf("Softmax properties:\n");
    printf("  Sum: %f (should be ~1.0)\n", sum);
    printf("  Range: [%f, %f]\n", min_val, max_val);
    printf("  Negative values: %d (should be 0)\n", negative_count);
    
    int valid = (fabsf(sum - 1.0f) < 0.1f && negative_count == 0) ? 0 : 1;
    return valid;
}

// Find predicted class and confidence
static void analyze_prediction(const float *output, unsigned long int size) {
    unsigned long int predicted_class = 0;
    float max_confidence = output[0];
    
    for (unsigned long int i = 1; i < size; i++) {
        if (output[i] > max_confidence) {
            max_confidence = output[i];
            predicted_class = i;
        }
    }
    
    printf("Prediction analysis:\n");
    printf("  Predicted class: %zu\n", predicted_class);
    printf("  Confidence: %.3f%%\n", max_confidence * 100.0f);
}

// Print detailed timing breakdown
static void print_timing_breakdown(const char *label, const layer_timing_t *timing) {
    printf("\n%s TIMING BREAKDOWN:\n", label);
    // printf("=" * 50);
    printf("\n");
    printf("  MatMul      : %6ld cycles (%5.1f%%)\n", 
           timing->matmul_cycles, 
           100.0 * timing->matmul_cycles / timing->total_cycles);
    printf("  BatchNorm   : %6ld cycles (%5.1f%%)\n", 
           timing->batchnorm_cycles,
           100.0 * timing->batchnorm_cycles / timing->total_cycles);
    printf("  ReLU        : %6ld cycles (%5.1f%%)\n", 
           timing->relu_cycles,
           100.0 * timing->relu_cycles / timing->total_cycles);
    printf("  Softmax     : %6ld cycles (%5.1f%%)\n", 
           timing->softmax_cycles,
           100.0 * timing->softmax_cycles / timing->total_cycles);
    // printf("  " "-" * 35);
    double total_cycles = (double)timing->matmul_cycles + timing->batchnorm_cycles +
                           timing->relu_cycles + timing->softmax_cycles;
    printf("\n");
    printf("  TOTAL       : %6ld cycles\n", total_cycles);
    printf("\n");
}

// Compare timing between scalar and vector
static void compare_timing(const layer_timing_t *scalar, const layer_timing_t *vector) {
    printf("\n");
    // printf("=" * 70);
    printf("\n");
    printf("SPEEDUP ANALYSIS\n");
    // printf("=" * 70);
    printf("\n");
    printf("Layer          | Scalar    | Vector    | Speedup  | Notes\n");
    // printf("-" * 70);
    printf("\n");
    
    double matmul_speedup = (double)scalar->matmul_cycles / vector->matmul_cycles;
    double bn_speedup = (double)scalar->batchnorm_cycles / vector->batchnorm_cycles;
    double relu_speedup = (double)scalar->relu_cycles / vector->relu_cycles;
    double softmax_speedup = (double)scalar->softmax_cycles / vector->softmax_cycles;
    double vector_total_cycles =(double)vector->matmul_cycles + vector->batchnorm_cycles +
                           vector->relu_cycles + vector->softmax_cycles;
    double total_speedup = (double)scalar->total_cycles / vector_total_cycles;

    printf("MatMul         | %9ld | %9ld | %6.2fx | Vectorized (strided load)\n", 
           scalar->matmul_cycles, vector->matmul_cycles, matmul_speedup);
    printf("BatchNorm      | %9ld | %9ld | %6.2fx | Scalar (both)\n", 
           scalar->batchnorm_cycles, vector->batchnorm_cycles, bn_speedup);
    printf("ReLU           | %9ld | %9ld | %6.2fx | Vectorized (mask)\n", 
           scalar->relu_cycles, vector->relu_cycles, relu_speedup);
    printf("Softmax        | %9ld | %9ld | %6.2fx | Vectorized (partial)\n", 
           scalar->softmax_cycles, vector->softmax_cycles, softmax_speedup);
    // printf("-" * 70);
    printf("\n");
    printf("TOTAL          | %9ld | %9ld | %6.2fx |\n", 
           scalar->total_cycles, vector_total_cycles, total_speedup);
    printf("\n");
}

int main() {
    printf("MNIST DNN LAYER TEST - DETAILED TIMING\n");
    // printf("=" * 70);
    printf("\n");
    
    // Setup configuration
    printf("Configuration: batch=%lu, input=%lu, output=%lu\n\n", 
           batch_size, input_size, output_size);
    
    dnn_config_t config = {
        .batch_size = batch_size,
        .input_size = input_size,
        .output_size = output_size
    };
    
    dnn_params_t params = {
        .weights = weights,
        .bn_gamma = bn_gamma,
        .bn_beta = bn_beta,
        .bn_mean = bn_mean,
        .bn_var = bn_var,
        .bn_epsilon = bn_epsilon
    };
    
    layer_timing_t scalar_timing = {0};
    layer_timing_t vector_timing = {0};
    
    // ========== SCALAR IMPLEMENTATION ==========
    printf("===== SCALAR IMPLEMENTATION =====\n");
    
    start_timer();
    dnn_layer_forward_scalar_timed(output_scalar, input_data, &params, &config, 
                                   temp_buffer, &scalar_timing);
    stop_timer();
    scalar_timing.total_cycles = get_timer();
    
    printf("\nScalar layer outputs:\n");
    print_first_5("  After MatMul   ", matmul_out_scalar);
    print_first_5("  After BatchNorm", bn_out_scalar);
    print_first_5("  After ReLU     ", relu_out_scalar);
    print_first_5("  After Softmax  ", output_scalar);
    
    print_timing_breakdown("SCALAR", &scalar_timing);
    
    printf("\nVerifying scalar result against gold...\n");
    int scalar_errors = verify_result(output_scalar, gold_output, output_size, THRESHOLD);
    printf("\n");
    validate_softmax(output_scalar, output_size);
    printf("\n");
    analyze_prediction(output_scalar, output_size);
    printf("Scalar implementation: %s\n", (scalar_errors <= 5) ? "PASSED" : "FAILED");
    
    // ========== VECTORIZED IMPLEMENTATION ==========
    printf("\n===== VECTORIZED IMPLEMENTATION =====\n");
    
    start_timer();
    dnn_layer_forward_timed(output_vec, input_data, &params, &config, 
                           temp_buffer, &vector_timing);
    stop_timer();
    vector_timing.total_cycles = get_timer();
    
    printf("\nVector layer outputs:\n");
    print_first_5("  After MatMul   ", matmul_out_vec);
    print_first_5("  After BatchNorm", bn_out_vec);
    print_first_5("  After ReLU     ", relu_out_vec);
    print_first_5("  After Softmax  ", output_vec);
    
    print_timing_breakdown("VECTOR", &vector_timing);
    
    printf("\nVerifying vectorized result against gold...\n");
    int vector_errors = verify_result(output_vec, gold_output, output_size, THRESHOLD);
    printf("\n");
    validate_softmax(output_vec, output_size);
    printf("\n");
    analyze_prediction(output_vec, output_size);
    printf("Vectorized implementation: %s\n", (vector_errors <= 5) ? "PASSED" : "FAILED");
    
    // ========== COMPARISON ==========
    compare_timing(&scalar_timing, &vector_timing);
    
    // Final result
    int overall_pass = (scalar_errors <= 5) && (vector_errors <= 5);
    printf("%s\n", overall_pass ? "MNIST DNN LAYER TEST PASSED" : "MNIST DNN LAYER TEST FAILED");
    
    return overall_pass ? 0 : 1;
}