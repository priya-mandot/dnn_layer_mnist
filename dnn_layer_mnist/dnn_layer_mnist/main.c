// Copyright 2025
//
// SPDX-License-Identifier: Apache-2.0
//
// MNIST DNN Layer Test Program for CVA6-Ara2

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

// Print first N values of an array
static void print_array(const char *name, const float *arr, size_t size, size_t limit) {
    printf("%s: ", name);
    size_t print_count = (size < limit) ? size : limit;
    for (size_t i = 0; i < print_count; i++) {
        printf("%.6f ", arr[i]);
    }
    if (size > limit) printf("...");
    printf("\n");
}

// Compare two arrays and print differences
static int compare_arrays(const float *arr1, const float *arr2, size_t size, const char *name) {
    int errors = 0;
    float max_diff = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = fabsf(arr1[i] - arr2[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > THRESHOLD && errors < 3) {
            printf("  Diff at [%zu]: scalar=%.6f, vector=%.6f, diff=%.6f\n", 
                   i, arr1[i], arr2[i], diff);
            errors++;
        }
    }
    if (errors > 3) printf("  ... and %d more differences\n", errors - 3);
    printf("%s max difference: %.6f\n", name, max_diff);
    return errors;
}

// Verification function
static int verify_result(const float *result, const float *gold, size_t size, float threshold) {
    int errors = 0;
    float max_error = 0.0f;
    float sum_result = 0.0f, sum_gold = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
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
static int validate_softmax(const float *output, size_t size) {
    float sum = 0.0f;
    int negative_count = 0;
    float min_val = output[0], max_val = output[0];
    
    for (size_t i = 0; i < size; i++) {
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
static void analyze_prediction(const float *output, size_t size) {
    size_t predicted_class = 0;
    float max_confidence = output[0];
    
    for (size_t i = 1; i < size; i++) {
        if (output[i] > max_confidence) {
            max_confidence = output[i];
            predicted_class = i;
        }
    }
    
    printf("Prediction analysis:\n");
    printf("  Predicted class: %zu\n", predicted_class);
    printf("  Confidence: %.3f%%\n", max_confidence * 100.0f);
}

int main() {
    printf("MNIST DNN LAYER TEST\n");
    printf("====================\n");
    
    // Setup configuration
    printf("batch_size: %lu, input_size: %lu, output_size: %lu\n", 
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
    printf("Configuration: %lux%lu -> %lu\n\n", batch_size, input_size, output_size);
    
    // Run scalar implementation
    printf("===== SCALAR IMPLEMENTATION =====\n");
    printf("Running SCALAR Implementation...\n");
    start_timer();
    dnn_layer_forward_scalar(output_scalar, input_data, &params, &config, temp_buffer);
    stop_timer();
    int64_t scalar_cycles = get_timer();
    printf("Scalar execution completed: %ld cycles\n\n", scalar_cycles);
    
    printf("Scalar output (first 10 values):\n");
    print_array("  ", output_scalar, output_size, 10);
    printf("\n");
    
    // Validate scalar result
    printf("Verifying scalar result against gold...\n");
    int scalar_errors = verify_result(output_scalar, gold_output, output_size, THRESHOLD);
    printf("\n");
    validate_softmax(output_scalar, output_size);
    printf("\n");
    analyze_prediction(output_scalar, output_size);
    printf("Scalar implementation: %s\n\n", (scalar_errors <= 5) ? "PASSED" : "FAILED");
    
    // Run vectorized implementation
    printf("===== VECTORIZED IMPLEMENTATION =====\n");
    printf("Running VECTORIZED Implementation...\n");
    start_timer();
    dnn_layer_forward(output_vec, input_data, &params, &config, temp_buffer);
    stop_timer();
    int64_t vector_cycles = get_timer();
    printf("Vector execution completed: %ld cycles\n\n", vector_cycles);
    
    printf("Vector output (first 10 values):\n");
    print_array("  ", output_vec, output_size, 10);
    printf("\n");
    
    // Validate vector result
    printf("Verifying vectorized result against gold...\n");
    int vector_errors = verify_result(output_vec, gold_output, output_size, THRESHOLD);
    printf("\n");
    validate_softmax(output_vec, output_size);
    printf("\n");
    analyze_prediction(output_vec, output_size);
    printf("Vectorized implementation: %s\n\n", (vector_errors <= 5) ? "PASSED" : "FAILED");
    
    // Compare scalar vs vector layer by layer
    printf("===== LAYER-BY-LAYER COMPARISON =====\n");
    printf("Comparing Scalar vs Vectorized Implementations...\n\n");
    
    printf("MatMul Output Comparison:\n");
    compare_arrays(matmul_out_scalar, matmul_out_vec, output_size, "MatMul");
    printf("\n");
    
    printf("BatchNorm Output Comparison:\n");
    compare_arrays(bn_out_scalar, bn_out_vec, output_size, "BatchNorm");
    printf("\n");
    
    printf("ReLU Output Comparison:\n");
    compare_arrays(relu_out_scalar, relu_out_vec, output_size, "ReLU");
    printf("\n");
    
    printf("Final Output Comparison:\n");
    int consistency_errors = verify_result(output_vec, output_scalar, output_size, THRESHOLD);
    printf("Implementation consistency: %s\n\n", (consistency_errors <= 5) ? "PASSED" : "FAILED");
    
    // Performance summary
    printf("===== PERFORMANCE SUMMARY =====\n");
    if (vector_cycles > 0 && scalar_cycles > 0) {
        double speedup = (double)scalar_cycles / (double)vector_cycles;
        printf("Scalar cycles: %lu\n", scalar_cycles);
        printf("Vector cycles: %lu\n", vector_cycles);
        printf("Measured speedup: %.2fx\n\n", speedup);
    }
    
    // Final result
    int overall_pass = (scalar_errors <= 5) && (vector_errors <= 5) && (consistency_errors <= 5);
    printf("%s\n", overall_pass ? "MNIST DNN LAYER TEST PASSED" : "MNIST DNN LAYER TEST FAILED");
    
    return overall_pass ? 0 : 1;
}