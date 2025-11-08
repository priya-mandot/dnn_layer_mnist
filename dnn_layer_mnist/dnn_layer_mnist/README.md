# Ara MNIST DNN Layer Test

MNIST DNN layer implementation with RISC-V vector acceleration for the CVA6-Ara2 processor.

## Overview

This application demonstrates a complete neural network forward pass using both scalar and vectorized implementations. The network processes MNIST handwritten digit data through matrix multiplication, batch normalization, ReLU activation, and softmax output layers.

## Features

- Complete DNN pipeline: MatMul → BatchNorm → ReLU → Softmax
- RISC-V vector acceleration using inline assembly
- Performance comparison between scalar and vectorized implementations
- Automatic verification against gold reference data
- Support for CVA6-Ara2 vector processor (VLEN=4096, 4 lanes)

## Vector Acceleration

The vectorized implementation uses RISC-V vector instructions:
- MatMul: `vfmacc.vf` (fused multiply-accumulate)
- BatchNorm: `vfadd.vv`, `vfsub.vv`, `vfmul.vv`, `vfdiv.vv`, `vfsqrt.v`
- ReLU: `vmfgt.vf` (vector mask for thresholding), `vfmax.vv`
- Softmax: `vfsub.vf`, `vfdiv.vf` (uses scalar operations for stability)

Multiple vector instruction implementations available (inline assembly + intrinsics)

## Build and Run

### Building
```bash
cd apps/dnn_layer_mnist
python3 script/gen_data.py > data.S  # Generate training data
cd ..
make bin/dnn_layer_mnist
```

### Running on Spike
```bash
spike --isa=rv64gcv_zfh_zvfh bin/dnn_layer_mnist
```

### Running on Hardware
```bash
./run_ara.sh
```

## Expected Results

- Training accuracy improvement from ~10% to ~95% over epochs
- Vector speedup of approximately 2-8x over scalar implementation
- Successful verification against gold reference data (≤5 errors per check)
- Valid softmax probability distribution (sum ≈ 1.0, all values ≥ 0)
- Correct digit classification with confidence scores

## File Structure

- `main.c`: Test program with scalar vs vector comparison
- `kernel/dnn_layer.h`: DNN layer interface definitions
- `kernel/dnn_layer.c`: Main DNN layer implementation
- `kernel/fmatmul.c`: Matrix multiplication kernels
- `kernel/batch_norm.c`: Batch normalization kernels  
- `kernel/relu.c`: ReLU activation kernels
- `kernel/softmax.c`: Softmax kernels
- `kernel/vector_utils.h`: Vector utility functions
- `script/gen_data.py`: Training data generator
- `data.S`: Generated input data and trained parameters
- `Makefile`: Build configuration

## Performance Metrics

The application reports:
- Execution cycles for both implementations
- Verification results with error counts
- Softmax validation (sum and range checks)
- Predicted digit class and confidence
- Measured speedup factor (vector vs scalar) 

## Verification

The test automatically verifies:
1. Scalar implementation against gold reference
2. Vector implementation against gold reference
3. Consistency between scalar and vector outputs
4. Softmax mathematical properties

Test passes with ≤5 errors per verification check (within 1% threshold).

## Implementation Notes

- All data structures are aligned to vector lane boundaries
- Vector operations handle arbitrary input sizes with tail handling
- Inline assembly ensures optimal vector instruction generation
- Comprehensive error checking and performance measurement included

This implementation demonstrates successful integration of RISC-V vector extensions with neural network workloads on the Ara vector processor.
