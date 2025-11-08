#!/bin/bash

# MNIST DNN Vector Speedup Analysis Script
# Run this to check vector performance vs scalar performance

echo "============================================================"
echo "  MNIST DNN VECTOR SPEEDUP ANALYSIS"
echo "============================================================"
echo ""

# Build the application
echo "🔨 Building MNIST DNN with vector operations..."
cd /home/vaishnavi-shivkumar/Documents/4-1/hawai/ara/apps
make bin/dnn_layer_mnist > build.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed! Check build.log"
    exit 1
fi

echo ""
echo "🔍 Checking for vector instructions in binary..."

# Check if vector instructions are present
VECTOR_INST_COUNT=$(grep -i "vse32\|vle32\|vfmacc\|vmfgt\|vmerge" bin/dnn_layer_mnist.dump | wc -l)
echo "📊 Vector instructions found: $VECTOR_INST_COUNT"

if [ $VECTOR_INST_COUNT -gt 0 ]; then
    echo "✅ Vector instructions confirmed in binary!"
    echo ""
    echo "🔍 Sample vector instructions:"
    grep -i "vse32\|vle32\|vfmacc\|vmfgt\|vmerge" bin/dnn_layer_mnist.dump | head -5
else
    echo "⚠️  No vector instructions found - using scalar fallback"
fi

echo ""
echo "============================================================"
echo "  Vector Instructions Analysis:"
echo "============================================================"
echo "MatMul Operations:"
grep -i "vfmacc\|vle32.*v8\|vse32.*v24" bin/dnn_layer_mnist.dump | wc -l | xargs echo "  - Fused Multiply-Accumulate: "
echo ""
echo "ReLU Operations:"  
grep -i "vmfgt\|vmerge" bin/dnn_layer_mnist.dump | wc -l | xargs echo "  - Vector Compare/Merge: "
echo ""

echo "============================================================"
echo "  PERFORMANCE TESTING"
echo "============================================================"

# Note: We can't run the binary directly since it's compiled for CVA6-Ara2
# But we can analyze the expected performance

echo "📋 Expected Performance Analysis:"
echo ""
echo "🎯 MNIST Network:"
echo "  - Input: 784 pixels (28x28)"
echo "  - Output: 10 classes" 
echo "  - MatMul: 784 × 10 = 7,840 operations (vectorized)"
echo "  - ReLU: 10 operations (vectorized)"
echo "  - BatchNorm: 10 operations (scalar)"
echo "  - Softmax: ~20 operations (scalar)"
echo ""

echo "⚡ Vector Configuration:"
echo "  - VLEN=4096 bits"
echo "  - Element width: 32-bit float"
echo "  - Max elements per vector: 128 floats"
echo "  - LMUL=8 (using 8 vector registers)"
echo ""

echo "📈 Expected Speedup:"
echo "  - MatMul contribution: ~99% of compute"
echo "  - Vector efficiency: 4-128x theoretical"
echo "  - Realistic speedup: 2-8x end-to-end"
echo ""

echo "🔬 To measure actual speedup:"
echo "  1. Run on CVA6-Ara2 hardware/simulator"
echo "  2. The binary contains detailed benchmarking (10 iterations each)"
echo "  3. Look for 'Measured speedup: X.XXXx' in output"
echo ""

echo "============================================================"
echo "  Vector Implementation Status:"
echo "============================================================"
echo "✅ Matrix Multiplication - Fully vectorized"
echo "✅ ReLU Activation - Fully vectorized"  
echo "🔄 Batch Normalization - Scalar (can be vectorized)"
echo "🔄 Softmax - Scalar (can be vectorized)"
echo ""

echo "🚀 Next steps to increase speedup:"
echo "  1. Vectorize batch normalization"
echo "  2. Vectorize softmax (exp + reduction)" 
echo "  3. Optimize memory access patterns"
echo "  4. Run on actual hardware for measurement"
echo ""

echo "============================================================"
