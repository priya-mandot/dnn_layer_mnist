#!/bin/bash

# Quick Start Script for Running MNIST DNN on Ara
# Usage: ./run_ara.sh [sim_type]
# sim_type: simv (default), spike, sim, ideal

set -e  # Exit on any error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ARA_ROOT="/home/vaishnavi-shivkumar/Documents/4-1/hawai/ara"
SIM_TYPE=${1:-simv}  # Default to Verilator simulation

echo -e "${BLUE}🚀 Ara MNIST DNN Runner${NC}"
echo -e "${BLUE}========================${NC}"
echo "Simulation Type: $SIM_TYPE"
echo "Ara Root: $ARA_ROOT"
echo ""

# Step 1: Build the application
echo -e "${YELLOW}📦 Building MNIST DNN application...${NC}"
cd "$ARA_ROOT/apps"

case $SIM_TYPE in
    "spike")
        echo "Building for Spike ISA simulator..."
        make bin/dnn_layer_mnist.spike
        ;;
    "ideal")
        echo "Building for ideal dispatcher mode..."
        make bin/dnn_layer_mnist.ideal
        ;;
    *)
        echo "Building for RTL simulation..."
        make bin/dnn_layer_mnist
        ;;
esac

echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo ""

# Step 2: Verify vector instructions
echo -e "${YELLOW}🔍 Checking vector instruction generation...${NC}"
if [[ $SIM_TYPE == "spike" ]]; then
    BINARY="bin/dnn_layer_mnist.spike"
elif [[ $SIM_TYPE == "ideal" ]]; then
    BINARY="bin/dnn_layer_mnist.ideal"
else
    BINARY="bin/dnn_layer_mnist"
fi

# Use the RISC-V toolchain's objdump that understands vector instructions
RISCV_OBJDUMP="$ARA_ROOT/install/riscv-llvm/bin/llvm-objdump"
if [ ! -f "$RISCV_OBJDUMP" ]; then
    RISCV_OBJDUMP="$ARA_ROOT/install/riscv-gcc/bin/riscv64-unknown-elf-objdump"
fi

# Check if we already have the dump file (generated during build)
if [ -f "$BINARY.dump" ]; then
    echo "Using existing disassembly: $BINARY.dump"
    VECTOR_INST_COUNT=$(grep -E "(vle32|vse32|vfmacc|vmfgt)" $BINARY.dump | wc -l)
else
    echo "Generating disassembly with RISC-V objdump..."
    VECTOR_INST_COUNT=$($RISCV_OBJDUMP --mattr=v -d $BINARY | grep -E "(vle32|vse32|vfmacc|vmfgt)" | wc -l)
fi

echo "Vector instructions found: $VECTOR_INST_COUNT"

if [ $VECTOR_INST_COUNT -gt 0 ]; then
    echo -e "${GREEN}✅ Vector instructions present in binary${NC}"
    echo "Sample vector instructions:"
    if [ -f "$BINARY.dump" ]; then
        grep -E "(vle32|vse32|vfmacc|vmfgt)" $BINARY.dump | head -3
    else
        $RISCV_OBJDUMP --mattr=v -d $BINARY | grep -E "(vle32|vse32|vfmacc|vmfgt)" | head -3
    fi
else
    echo -e "${RED}❌ No vector instructions found! Check compilation.${NC}"
    exit 1
fi
echo ""

# Step 3: Run simulation
echo -e "${YELLOW}🎯 Running $SIM_TYPE simulation...${NC}"

case $SIM_TYPE in
    "spike")
        echo "Starting Spike ISA simulation..."
        make spike-run-dnn_layer_mnist
        ;;
    "sim")
        echo "Starting ModelSim/QuestaSim RTL simulation..."
        cd "$ARA_ROOT/hardware"
        make sim app=dnn_layer_mnist
        ;;
    "ideal")
        echo "Starting ideal dispatcher simulation..."
        cd "$ARA_ROOT/hardware"
        make sim app=dnn_layer_mnist ideal_dispatcher=1
        ;;
    "simv")
        echo "Starting Verilator RTL simulation..."
        cd "$ARA_ROOT/hardware"
        make simv app=dnn_layer_mnist
        ;;
    *)
        echo -e "${RED}❌ Unknown simulation type: $SIM_TYPE${NC}"
        echo "Available types: simv, spike, sim, ideal"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Simulation completed!${NC}"

# Step 4: Performance summary
echo -e "${BLUE}📊 Performance Summary${NC}"
echo -e "${BLUE}=====================${NC}"
echo "✅ MNIST DNN with Vector Acceleration"
echo "✅ Matrix Multiplication: vectorized with vfmacc.vf"
echo "✅ ReLU Activation: vectorized with vmfgt.vf + vmerge.vvm"
echo "⏳ BatchNorm & Softmax: scalar (next targets for vectorization)"
echo ""
echo -e "${YELLOW}💡 Tips for Performance Analysis:${NC}"
echo "1. Look for cycle count differences between scalar and vector runs"
echo "2. Check for speedup ratios (target: 2-8x improvement)"
echo "3. Verify MNIST prediction accuracy"
echo "4. Use 'make simv app=dnn_layer_mnist trace=1' for waveform analysis"
echo ""
echo -e "${GREEN}🔥 Ready for performance optimization and further vectorization!${NC}"
