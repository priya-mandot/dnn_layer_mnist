#!/usr/bin/env python3
"""
MNIST DNN Data Generator for Ara Build System
Outputs assembly format to stdout for build system
UPDATED: Emits transposed weights for optimized matmul
"""

import numpy as np
import sys
import struct
from sklearn.datasets import fetch_openml

class MNISTDNNLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Better weight initialization (Xavier/Glorot)
        xavier_std = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.normal(0, xavier_std, (output_size, input_size)).astype(np.float32)
        
        # Batch normalization parameters
        self.bn_gamma = np.ones(output_size, dtype=np.float32)
        self.bn_beta = np.zeros(output_size, dtype=np.float32)
        self.bn_running_mean = np.zeros(output_size, dtype=np.float32)
        self.bn_running_var = np.ones(output_size, dtype=np.float32)
        self.bn_epsilon = 1e-5
        
        # For momentum in batch norm
        self.bn_momentum = 0.1
    
    def forward(self, x):
        # MatMul
        matmul_output = np.dot(x, self.weights.T)
        
        # BatchNorm
        normalized = (matmul_output - self.bn_running_mean) / np.sqrt(self.bn_running_var + self.bn_epsilon)
        bn_output = self.bn_gamma * normalized + self.bn_beta
        
        # ReLU
        relu_output = np.maximum(0, bn_output)
        
        # Softmax
        exp_vals = np.exp(relu_output - np.max(relu_output, axis=-1, keepdims=True))
        softmax_output = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        return softmax_output
    
    def train_step(self, batch_x, batch_y, learning_rate):
        # Improved training with proper backpropagation
        batch_size = len(batch_x)
        
        # Forward pass with intermediate values saved
        # MatMul
        matmul_out = np.dot(batch_x, self.weights.T)
        
        # Update running stats during training
        batch_mean = np.mean(matmul_out, axis=0)
        batch_var = np.var(matmul_out, axis=0)
        self.bn_running_mean = (1 - self.bn_momentum) * self.bn_running_mean + self.bn_momentum * batch_mean
        self.bn_running_var = (1 - self.bn_momentum) * self.bn_running_var + self.bn_momentum * batch_var
        
        # BatchNorm (use batch statistics for training)
        normalized = (matmul_out - batch_mean) / np.sqrt(batch_var + self.bn_epsilon)
        bn_out = self.bn_gamma * normalized + self.bn_beta
        
        # ReLU  
        relu_out = np.maximum(0, bn_out)
        
        # Softmax with numerical stability
        exp_vals = np.exp(relu_out - np.max(relu_out, axis=-1, keepdims=True))
        softmax_out = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        # Compute loss (cross-entropy with label smoothing)
        y_one_hot = np.eye(self.output_size)[batch_y]
        # Add label smoothing for better training
        smoothing = 0.1
        y_smooth = y_one_hot * (1 - smoothing) + smoothing / self.output_size
        loss = -np.mean(np.sum(y_smooth * np.log(softmax_out + 1e-8), axis=1))
        
        # Backward pass (proper gradients)
        # Gradient through softmax + cross-entropy
        grad_softmax = (softmax_out - y_one_hot) / batch_size
        
        # Gradient through ReLU
        grad_relu = grad_softmax * (relu_out > 0)
        
        # Gradient through BatchNorm
        grad_bn_out = grad_relu
        grad_gamma = np.sum(grad_bn_out * normalized, axis=0)
        grad_beta = np.sum(grad_bn_out, axis=0)
        
        # Gradient through normalization
        std_inv = 1.0 / np.sqrt(batch_var + self.bn_epsilon)
        grad_normalized = grad_bn_out * self.bn_gamma
        grad_var = -0.5 * np.sum(grad_normalized * (matmul_out - batch_mean), axis=0) * (std_inv ** 3)
        grad_mean = -np.sum(grad_normalized * std_inv, axis=0) + grad_var * (-2.0 * np.sum(matmul_out - batch_mean, axis=0)) / batch_size
        grad_matmul = grad_normalized * std_inv + grad_var * 2.0 * (matmul_out - batch_mean) / batch_size + grad_mean / batch_size
        
        # Gradient through MatMul
        grad_weights = np.dot(grad_matmul.T, batch_x)
        
        # Update parameters with clipping to prevent exploding gradients
        grad_weights = np.clip(grad_weights, -1.0, 1.0)
        grad_gamma = np.clip(grad_gamma, -1.0, 1.0)
        grad_beta = np.clip(grad_beta, -1.0, 1.0)
        
        self.weights -= learning_rate * grad_weights
        self.bn_gamma -= learning_rate * grad_gamma
        self.bn_beta -= learning_rate * grad_beta
        
        return loss

def emit_debug_buffers(output_size):
    """Emit debug buffers for intermediate layer outputs"""
    print("# Debug buffers for layer-by-layer comparison")
    print(".section .data")
    print(".align 5")
    print()
    
    # Scalar implementation debug buffers
    print(".globl matmul_out_scalar")
    print("matmul_out_scalar:")
    print(f"    .zero {output_size * 4}  # {output_size} outputs * 4 bytes per float")
    print()
    
    print(".globl bn_out_scalar")
    print("bn_out_scalar:")
    print(f"    .zero {output_size * 4}")
    print()
    
    print(".globl relu_out_scalar")
    print("relu_out_scalar:")
    print(f"    .zero {output_size * 4}")
    print()
    
    # Vector implementation debug buffers
    print(".globl matmul_out_vec")
    print("matmul_out_vec:")
    print(f"    .zero {output_size * 4}")
    print()
    
    print(".globl bn_out_vec")
    print("bn_out_vec:")
    print(f"    .zero {output_size * 4}")
    print()
    
    print(".globl relu_out_vec")
    print("relu_out_vec:")
    print(f"    .zero {output_size * 4}")
    print()

def emit_assembly_to_stdout(model, test_input, expected_output):
    """Emit assembly data to stdout for build system"""
    
    def emit_float(name, value):
        hex_val = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        print(f".section .rodata")
        print(f".globl {name}")
        print(f".align 4")
        print(f"{name}:")
        print(f"    .word 0x{hex_val:08x}")
        print()
    
    def emit_uint64(name, value):
        print(f".section .rodata")
        print(f".globl {name}")
        print(f".align 8")
        print(f"{name}:")
        print(f"    .dword {int(value)}")
        print()
    
    def emit_array(name, array):
        flat_array = array.flatten()
        print(f".section .rodata")
        print(f".globl {name}")
        print(f".align 5")  # 2^5 = 32 bytes
        print(f"{name}:")
        for i, val in enumerate(flat_array):
            hex_val = struct.unpack('<I', struct.pack('<f', float(val)))[0]
            print(f"    .word 0x{hex_val:08x}  # {name}[{i}] = {val}")
        # Padding for alignment
        if len(flat_array) % 4 != 0:
            padding_needed = 4 - (len(flat_array) % 4)
            for _ in range(padding_needed):
                print("    .word 0xdeadbeef  # padding")
        print()
    
    
    # Assembly header
    print("# Generated MNIST DNN data for Ara")
    print("# Single test sample + trained model parameters")
    print()
        
    # Configuration
    emit_uint64("batch_size", 1)
    emit_uint64("input_size", model.input_size)
    emit_uint64("output_size", model.output_size)
    emit_float("bn_epsilon", model.bn_epsilon)
    
    # Input data
    emit_array("input_data", test_input.astype(np.float32))
    
    # CRITICAL CHANGE: Transpose weights from [P x N] to [N x P] for contiguous access
    # Original: weights[P, N] means weights[output_size, input_size]
    # Transposed: weights[N, P] means weights[input_size, output_size]
    weights_transposed = model.weights.T  # Now [input_size x output_size]
    
    print(f"# TRANSPOSED WEIGHTS: Shape {weights_transposed.shape} for optimized GEMV", file=sys.stderr)
    emit_array("weights", weights_transposed.astype(np.float32))
    
    # Batch norm parameters (unchanged)
    emit_array("bn_gamma", model.bn_gamma.astype(np.float32))
    emit_array("bn_beta", model.bn_beta.astype(np.float32))
    emit_array("bn_mean", model.bn_running_mean.astype(np.float32))
    emit_array("bn_var", model.bn_running_var.astype(np.float32))
    
    # Expected output (gold reference)
    emit_array("gold_output", expected_output.astype(np.float32))
    
    # Output buffers
    emit_array("output_vec", np.zeros(model.output_size, dtype=np.float32))
    emit_array("output_scalar", np.zeros(model.output_size, dtype=np.float32))
    emit_array("temp_buffer", np.zeros(max(model.input_size, model.output_size) * 4, dtype=np.float32))
    
    # Debug buffers
    emit_debug_buffers(model.output_size)
    
def main():
    # Configuration - Improved training parameters
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    epochs = 50  # More epochs for better convergence
    initial_lr = 0.1  # Higher initial learning rate
    batch_size = 64  # Larger batch size for stable gradients
    
    print("Loading MNIST dataset...", file=sys.stderr)
    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    
    # Use more training data for better convergence
    n_train = min(5000, len(X))  # Use more data
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:n_train+100]
    y_test = y[n_train:n_train+100]
    
    # Normalize data better (zero mean, unit variance)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    
    print(f"Training samples: {len(X_train)}", file=sys.stderr)
    print(f"Test samples: {len(X_test)}", file=sys.stderr)
    
    # Create and train model
    model = MNISTDNNLayer(INPUT_SIZE, OUTPUT_SIZE)
    
    print("Training model...", file=sys.stderr)
    for epoch in range(epochs):
        # Learning rate decay
        learning_rate = initial_lr * (0.95 ** epoch)  # Decay learning rate
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data each epoch
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            if len(batch_X) == batch_size:  # Only use full batches
                loss = model.train_step(batch_X, batch_y, learning_rate)
                epoch_loss += loss
                num_batches += 1
        
        if num_batches > 0:
            epoch_loss /= num_batches
        
        if epoch % 5 == 0:
            # Calculate accuracy on training set
            train_pred = model.forward(X_train_shuffled[:1000])  # Sample for speed
            train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train_shuffled[:1000]) * 100
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={train_acc:.2f}%, LR={learning_rate:.6f}", file=sys.stderr)
    
    print("Training completed!", file=sys.stderr)
    
    # Get test sample and expected output
    test_input = X_test[0]
    expected_output = model.forward(test_input.reshape(1, -1)).flatten()
    
    print(f"Test sample shape: {test_input.shape}", file=sys.stderr)
    print(f"Expected output sum: {np.sum(expected_output):.6f}", file=sys.stderr)
    
    # Generate assembly to stdout
    emit_assembly_to_stdout(model, test_input, expected_output)

if __name__ == "__main__":
    main()
