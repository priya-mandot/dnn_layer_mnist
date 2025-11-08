# MNIST DNN Layer Test Application
# Copyright 2025, SPDX-License-Identifier: Apache-2.0

# Training and verification targets for local development
.PHONY: train verify test-py clean-data

train:
	@echo "Training MNIST DNN model..."
	python3 train_mnist_dnn.py
	@echo "Training completed. Generated data.S"

verify: data.S
	@echo "Running Python verification..."
	python3 mnist_dnn_verification.py
	@echo "Python verification completed."

test-py: verify

clean-data:
	rm -f data.S mnist_data.S *.png *.log

# The Ara build system will automatically find main.c and build it
# It will call script/gen_data.py to generate data.S
