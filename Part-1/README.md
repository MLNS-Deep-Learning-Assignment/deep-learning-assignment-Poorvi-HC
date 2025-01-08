# Report

## Overview
This report describes a Convolutional Neural Network (CNN) for predicting labels from grayscale image data, implemented in TensorFlow/Keras.

---

## Model Architecture

The CNN consists of:

1. **Convolutional Layers**:
   - Two blocks with `Conv2D`, `MaxPooling2D`, and `Dropout` layers.
   - Filters: 32 and 64; Kernel: 3x3; Activation: ReLU.

2. **Fully Connected Layers**:
   - `Flatten` layer for feature extraction.
   - Dense layer with 128 neurons and Dropout.
   - Output layer with a single neuron (linear activation).

---

## Data Pipeline

1. **Loading**:
   Combines multiple `.npy` files for data and labels.
2. **Preprocessing**:
   - Normalization: Images scaled to [0, 1].
   - Channel expansion: Adds a grayscale channel dimension.
   - Train-test split: 80% training, 20% validation.

---

## Training and Metrics

- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam (learning rate = 0.001).
- **Metrics**: Mean Absolute Error (MAE).
- **Epochs**: 10; Batch size: 32.
- **Validation Performance**:
  - Reports final loss and MAE.
  
---

## Evaluation and Improvements

- **Evaluation**:
   - Accuracy computed as predictions within a threshold of 1.0.
   - Visualization of predictions for qualitative analysis.
- **Potential Enhancements**:
   - Add data augmentation.
   - Tune hyperparameters and dropout rates.
   - Use callbacks (e.g., early stopping).

---

This model is a solid baseline for regression tasks with grayscale images, balancing simplicity and performance.
