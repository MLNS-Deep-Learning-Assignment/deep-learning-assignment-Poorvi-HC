# **Regression Model for Digit Sum Prediction**

## **Overview**
This report discusses the implementation of a deep learning model aimed at predicting the sum of digits in grayscale images. The model architecture, training process, and evaluation are presented with a focus on the regression task. The methodology combines robust data preprocessing, a custom CNN architecture with an attention mechanism, and effective training strategies to achieve high accuracy.

---

## **1. Data Preprocessing**
### **1.1 Data Loading**
- Data is loaded from `.npy` files (`data0.npy`, `data1.npy`, `data2.npy`) representing grayscale images and corresponding labels.
- All datasets are concatenated to form a unified dataset for training, validation, and testing.

### **1.2 Normalization and Augmentation**
- Pixel values are normalized to the range `[0, 1]` by dividing by `255.0`.
- Channel dimensions are added for grayscale images to make them compatible with the model input.
- Gaussian noise (`noise_factor = 0.05`) is introduced to improve model robustness.

### **1.3 Data Splitting**
- The dataset is split into training and validation sets with an 80:20 ratio using `train_test_split`.

---

## **2. Model Architecture**
The model employs a custom Convolutional Neural Network (CNN) with the following components:

### **2.1 Input Layer**
- Accepts images of shape `(height, width, 1)`.

### **2.2 Convolutional Blocks**
- **Block 1:** 
  - Two `Conv2D` layers with `64` filters and ReLU activation.
  - Batch Normalization for stable learning.
  - MaxPooling and Dropout to reduce overfitting.
- **Block 2:** 
  - Similar to Block 1 but with `128` filters.
- **Block 3:**
  - Two `Conv2D` layers with `256` filters and ReLU activation.
  - Attention mechanism for feature enhancement.

### **2.3 Attention Mechanism**
- A `Conv2D` layer with a sigmoid activation generates attention weights.
- Element-wise multiplication is performed between attention weights and feature maps, highlighting relevant features.

### **2.4 Fully Connected Layers**
- Flattened feature maps are passed through two dense layers:
  - `512` and `256` neurons with ReLU activation.
  - Batch Normalization and Dropout for regularization.
- **Output Layer:**
  - A single neuron with linear activation for regression.

---

## **3. Training and Optimization**
### **3.1 Loss Function**
- **Mean Squared Error (MSE):** Used as the loss function to minimize the error in regression.

### **3.2 Optimizer**
- **Adam Optimizer:** 
  - Learning rate: `0.0005`.
  - Optimizes weights for faster convergence.

### **3.3 Callbacks**
- **EarlyStopping:** Stops training when validation loss stops improving.
- **ReduceLROnPlateau:** Reduces learning rate if validation loss plateaus.
- **ModelCheckpoint:** Saves the best model based on validation loss.

### **3.4 Training Process**
- The model is trained for up to 40 epochs with a batch size of 32.
- Training and validation losses are tracked throughout the process.

---

## **4. Evaluation**
### **4.1 Metrics**
- **Mean Absolute Error (MAE):** Measures the average prediction error.
- **Root Mean Squared Error (RMSE):** Penalizes larger errors more heavily.
- **Accuracy Metrics:**
  - Predictions within ±0.5, ±1.0, and ±2.0 digits are calculated to evaluate practical performance.

### **4.2 Visualization**
- True and predicted values are displayed for a subset of validation samples, with errors highlighted in different colors (green for low error, red for high error).

---

## **5. Key Features of the Code**
- **Custom Architecture:** Combines convolutional layers, batch normalization, attention mechanism, and dropout.
- **Robust Preprocessing:** Includes noise augmentation to enhance generalization.
- **Efficient Callbacks:** Ensures the best-performing model is retained while avoiding overfitting.

---

## **6. Results**
The model demonstrates:
- High accuracy with predictions within ±1 digit for most samples.
- Effective generalization due to noise augmentation and dropout.
- Robust training facilitated by advanced callbacks.

---

<!-- ## **7. Suggestions for Improvement**
- Explore additional data augmentation techniques (e.g., rotation, scaling).
- Experiment with advanced attention mechanisms like Squeeze-and-Excitation blocks.
- Fine-tune hyperparameters such as learning rate and dropout rate.

--- -->

## **Conclusion**
This regression model effectively predicts the sum of digits in images by leveraging a custom CNN with an attention mechanism. With careful preprocessing, architecture design, and optimization strategies, the model achieves promising results for this challenging task.
