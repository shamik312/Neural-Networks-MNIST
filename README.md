# 🧠 Neural Network for Handwritten Digit Classification (MNIST)

This project implements a **Multilayer Perceptron (MLP)** from scratch using Python to classify handwritten digits from the **MNIST dataset**. The objective is to understand the mathematical foundations of neural networks, including forward propagation, backpropagation, gradient descent, and performance evaluation metrics.

---

## 📌 Project Aim

To design and analyze a neural network model that studies the behavior and accuracy of **multilayer perceptrons** in predicting handwritten digits (0–9) from the MNIST dataset.

---

## 📊 Dataset Used

**MNIST Dataset**
- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 (grayscale)
- Flattened input size: 784 features
- Output classes: 10 digits (0–9)

---

## 🏗️ Network Architecture

- **Input Layer**: 784 neurons (+ bias)
- **Hidden Layer**: 20 neurons (+ bias)
- **Output Layer**: 10 neurons
- **Activation Function**: Sigmoid
- **Loss Function**: Mean Squared Error (used in implementation)
- **Learning Methods**:
  - Batch Gradient Descent
  - Stochastic Gradient Descent
  - Mini-Batch Gradient Descent (conceptual discussion)

---

## ⚙️ Implementation Details

- Implemented **forward propagation** and **backward propagation** manually
- Weights and biases initialized randomly
- Bias terms merged with weight matrices for simplicity
- Training performed over user-defined epochs and learning rate
- Accuracy calculated after each epoch

---

## 🧮 Mathematical Concepts Covered

- Gradient Descent optimization
- Sigmoid activation and derivative
- Forward propagation equations
- Backpropagation with analytical gradient derivation
- Weight update rules
- Confusion matrix:
  - True Positive (TP)
  - True Negative (TN)
  - False Positive (FP)
  - False Negative (FN)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve (One-vs-All for each digit class)
- Macro-averaged performance metrics

ROC curves for all 10 digits are plotted to visualize classifier performance.

---

## 🖥️ User Interaction

- User inputs:
  - Learning rate
  - Number of training epochs
  - Image index (0–59999)
- Output:
  - Predicted digit label
  - Display of the corresponding handwritten image
  - Epoch-wise accuracy updates

---

## 🧑‍💻 Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow (only for loading MNIST dataset)
- Keras Dataset API

---

## 📁 File Overview

- MNIST data saved locally as `.npz`
- Single Python script for:
  - Data preprocessing
  - Model training
  - Evaluation
  - Visualization
  - User testing

---

## 📌 Results Summary

- The neural network achieves **high classification accuracy**
- Demonstrates effective learning through backpropagation
- Confirms theoretical gradient derivations through implementation
- Successfully classifies handwritten digits from unseen data

---

## 🎓 Learning Outcomes

- Strong understanding of neural network fundamentals
- Hands-on experience with matrix-based ML computation
- Clear insight into optimization, loss functions, and evaluation
- Practical exposure to deep learning concepts without high-level libraries

---

## 👨‍🎓 Author

**Shamik Mukherji**  
B.Tech, Computer Science and Engineering  
VIT Vellore  
May 2025

---

## 📜 Acknowledgement

This project was completed under the guidance of **Prof. Rajat De**,  
Indian Statistical Institute (ISI), Kolkata.

---

## 📄 License

This project is intended for **academic and educational use**.
