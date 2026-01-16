import numpy as np
import os

# --------------------------------------------------
# Get absolute path of this file's directory
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# File paths
# --------------------------------------------------
MNIST_PATH = os.path.join(BASE_DIR, "mnist.npz")
W_I_H_PATH = os.path.join(BASE_DIR, "w_i_h.npy")
W_H_O_PATH = os.path.join(BASE_DIR, "w_h_o.npy")

# --------------------------------------------------
# Load MNIST dataset
# --------------------------------------------------
def load_data():
    with np.load(MNIST_PATH) as f:
        images = f["x_train"]
        labels = f["y_train"]

    images = images.astype("float32") / 255.0
    images = images.reshape(images.shape[0], 784)

    return images, labels

images, labels = load_data()

# --------------------------------------------------
# Load trained weights
# --------------------------------------------------
w_i_h = np.load(W_I_H_PATH)
w_h_o = np.load(W_H_O_PATH)

# --------------------------------------------------
# Activation function
# --------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --------------------------------------------------
# Prediction function (used by Flask)
# --------------------------------------------------
def predict_digit(index):
    img = images[index].reshape(784, 1)
    img_aug = np.vstack(([1], img))

    # Forward propagation
    h = sigmoid(w_i_h @ img_aug)
    h_aug = np.vstack(([1], h))

    o = sigmoid(w_h_o @ h_aug)

    return int(np.argmax(o))
