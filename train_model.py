import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST
(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(60000, 784)

# One-hot encode labels
y_train_oh = np.eye(10)[y_train]

# Initialize weights
w_i_h = np.random.uniform(-0.5, 0.5, (20, 785))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 21))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.1
epochs = 5  # keep small for now

for _ in range(epochs):
    for img, label in zip(x_train, y_train_oh):
        img = img.reshape(784, 1)
        img_aug = np.vstack(([1], img))

        # Forward
        h = sigmoid(w_i_h @ img_aug)
        h_aug = np.vstack(([1], h))
        o = sigmoid(w_h_o @ h_aug)

        # Backprop
        delta_o = o - label.reshape(10, 1)
        w_h_o -= learning_rate * delta_o @ h_aug.T

        delta_h = (w_h_o.T @ delta_o) * h_aug * (1 - h_aug)
        w_i_h -= learning_rate * delta_h[1:] @ img_aug.T

# SAVE WEIGHTS
np.save("w_i_h.npy", w_i_h)
np.save("w_h_o.npy", w_h_o)

print("Weights saved successfully!")
