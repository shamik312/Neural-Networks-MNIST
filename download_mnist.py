import numpy as np
from tensorflow.keras.datasets import mnist

# Download MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save as .npz file
np.savez(
    "mnist.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print("mnist.npz saved successfully!")
