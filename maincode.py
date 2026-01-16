import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.datasets import mnist #imports the MNIST dataset from Keras, which is a high-level API for building and training deep learning models in TensorFlow
from sklearn.metrics import roc_curve, auc, roc_auc_score
# Save MNIST data once (only the first time)
(x_train, y_train), (_, _) = mnist.load_data()
np.savez("mnist.npz", x_train=x_train, y_train=y_train) # saving mnist data to a file named mnist.npz

# Function to load and preprocess the dataset
def get_mnist():
    data_path = pathlib.Path(__file__).parent / "mnist.npz"
    with np.load(data_path) as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255 #normalizes the pixel values of image data so that each pixel is between 0.0 and 1.0 instead of 0 to 255.
    images = images.reshape((images.shape[0], 784))  # 60000x784
    labels = np.eye(10)[labels]  # one-hot encode
    return images, labels

# Load data
images, labels = get_mnist()

# Initialize weights and biases
w_i_h = np.random.uniform(-0.5, 0.5, (20, 785))  # input → hidden #20x785
w_h_o = np.random.uniform(-0.5, 0.5, (10, 21))   # hidden → output #10x21


# Training hyperparameters
learn_rate = float(input("Enter learning rate:"))
epochs = int(input("Enter number of iterations:"))

# Training loop
for epoch in range(epochs):
    nr_correct = 0
    for img, l in zip(images, labels):
        img = img.reshape(784, 1) # Original 784×1 vector
        img_augmented = np.vstack(([1], img)) #785x1
        l = l.reshape(10, 1) #10x1

        # --- Forward propagation ---
        h_pre = w_i_h @ img_augmented #20x1
        h = 1 / (1 + np.exp(-h_pre))  # Sigmoid activation #20x1
        h_augmented = np.vstack(([1], h)) #21x1
        o_pre =w_h_o @ h_augmented  #10x1
        o = 1 / (1 + np.exp(-o_pre))  # Sigmoid activation #10x1

        # --- Error & accuracy ---
        e = (1 / len(o)) * np.sum((o - l) ** 2) # error is calculated through Mean Square Error not Cross-Entropy
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # --- Backpropagation ---
        delta_o = o - l #10x1
        w_h_o -= learn_rate * delta_o @ h_augmented.T #10x21

        delta_h = (w_h_o.T @ delta_o) * (h_augmented * (1 - h_augmented)) #21x1
        w_i_h -= learn_rate * delta_h[1:] @ img_augmented.T #20x785


    # Epoch accuracy
    acc = round((nr_correct / images.shape[0]) * 100, 2)
    print(f"Epoch {epoch + 1}: Accuracy = {acc}%")
all_preds = []  
all_labels = []

for img, l in zip(images, labels):
    img = img.reshape(784, 1)
    img_augmented = np.vstack(([1], img))

    h_pre = w_i_h @ img_augmented
    h = 1 / (1 + np.exp(-h_pre))
    h_augmented = np.vstack(([1], h))
    o_pre = w_h_o @ h_augmented
    o = 1 / (1 + np.exp(-o_pre))

    all_preds.append(o.flatten()) #10x1
    all_labels.append(l.flatten()) #10x1
all_preds = np.array(all_preds) 
all_labels = np.array(all_labels)
y_true = np.argmax(all_labels, axis=1)  
y_pred = np.argmax(all_preds, axis=1)

accuracy = np.sum(y_true == y_pred) / len(y_true)

num_classes = 10
precision_per_class = []
recall_per_class = []
f1_per_class = []

for cls in range(num_classes):
    tp = np.sum((y_pred == cls) & (y_true == cls))  #how many elemsnts have actual and predicted values same
    fp = np.sum((y_pred == cls) & (y_true != cls))   #how many elements have the actual value wrong but prediction is correct
    fn = np.sum((y_pred != cls) & (y_true == cls))   # how many element have actual value correct but prediction is wrong

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    precision_per_class.append(precision)
    recall_per_class.append(recall)
    f1_per_class.append(f1)

macro_precision = np.mean(precision_per_class)
macro_recall = np.mean(recall_per_class)
macro_f1 = np.mean(f1_per_class)

print("\n--- Manual Classification Metrics ---")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {macro_precision * 100:.2f}%")
print(f"Recall   : {macro_recall * 100:.2f}%")
print(f"F1 Score : {macro_f1 * 100:.2f}%")
# Calculate ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Plotting ROC curves
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f"Digit {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Each Digit Class (One-vs-All)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



# --- Testing loop ---
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index].reshape(784, 1)
    label = np.argmax(labels[index])
    img_augmented = np.vstack(([1], img))
    

    # Forward pass
    h_pre = w_i_h @ img_augmented
    h = 1 / (1 + np.exp(-h_pre))
    h_augmented = np.vstack(([1], h))
    o_pre =  w_h_o @ h_augmented
    o = 1 / (1 + np.exp(-o_pre))

    prediction = np.argmax(o)
    print(f"Label: {label}")
    plt.imshow(img.reshape(28, 28), cmap="Greys")
    plt.show()

