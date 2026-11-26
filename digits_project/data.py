"""
data.py: load the handwritten digits from the two CSV files
==========================================================
Yann Imanzi Shema · A-level Computer Science

THE DATA is in two CSV files in this folder (open them in Excel to see it):
    training_digits_10000.csv   10,000 digits the models learn from
    test_digits_2000.csv         2,000 different digits they are tested on
One row = one digit: image, label (the right answer, 0 to 9), mnist_file and
mnist_row (where it is in the original MNIST file), then pixel0 to pixel783
(the 28 x 28 = 784 pixels, 0 = background, 255 = full ink).

SOURCE: the MNIST database of handwritten digits, by Yann LeCun, Corinna
Cortes and Christopher Burges: https://yann.lecun.com/exdb/mnist/
(licence: Creative Commons Attribution-Share Alike 3.0)
I downloaded it as mnist.npz, the copy hosted for Keras/TensorFlow:
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
make_csv_files() at the bottom picks my 12,000 digits from it and writes
the two CSV files. It only runs if the CSV files are missing.

Every other file starts with   from data import load_data
so KNN, logistic regression, K-means and the neural network all learn from
exactly the same 10,000 digits and are tested on exactly the same 2,000.

Run this file on its own to see what one digit looks like to the computer.
"""

import os                    # built into Python: checks whether a file is already there
import urllib.request        # built into Python: downloads mnist.npz if it is missing

import numpy as np                                    # numpy: fast maths on big grids of numbers
from sklearn.model_selection import train_test_split  # scikit-learn: splits data into training and test sets

TRAIN_CSV = "training_digits_10000.csv"   # the 10,000 digits to learn from
TEST_CSV = "test_digits_2000.csv"         # the 2,000 digits to test on
MNIST_FILE = "mnist.npz"                  # the original MNIST download the CSV files were made from
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def read_csv(path):
    # Columns: 0 image, 1 label, 2 mnist_file, 3 mnist_row, then 4 to 787 = the 784 pixels.
    table = np.loadtxt(path, delimiter=",", skiprows=1, usecols=[1] + list(range(4, 788)), dtype=int)
    y = table[:, 0]     # the right answers (0 to 9), one per row
    X = table[:, 1:]    # the pictures: 784 pixel numbers per row
    return X, y


def load_data():
    # Make the two CSV files from MNIST the first time (only if they are missing).
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
        make_csv_files()

    # Every image is a grid of 28 x 28 = 784 pixels, stored as one row of the CSV file.
    # Each pixel is a number from 0 (black background) to 255 (white ink).
    X_train, y_train = read_csv(TRAIN_CSV)   # X = the pictures, y = the correct answers
    X_test, y_test = read_csv(TEST_CSV)      # a separate set the models must never learn from

    # Scale pixels from 0-255 down to 0-1, to keep the numbers small and friendly for training.
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # OUTPUT: "Training on 10,000 digits from training_digits_10000.csv", then the test line. Every file prints this first.
    print(f"Training on {len(X_train):,} digits from {TRAIN_CSV}\nTesting on {len(X_test):,} unseen digits from {TEST_CSV}\n")
    return X_train, X_test, y_train, y_test


def make_csv_files():
    # Only needed once. Downloads MNIST (if needed), picks my 12,000 digits and saves them as CSV.
    if not os.path.exists(MNIST_FILE):
        print("Downloading MNIST (11 MB, only the first time)...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_FILE)
    data = np.load(MNIST_FILE)   # MNIST comes as 60,000 training digits and 10,000 separate test digits

    # Golden rule of machine learning: never test on the data you learned from.
    # 10,000 digits to learn from (out of MNIST's 60,000) and 2,000 completely new ones to test on
    # (out of MNIST's separate 10,000). stratify = keep the same share of each digit in both sets.
    # random_state = a fixed seed, so it always picks the same digits.
    rows_train, _ = train_test_split(np.arange(60_000), train_size=10_000, stratify=data["y_train"], random_state=42)
    rows_test, _ = train_test_split(np.arange(10_000), train_size=2_000, stratify=data["y_test"], random_state=42)

    for path, part, rows in [(TRAIN_CSV, "x_train", rows_train), (TEST_CSV, "x_test", rows_test)]:
        pictures = data[part][rows].reshape(-1, 784)            # flatten each 28 x 28 grid into one row of 784 numbers
        answers = data[part.replace("x", "y")][rows]
        with open(path, "w") as f:
            f.write("image,label,mnist_file,mnist_row," + ",".join(f"pixel{i}" for i in range(784)) + "\n")
            for n in range(len(rows)):
                f.write(f"{n + 1},{answers[n]},{part},{rows[n]}," + ",".join(map(str, pictures[n])) + "\n")
        print(f"Saved {len(rows):,} digits to {path}")


# This part only runs when you run data.py itself, not when another file imports it.
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    np.set_printoptions(linewidth=200)
    # OUTPUT: "The first training digit is a 6...", then its 784 numbers as 28 rows of 28
    print(f"The first training digit is a {y_train[0]}. To the computer it is just these 784 numbers:\n")
    print((X_train[0] * 255).astype(int).reshape(28, 28))

    import matplotlib.pyplot as plt   # matplotlib: draws the graphs

    # OUTPUT: a window with 20 training digits and their answers
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.6))
    fig.suptitle("20 training digits, each with the answer (y) the computer is given")
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
        ax.set_title(str(y_train[i]))
        ax.axis("off")
    plt.tight_layout()
    plt.show()
