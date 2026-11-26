"""
Can a computer read your handwriting?
=====================================
Yann Imanzi Shema · A-level Computer Science

Four ways to teach a computer to read handwritten digits (0 to 9):

  1. K-nearest neighbours (KNN)   "you are who your friends are"
  2. Logistic regression          one artificial neuron per digit
  3. K-means clustering           no labels at all: find the groups yourself
  4. A small neural network       and what activation functions do inside it

Data: MNIST, 70,000 handwritten digits, 28 x 28 pixels each.
It is the dataset behind Kaggle's "Digit Recognizer" competition.

Run it with:   python digits_ai.py
You need:      pip install numpy scikit-learn matplotlib
The whole thing takes about 2 minutes on a normal laptop.
"""

import os
import time
import urllib.request
import warnings

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")   # hides "did not fully converge" messages from the neural nets

TUNE = False        # True = redo the cross-validation searches (adds about 2 minutes)
SHOW_PLOTS = True   # False = text only, no pictures


# ---------------------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------------------
# Every image is a grid of 28 x 28 = 784 pixels.
# Each pixel is a number from 0 (black background) to 255 (white ink).
# So to the computer, a "picture of a 7" is just a list of 784 numbers.

DATA_FILE = "mnist.npz"
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

if not os.path.exists(DATA_FILE):
    print("Downloading MNIST (11 MB, only the first time)...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

data = np.load(DATA_FILE)
X_all = data["x_train"].reshape(-1, 784)    # flatten each 28x28 grid into one row of 784 numbers
y_all = data["y_train"]                      # the correct answer (0 to 9) for each image
X_unseen = data["x_test"].reshape(-1, 784)   # a separate set the models must never learn from
y_unseen = data["y_test"]

print(f"Loaded {len(X_all) + len(X_unseen):,} handwritten digits, each {X_all.shape[1]} pixels.")


# ---------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
# Golden rule of machine learning: never test on the data you learned from.
# That's like marking your own homework with the answer sheet open.
#
# I use 10,000 digits to learn from and 2,000 completely new ones to test on.
# stratify = keep the same share of each digit in both sets.
# random_state = a fixed seed, so anyone who runs this gets the same numbers.

X_train, _, y_train, _ = train_test_split(X_all, y_all, train_size=10_000,
                                          stratify=y_all, random_state=42)
X_test, _, y_test, _ = train_test_split(X_unseen, y_unseen, train_size=2_000,
                                        stratify=y_unseen, random_state=42)

# Scale pixels from 0-255 down to 0-1. Every pixel already uses the same
# scale, so this is just to keep the numbers small and friendly for training.
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Training on {len(X_train):,} digits, testing on {len(X_test):,} unseen digits.\n")

results = {}   # we'll collect every model's score here


# ---------------------------------------------------------------------------
# 3. K-NEAREST NEIGHBOURS
# ---------------------------------------------------------------------------
# To read a new digit: find the k training digits that look most like it,
# and let them vote. "Look like" = Euclidean distance, which is Pythagoras'
# theorem stretched to 784 dimensions:
#
#     distance = sqrt( sum over all 784 pixels of (pixel_a - pixel_b)^2 )
#
# KNN has no training step. fit() just memorises the data. That's why it's
# called a "lazy learner".
#
# Choosing k: I tried every odd k from 1 to 15 with 5-fold cross-validation.
# (Split the training set into 5 parts, train on 4, test on the 5th, repeat
# 5 times, average.) The test set stays locked away the whole time.

if TUNE:
    print("Choosing k with 5-fold cross-validation...")
    for k in range(1, 16, 2):
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
        print(f"  k = {k:2d}   cross-validation accuracy {cv.mean():.2%}")
BEST_K = 1   # the winner: k = 1 scored 94.9% in cross-validation

start = time.time()
knn = KNeighborsClassifier(n_neighbors=BEST_K)
knn.fit(X_train, y_train)                    # "training" = storing 10,000 images
knn_acc = knn.score(X_test, y_test)          # the real work happens here
print(f"KNN (k = {BEST_K}): {knn_acc:.1%} of unseen digits correct  ({time.time() - start:.1f}s)")
results["K-nearest neighbours"] = knn_acc


# ---------------------------------------------------------------------------
# 4. LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
# One artificial neuron per digit. Each neuron gives every pixel a weight,
# multiplies, and adds it all up into a score:
#
#     z = w1*x1 + w2*x2 + ... + w784*x784 + b
#
# Softmax then turns the 10 scores into 10 probabilities that add up to 100%.
# The highest probability wins. (With only two classes, softmax becomes the
# sigmoid function, 1 / (1 + e^-z).)
#
# Training = gradient descent: nudge every weight a tiny bit in the
# direction that makes the error smaller, thousands of times.
#
# C controls regularisation: a smaller C stops the weights from getting huge
# and memorising the training data. Cross-validation picked C = 0.1.

if TUNE:
    print("Choosing C with 5-fold cross-validation...")
    for C in [0.01, 0.1, 1.0]:
        cv = cross_val_score(LogisticRegression(C=C, max_iter=1000), X_train, y_train, cv=5)
        print(f"  C = {C:<5} cross-validation accuracy {cv.mean():.2%}")

start = time.time()
logreg = LogisticRegression(C=0.1, max_iter=1000)
logreg.fit(X_train, y_train)
lr_acc = logreg.score(X_test, y_test)
print(f"Logistic regression: {lr_acc:.1%} correct  ({time.time() - start:.1f}s)")
results["Logistic regression"] = lr_acc


# ---------------------------------------------------------------------------
# 5. K-MEANS CLUSTERING (unsupervised: no labels!)
# ---------------------------------------------------------------------------
# K-means is never told which digit is which. It just looks for groups:
#   1. drop k centres at random
#   2. give every image to its nearest centre
#   3. move each centre to the average of its images
#   4. repeat 2 and 3 until nothing moves
#
# Don't mix it up with KNN: same letter k, totally different job.
# KNN's k = how many neighbours vote. K-means' k = how many groups to find.
#
# To score it, I name each cluster after the digit most common inside it.
# (That's the only time labels are used, and only for marking.)

start = time.time()
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(X_train)                                   # look: no y_train!
cluster_names = np.array([np.bincount(y_train[kmeans.labels_ == c], minlength=10).argmax()
                          for c in range(10)])
km_acc = (cluster_names[kmeans.predict(X_test)] == y_test).mean()
print(f"K-means (10 clusters, no labels): {km_acc:.1%} correct  ({time.time() - start:.1f}s)")
print(f"  It named its 10 clusters: {sorted(cluster_names.tolist())}")
print(f"  Missing digits: {sorted(set(range(10)) - set(cluster_names.tolist()))}  (it found two kinds of 1 instead)")
results["K-means (no labels)"] = km_acc


# ---------------------------------------------------------------------------
# 6. NEURAL NETWORK + ACTIVATION FUNCTIONS
# ---------------------------------------------------------------------------
# A neural network stacks layers of neurons. Each neuron does:
#
#     output = activation( w1*x1 + w2*x2 + ... + b )
#
# The activation function is the "spark": it decides how strongly the neuron
# fires. Without one (activation = "identity"), stacking layers is pointless:
# two linear layers squash into one linear layer, so the whole network is
# just logistic regression in a trench coat.
#
# Experiment 1: same network (784 -> 128 -> 10), four activation functions.

print("\nExperiment: one hidden layer of 128 neurons, four activation functions")
mlp_results = {}
for act in ["identity", "logistic", "tanh", "relu"]:    # "logistic" = sigmoid
    start = time.time()
    net = MLPClassifier(hidden_layer_sizes=(128,), activation=act, max_iter=60, random_state=42)
    net.fit(X_train, y_train)
    mlp_results[act] = net.score(X_test, y_test)
    print(f"  {act:9s} {mlp_results[act]:.1%}  ({time.time() - start:.1f}s)")
results["Neural network (ReLU)"] = mlp_results["relu"]

# Experiment 2: the vanishing gradient. Sigmoid's slope is at most 0.25, so
# every layer the error signal passes back through shrinks it. Stack 8 layers
# and almost nothing reaches the first layer, so it can't learn.
# ReLU's slope is exactly 1 for any positive input, so the signal survives.

print("\nExperiment: deep networks (8 hidden layers of 64 neurons)")
for act in ["logistic", "relu"]:
    net = MLPClassifier(hidden_layer_sizes=(64,) * 8, activation=act, max_iter=30, random_state=42)
    net.fit(X_train, y_train)
    print(f"  {act:9s} {net.score(X_test, y_test):.1%}   (random guessing = 10%)")


# ---------------------------------------------------------------------------
# 7. RESULTS
# ---------------------------------------------------------------------------
print("\nFinal scores on 2,000 digits none of the models had ever seen:")
for name, acc in sorted(results.items(), key=lambda r: -r[1]):
    print(f"  {name:24s} {acc:6.1%}  " + "#" * round(acc * 40))

pred = knn.predict(X_test)
cm = confusion_matrix(y_test, pred)
np.fill_diagonal(cm, 0)
a, b = np.unravel_index(cm.argmax(), cm.shape)
print(f"\nKNN's most common mistake: reading a {a} as a {b} ({cm[a, b]} times).")


# ---------------------------------------------------------------------------
# 8. PICTURES (optional)
# ---------------------------------------------------------------------------
if SHOW_PLOTS:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 10, figsize=(14, 5))
    fig.suptitle("Top: what logistic regression learned (red = votes for, blue = against)\n"
                 "Middle: K-means' 10 cluster centres, found without labels\n"
                 "Bottom: digits KNN got wrong (true → predicted)")
    for d in range(10):
        w = logreg.coef_[d].reshape(28, 28)
        lim = np.abs(w).max()
        axes[0, d].imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[0, d].set_title(str(d))
        axes[1, d].imshow(kmeans.cluster_centers_[d].reshape(28, 28), cmap="gray")
        axes[1, d].set_title(f"looks like {cluster_names[d]}", fontsize=8)
    wrong = np.where(pred != y_test)[0][:10]
    for j, i in enumerate(wrong):
        axes[2, j].imshow(X_test[i].reshape(28, 28), cmap="gray")
        axes[2, j].set_title(f"{y_test[i]} → {pred[i]}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
