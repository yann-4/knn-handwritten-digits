"""
neural_network.py: a neural network and activation functions
============================================================
Yann Imanzi Shema · A-level Computer Science

A neural network stacks layers of neurons. Each neuron does:

    output = activation( w1*x1 + w2*x2 + ... + b )

The activation function is the "spark": it decides how strongly the neuron
fires. Without one (activation = "identity"), stacking layers is pointless:
two linear layers squash into one linear layer, so the whole network is
just logistic regression in a trench coat.

Experiment 1: same network (784 -> 128 -> 10), four activation functions.
Experiment 2: the vanishing gradient, with 8 layers.

This is the slowest file: about a minute.
"""

# ---- STEP 1: import the libraries ----
import time                                    # built into Python: a stopwatch for the timings
import warnings                                # built into Python: lets me hide warning messages

import numpy as np                             # numpy: fast maths on big grids of numbers
from sklearn.neural_network import MLPClassifier  # scikit-learn: the ready-made neural network

from data import load_data                     # my own file: loads and splits the digits

warnings.filterwarnings("ignore")   # hides "did not fully converge" messages from the neural nets

SHOW_PLOTS = True   # False = text only, no pictures

# ---- STEP 2: load the data ----
# X = the pictures (784 pixel numbers per digit), y = the answers (0 to 9).
# X_train, y_train: 10,000 digits to learn from. X_test, y_test: 2,000 to test on.
# OUTPUT: load_data() prints "Training on 10,000 digits, testing on 2,000 unseen digits."
X_train, X_test, y_train, y_test = load_data()

# ---- STEPS 3 to 5: build, train and test one network per activation function ----
# OUTPUT: "Experiment 1: ..." then one line per activation function, e.g. "  relu      95.8%  (7.4s)"
print("Experiment 1: one hidden layer of 128 neurons, four activation functions")
mlp_results = {}
for act in ["identity", "logistic", "tanh", "relu"]:    # "logistic" = sigmoid
    start = time.time()
    # STEP 3: build a network: 784 inputs -> 128 neurons -> 10 outputs
    net = MLPClassifier(hidden_layer_sizes=(128,), activation=act, max_iter=60, random_state=42)
    # STEP 4: train it (backpropagation: pass the error backwards, nudge every weight)
    net.fit(X_train, y_train)
    # STEP 5: test it. How it predicts one digit: the pixels pass through the 128 neurons
    # (weighted sum + activation), the 10 outputs go through softmax, and the highest wins.
    mlp_results[act] = net.score(X_test, y_test)
    print(f"  {act:9s} {mlp_results[act]:.1%}  ({time.time() - start:.1f}s)")

# ---- STEP 6: look closer: experiment 2, the vanishing gradient ----
# Experiment 2: the vanishing gradient. Sigmoid's slope is at most 0.25, so
# every layer the error signal passes back through shrinks it. Stack 8 layers
# and almost nothing reaches the first layer, so it can't learn.
# ReLU's slope is exactly 1 for any positive input, so the signal survives.

# OUTPUT: "Experiment 2: ..." then "logistic 11.3%" and "relu 95.7%"
print("\nExperiment 2: deep networks (8 hidden layers of 64 neurons)")
for act in ["logistic", "relu"]:
    net = MLPClassifier(hidden_layer_sizes=(64,) * 8, activation=act, max_iter=30, random_state=42)
    net.fit(X_train, y_train)
    print(f"  {act:9s} {net.score(X_test, y_test):.1%}   (random guessing = 10%)")

# ---- STEP 7: draw the graph ----
# OUTPUT: a window with the four activation functions (left) and their scores (right)
if SHOW_PLOTS:
    import matplotlib.pyplot as plt   # matplotlib: draws the graphs

    z = np.linspace(-4, 4, 200)                      # 200 input values from -4 to 4
    shapes = {"identity": z,
              "logistic": 1 / (1 + np.exp(-z)),      # sigmoid
              "tanh": np.tanh(z),
              "relu": np.maximum(0, z)}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for act in ["relu", "tanh", "logistic", "identity"]:   # identity last and dashed, so it shows on top of ReLU
        style = "--" if act == "identity" else "-"
        left.plot(z, shapes[act], linewidth=2, linestyle=style, label=act)
    left.set_title("The four activation functions")
    left.set_xlabel("weighted sum going in (z)")
    left.set_ylabel("what the neuron outputs")
    left.set_ylim(-1.5, 4)
    left.grid(alpha=0.3)
    left.legend()

    names = list(mlp_results)
    scores = [mlp_results[act] * 100 for act in names]
    right.bar(names, scores, color="#4C78A8")
    for x, s in enumerate(scores):
        right.text(x, s + 1.5, f"{s:.1f}%", ha="center")
    right.set_title("Accuracy on 2,000 unseen digits")
    right.set_ylim(0, 105)
    right.set_ylabel("% correct")
    plt.tight_layout()
    plt.show()
