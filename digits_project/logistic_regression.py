"""
logistic_regression.py: logistic regression
===========================================
Yann Imanzi Shema · A-level Computer Science

One artificial neuron per digit. Each neuron gives every pixel a weight,
multiplies, and adds it all up into a score:

    z = w1*x1 + w2*x2 + ... + w784*x784 + b

Softmax then turns the 10 scores into 10 probabilities that add up to 100%.
The highest probability wins. (With only two classes, softmax becomes the
sigmoid function, 1 / (1 + e^-z).)

Training = gradient descent: nudge every weight a tiny bit in the
direction that makes the error smaller, thousands of times.

C controls regularisation: a smaller C stops the weights from getting huge
and memorising the training data. Cross-validation picked C = 0.1.
"""

# ---- STEP 1: import the libraries ----
import time                                          # built into Python: a stopwatch for the timings

import numpy as np                                   # numpy: fast maths on big grids of numbers
from sklearn.model_selection import cross_val_score  # scikit-learn: 5-fold cross-validation (to choose C)
from sklearn.linear_model import LogisticRegression  # scikit-learn: the ready-made logistic regression model

from data import load_data                           # my own file: loads and splits the digits

TUNE = False        # True = redo the cross-validation search for C (adds about a minute)
SHOW_PLOTS = True   # False = text only, no pictures

# ---- STEP 2: load the data ----
# X = the pictures (784 pixel numbers per digit), y = the answers (0 to 9).
# X_train, y_train: 10,000 digits to learn from. X_test, y_test: 2,000 to test on.
# OUTPUT: load_data() prints "Training on 10,000 digits, testing on 2,000 unseen digits."
X_train, X_test, y_train, y_test = load_data()

if TUNE:
    print("Choosing C with 5-fold cross-validation...")
    for C in [0.01, 0.1, 1.0]:
        cv = cross_val_score(LogisticRegression(C=C, max_iter=1000), X_train, y_train, cv=5)
        print(f"  C = {C:<5} cross-validation accuracy {cv.mean():.2%}")

# ---- STEP 3: build the model ----
start = time.time()
logreg = LogisticRegression(C=0.1, max_iter=1000)

# ---- STEP 4: train it ----
# Learns 784 weights + 1 bias for each digit by gradient descent (sklearn uses L-BFGS).
logreg.fit(X_train, y_train)

# ---- STEP 5: test it on the 2,000 unseen digits, print the score ----
# How it predicts one digit: for each digit 0 to 9, score = pixels x weights + bias.
# Softmax turns the 10 scores into probabilities, and the highest one wins.
lr_acc = logreg.score(X_test, y_test)
# OUTPUT: "Logistic regression: 91.5% correct  (5.9s)"
print(f"Logistic regression: {lr_acc:.1%} correct  ({time.time() - start:.1f}s)")

# ---- STEP 6: look closer: softmax in action, the 10 probabilities for one test digit ----
EXAMPLE = 3   # which test digit to look at (test digit 3 is a 9 that looks a bit like a 4; try others)
probs = logreg.predict_proba(X_test[EXAMPLE:EXAMPLE + 1])[0]   # one row in, take its 10 probabilities out
# OUTPUT: "Test digit 3 is really a 9.", then one line per digit with a # bar, then the total
print(f"\nTest digit {EXAMPLE} is really a {y_test[EXAMPLE]}. The model's probabilities:")
for d in range(10):
    print(f"  {d}: {probs[d]:6.1%}  " + "#" * round(probs[d] * 40))
print(f"  They add up to {probs.sum():.0%}. The highest is {probs.argmax()}, so that's its answer.")

# ---- STEP 7: draw the graph ----
# OUTPUT: a window with each digit's 784 weights drawn as a picture
if SHOW_PLOTS:
    import matplotlib.pyplot as plt   # matplotlib: draws the graphs

    fig, axes = plt.subplots(1, 10, figsize=(14, 2.6))
    fig.suptitle("What logistic regression learned: each digit's 784 weights (red = votes for, blue = against)")
    for d in range(10):
        w = logreg.coef_[d].reshape(28, 28)
        lim = np.abs(w).max()
        axes[d].imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[d].set_title(str(d))
        axes[d].axis("off")
    plt.tight_layout()
    plt.show()
