# ---- STEP 1: import the libraries ----
import time                                          # built into Python: a stopwatch for the timings

import numpy as np                                   # numpy: fast maths on big grids of numbers
from sklearn.model_selection import cross_val_score  # scikit-learn: 5-fold cross-validation (to choose k)
from sklearn.neighbors import KNeighborsClassifier   # scikit-learn: the ready-made KNN model
from sklearn.metrics import confusion_matrix         # scikit-learn: table of real digit vs guessed digit

from data import load_data                           # my own file: loads and splits the digits

TUNE = False        # True = redo the cross-validation search for k (adds about a minute)
SHOW_PLOTS = True   # False = text only, no pictures

# ---- STEP 2: load the data ----
# X = the pictures (784 pixel numbers per digit), y = the answers (0 to 9).
# X_train, y_train: 10,000 digits to learn from. X_test, y_test: 2,000 to test on.
# OUTPUT: load_data() prints "Training on 10,000 digits, testing on 2,000 unseen digits."
X_train, X_test, y_train, y_test = load_data()

# Choosing k: I tried odd values of k with 5-fold cross-validation (choose_k.py does
# the full search, k = 1 to 1001, with a graph): split the training set into 5 parts,
# train on 4, test on the 5th, repeat 5 times, average. The test set stays locked away.
if TUNE:
    print("Choosing k with 5-fold cross-validation...")
    for k in range(1, 16, 2):
        cv = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
        print(f"  k = {k:2d}   cross-validation accuracy {cv.mean():.2%}")
BEST_K = 1   # the winner: k = 1 scored 94.9% in cross-validation

# ---- STEP 3: build the model ----
start = time.time()
knn = KNeighborsClassifier(n_neighbors=BEST_K)

# ---- STEP 4: train it ----
knn.fit(X_train, y_train)                    # "training" = storing 10,000 images

# ---- STEP 5: test it on the 2,000 unseen digits, print the score ----
# How KNN predicts one digit: measure its distance to all 10,000 training digits,
# take the k closest, and they vote. score() does that for every test digit,
# then works out the share it got right.
knn_acc = knn.score(X_test, y_test)          # the real work happens here
# OUTPUT: "KNN (k = 1): 94.4% of unseen digits correct  (0.2s)"
print(f"KNN (k = {BEST_K}): {knn_acc:.1%} of unseen digits correct  ({time.time() - start:.1f}s)")

# ---- STEP 6: look closer: which mistake does it make most often? ----
# Confusion matrix: row = the real digit, column = KNN's guess.
pred = knn.predict(X_test)
cm = confusion_matrix(y_test, pred)
np.fill_diagonal(cm, 0)                      # the diagonal holds the correct answers, so zero it
a, b = np.unravel_index(cm.argmax(), cm.shape)
# OUTPUT: "KNN's most common mistake: reading a 5 as a 3 (7 times)."
print(f"KNN's most common mistake: reading a {a} as a {b} ({cm[a, b]} times).")

# ---- STEP 7: draw the graphs ----
# OUTPUT: two windows: 10 test digits KNN got wrong, and the confusion matrix
if SHOW_PLOTS:
    import matplotlib.pyplot as plt   # matplotlib: draws the graphs

    fig, axes = plt.subplots(1, 10, figsize=(14, 2.6))
    fig.suptitle("Digits KNN got wrong (true → predicted)")
    wrong = np.where(pred != y_test)[0][:10]
    for j, i in enumerate(wrong):
        axes[j].imshow(X_test[i].reshape(28, 28), cmap="gray")
        axes[j].set_title(f"{y_test[i]} → {pred[i]}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    # Figure 2: the confusion matrix, to see WHICH digits get mixed up.
    # Row = the real digit, column = KNN's guess. Each number = how many test digits.
    # The colour shows only the mistakes (cm, whose diagonal was set to 0 in step 6).
    # The pairs it mixes up (5 and 3, 7 and 1, 4 and 9) have ink in many of the same
    # pixels, and the pixels are the only features KNN has.
    full = confusion_matrix(y_test, pred)             # again, this time keeping the correct ones
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.imshow(cm, cmap="Reds", vmax=12)              # darker red = more mistakes
    for r in range(10):
        for c in range(10):
            if full[r, c] > 0:                        # write the count in its box
                ax2.text(c, r, full[r, c], ha="center", va="center", fontsize=9,
                         color="gray" if r == c else "black")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xlabel("KNN's guess")
    ax2.set_ylabel("real digit")
    ax2.set_title("Confusion matrix: grey = right, red = mistakes")
    fig2.tight_layout()
    plt.show()                                        # opens both windows; the program waits until you close them

# ---- STEP 8: open the drawing window ----
# Draw numbers with the mouse, or a finger or stylus on a touch screen: KNN reads
# each digit, shows its probability, and solves the sum. The window's code is in knn_window.py.
# It uses k = 10, so it can show a probability: with k = 1 it could only
# ever say 100% or 0%. (It opens after you close the two picture windows from step 7.)
SHOW_WINDOW = True   # False = don't open the drawing window
if SHOW_WINDOW:
    from knn_window import open_window                # my own file: the drawing window
    knn10 = KNeighborsClassifier(n_neighbors=10)
    knn10.fit(X_train, y_train)
    acc10 = knn10.score(X_test, y_test)               # its accuracy (94.2%), shown in the window's top bar
    # OUTPUT: a window. Each time you press "Read and solve" it also prints the sum, e.g. "47 + 5 = 52"
    open_window(knn10, acc10)
