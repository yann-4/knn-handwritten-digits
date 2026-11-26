"""
choose_k.py: how to choose k for KNN from the data you have
==========================================================
Yann Imanzi Shema · A-level Computer Science

k = how many neighbours vote. No formula gives the best k, because it
depends on the data. So you try the candidates on your own data:

  1. Lock the test set away. k must be chosen without it, or the final
     score would be a cheat (k would be picked to suit those 2,000 digits).
  2. Pick candidates: small odd numbers (odd avoids some tie votes), the
     rule of thumb k = square root of n, and a few big ones to see the trend.
  3. Score every candidate with 5-fold cross-validation: split the 10,000
     training digits into 5 parts, learn from 4, test on the 5th, repeat 5
     times so every part is tested once, and average the 5 scores.
  4. Choose the k with the best cross-validation score.
  5. Only then use the test set, once, to report the final score.

Too small a k copies noise: one odd-looking neighbour decides on its own.
Too big a k blurs: far-away digits that look different get a vote too.
It takes about a minute, because it measures every distance many times.
"""

# ---- STEP 1: import the libraries ----
import numpy as np                                    # numpy: fast maths on big grids of numbers
from sklearn.model_selection import cross_val_score   # scikit-learn: 5-fold cross-validation
from sklearn.neighbors import KNeighborsClassifier    # scikit-learn: the ready-made KNN model

from data import load_data                            # my own file: loads and splits the digits

SHOW_PLOTS = True   # False = text only, no graph

# ---- STEP 2: load the data (the test set stays locked away until step 6) ----
X_train, X_test, y_train, y_test = load_data()
n = len(X_train)                                      # 10,000 training digits

# ---- STEP 3: the candidates for k ----
root_n = int(np.sqrt(n))                              # rule of thumb: square root of n = 100
CANDIDATES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, root_n + 1, 301, 1001]   # all odd (100 + 1 = 101)

# ---- STEP 4: score every candidate on the training digits only ----
# cross-validation = digits it did NOT learn from; training = digits it DID learn from
# OUTPUT: one line per k, e.g. "k =   1   cross-validation 94.9%   training 100.0%"
cv_scores, train_scores = [], []
for k in CANDIDATES:
    model = KNeighborsClassifier(n_neighbors=k)
    cv = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1).mean()   # average of the 5 tests
    train = model.fit(X_train, y_train).score(X_train, y_train)             # tested on what it memorised
    cv_scores.append(cv)
    train_scores.append(train)
    print(f"k = {k:4d}   cross-validation {cv:.1%}   training {train:.1%}")

# ---- STEP 5: choose the k with the best cross-validation score ----
best_k = CANDIDATES[int(np.argmax(cv_scores))]
# OUTPUT: "Best k = 1 (cross-validation 94.9%). Rule of thumb k = 101 only scored 88.0%."
print(f"Best k = {best_k} (cross-validation {max(cv_scores):.1%}). "
      f"Rule of thumb k = {root_n + 1} only scored {cv_scores[CANDIDATES.index(root_n + 1)]:.1%}.")

# ---- STEP 6: only now, test the chosen k once on the 2,000 locked-away digits ----
final = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train).score(X_test, y_test)
# OUTPUT: "Test set with k = 1: 94.4%"
print(f"Test set with k = {best_k}: {final:.1%}")

# ---- STEP 7: draw the graph ----
# OUTPUT: a window: accuracy against k, for new digits (blue) and memorised digits (grey)
if SHOW_PLOTS:
    import matplotlib.pyplot as plt   # matplotlib: draws the graphs

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(CANDIDATES, np.array(train_scores) * 100, "o--", color="#8A93A3", lw=2, label="training digits (memorised)")
    ax.plot(CANDIDATES, np.array(cv_scores) * 100, "o-", color="#2F6FD0", lw=2, label="cross-validation (new digits)")
    ax.axvline(root_n + 1, color="#8A93A3", ls=":", lw=1.5)
    ax.text(root_n + 1, 99, " rule of thumb: k = √n", color="#5B6472", va="top")
    ax.annotate(f"best: k = {best_k}", (best_k, max(cv_scores) * 100), xytext=(12, -28),
                textcoords="offset points", color="#2F6FD0", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#2F6FD0"))
    ax.set_xscale("log")                                # 1, 10, 100, 1000 equally spaced
    ax.set_xticks(CANDIDATES)
    ax.set_xticklabels(CANDIDATES)
    ax.minorticks_off()
    ax.set_xlabel("k (how many neighbours vote)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Choosing k: the best k scores highest on digits it hasn't seen")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
