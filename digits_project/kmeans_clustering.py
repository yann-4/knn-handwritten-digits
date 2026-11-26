"""
kmeans_clustering.py: K-means clustering (unsupervised: no labels!)
===================================================================
Yann Imanzi Shema · A-level Computer Science

K-means is never told which digit is which. It just looks for groups:
  1. drop k centres at random
  2. give every image to its nearest centre
  3. move each centre to the average of its images
  4. repeat 2 and 3 until nothing moves

Don't mix it up with KNN: same letter k, totally different job.
KNN's k = how many neighbours vote. K-means' k = how many groups to find.

To score it, I name each cluster after the digit most common inside it.
(That's the only time labels are used, and only for marking.)
"""

# ---- STEP 1: import the libraries ----
import time                       # built into Python: a stopwatch for the timings

import numpy as np                # numpy: fast maths on big grids of numbers
from sklearn.cluster import KMeans  # scikit-learn: the ready-made K-means model

from data import load_data        # my own file: loads and splits the digits

SHOW_PLOTS = True   # False = text only, no pictures

# ---- STEP 2: load the data ----
# X = the pictures (784 pixel numbers per digit), y = the answers (0 to 9).
# X_train, y_train: 10,000 digits to learn from. X_test, y_test: 2,000 to test on.
# OUTPUT: load_data() prints "Training on 10,000 digits, testing on 2,000 unseen digits."
X_train, X_test, y_train, y_test = load_data()

# ---- STEP 3: build the model ----
start = time.time()
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)

# ---- STEP 4: train it, WITHOUT the answers ----
kmeans.fit(X_train)                                   # look: no y_train!

# ---- STEP 5: test it on the 2,000 unseen digits, print the score ----
# Name each cluster after the real digit most common inside it (answers used only for marking).
# How it predicts one digit: find the nearest of the 10 cluster centres, use that cluster's name.
cluster_names = np.array([np.bincount(y_train[kmeans.labels_ == c], minlength=10).argmax()
                          for c in range(10)])
km_acc = (cluster_names[kmeans.predict(X_test)] == y_test).mean()
# OUTPUT: "K-means (10 clusters, no labels): 57.7% correct  (4.7s)"
print(f"K-means (10 clusters, no labels): {km_acc:.1%} correct  ({time.time() - start:.1f}s)")

# ---- STEP 6: look closer: what did it name its clusters? ----
# OUTPUT: "It named its 10 clusters: [0, 1, 1, 2, ...]" and "Missing digits: [5]"
print(f"  It named its 10 clusters: {sorted(cluster_names.tolist())}")
print(f"  Missing digits: {sorted(set(range(10)) - set(cluster_names.tolist()))}  (it found two kinds of 1 instead)")

# ---- STEP 7: draw the graph ----
# OUTPUT: a window with the 10 cluster centres (the average picture of each group)
if SHOW_PLOTS:
    import matplotlib.pyplot as plt   # matplotlib: draws the graphs

    fig, axes = plt.subplots(1, 10, figsize=(14, 2.6))
    fig.suptitle("K-means' 10 cluster centres: the average image of each group, found without labels")
    for d in range(10):
        axes[d].imshow(kmeans.cluster_centers_[d].reshape(28, 28), cmap="gray")
        axes[d].set_title(f"looks like {cluster_names[d]}", fontsize=10)
        axes[d].axis("off")
    plt.tight_layout()
    plt.show()
