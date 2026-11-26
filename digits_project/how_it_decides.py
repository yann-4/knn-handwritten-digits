"""
how_it_decides.py: how each model decides which digit it is
===========================================================
Yann Imanzi Shema · A-level Computer Science

My other files call  model.predict()  and just get an answer back.
This file opens predict() up. It takes ONE test digit (test digit 3,
a handwritten 9) and does each model's decision BY HAND with numpy,
printing every stage. Then it checks that my hand-made answer is
exactly what scikit-learn's predict() said.

  KNN                  measure the distance to all 10,000 training digits; the closest decides
  Logistic regression  10 scores = pixels x weights + bias; softmax; the highest wins
  K-means              measure the distance to the 10 cluster centres; use the nearest one's name
  Neural network       pixels -> 128 ReLU neurons -> 10 scores; softmax; the highest wins

The models are built exactly as in knn.py, logistic_regression.py,
kmeans_clustering.py and neural_network.py, so they give the same answers.
"""

# ---- STEP 1: import the libraries ----
import warnings                                       # built into Python: hides "did not fully converge" messages
import numpy as np                                    # numpy: does the maths by hand in this file
from sklearn.neighbors import KNeighborsClassifier    # scikit-learn: the four models
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

from data import load_data                            # my own file: loads the digits from the CSV files

warnings.filterwarnings("ignore")

# ---- STEP 2: load the data and pick one test digit ----
X_train, X_test, y_train, y_test = load_data()
N = 3                                                 # test digit number 3 (counting from 0)
x = X_test[N]                                         # its 784 pixels, each 0 to 1
# OUTPUT: "Test digit 3. The real answer is 9...", then the digit drawn with # for ink
print(f"Test digit {N}. The real answer is {y_test[N]}, but the models are never told that.")
for row in x.reshape(28, 28)[3:25]:                   # 28 rows of 28 pixels (skip the empty top and bottom)
    print("    " + "".join("#" if p > 0.5 else "+" if p > 0.1 else "." for p in row))


def softmax(scores):
    # Turns 10 scores into 10 probabilities that add up to 100%. A bigger score gets a bigger share.
    e = np.exp(scores - scores.max())
    return e / e.sum()


def same(mine, sklearn):
    return f"my answer {mine}, scikit-learn's predict() {sklearn}: {'SAME' if mine == sklearn else 'DIFFERENT'}"


# ---- STEP 3: KNN, by hand ----
# OUTPUT: the 5 closest training digits with their distances, then the decision
print("\n1. KNN: which training digits look most like it?")
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)   # k = 1, the best k from knn.py
distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))             # distance to ALL 10,000 (Pythagoras over 784 pixels)
closest = np.argsort(distances)[:5]                                # positions of the 5 smallest distances
for rank, i in enumerate(closest, 1):
    print(f"   {rank}. training digit {i:5d} is a {y_train[i]}   distance {distances[i]:.2f}")
knn_guess = y_train[closest[0]]                                    # k = 1: copy the answer of the closest one
votes = np.bincount(y_train[closest], minlength=10)                # (with k = 5 they would vote)
print(f"   k = 1, so the closest one decides: it is a {knn_guess}.  (With k = 5 the vote is "
      + ", ".join(f"{votes[d]} for {d}" for d in np.argsort(-votes) if votes[d]) + ")")
print("   " + same(knn_guess, knn.predict([x])[0]))

# ---- STEP 4: logistic regression, by hand ----
# OUTPUT: a score for each digit, its probability after softmax, then the decision
print("\n2. Logistic regression: one score per digit")
logreg = LogisticRegression(C=0.1, max_iter=1000).fit(X_train, y_train)
W, b = logreg.coef_, logreg.intercept_                # W: 10 rows of 784 weights (one row per digit), b: 10 biases
scores = W @ x + b                                    # for each digit: every pixel x its weight, all added up, + bias
probs = softmax(scores)                               # 10 scores -> 10 probabilities
for d in range(10):
    print(f"   {d}: score {scores[d]:6.2f}  ->  {probs[d]:6.1%}  {'#' * round(probs[d] * 40)}")
lr_guess = scores.argmax()                            # the highest score wins
print(f"   The highest score is for {lr_guess}, so it is a {lr_guess}.")
print("   " + same(lr_guess, logreg.predict([x])[0])
      + f"  (probabilities match predict_proba(): {np.allclose(probs, logreg.predict_proba([x])[0])})")

# ---- STEP 5: K-means, by hand ----
# OUTPUT: the distance to each of the 10 cluster centres, nearest first, then the decision
print("\n3. K-means: which of the 10 cluster centres is it nearest to?")
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X_train)
cluster_names = np.array([np.bincount(y_train[kmeans.labels_ == c], minlength=10).argmax()
                          for c in range(10)])        # name each cluster after its most common real digit
centre_dist = np.sqrt(((kmeans.cluster_centers_ - x) ** 2).sum(axis=1))   # distance to each of the 10 centres
for c in np.argsort(centre_dist):
    print(f"   cluster {c} (named {cluster_names[c]})   distance {centre_dist[c]:.2f}")
nearest = centre_dist.argmin()                        # the nearest centre wins
km_guess = cluster_names[nearest]
print(f"   The nearest centre is cluster {nearest}, named {km_guess}, so it is a {km_guess}.")
print("   " + same(km_guess, cluster_names[kmeans.predict([x])[0]]))

# ---- STEP 6: the neural network (ReLU), by hand ----
# OUTPUT: how many hidden neurons fired, then the 10 output scores and probabilities, then the decision
print("\n4. Neural network (ReLU): 784 pixels -> 128 neurons -> 10 outputs")
net = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", max_iter=60, random_state=42).fit(X_train, y_train)
W1, W2 = net.coefs_                                   # W1: 784 x 128 weights, W2: 128 x 10 weights
b1, b2 = net.intercepts_                              # b1: 128 biases, b2: 10 biases
hidden = np.maximum(0, x @ W1 + b1)                   # each neuron: weighted sum of the pixels + bias, then ReLU (below 0 -> 0)
print(f"   Hidden layer: {(hidden > 0).sum()} of the 128 neurons fired. ReLU set the other {(hidden == 0).sum()} to 0.")
out = hidden @ W2 + b2                                # each output: weighted sum of the 128 neurons + bias
probs = softmax(out)
for d in range(10):
    print(f"   {d}: score {out[d]:6.2f}  ->  {probs[d]:6.1%}  {'#' * round(probs[d] * 40)}")
nn_guess = out.argmax()                               # the highest output wins
print(f"   The highest output is for {nn_guess}, so it is a {nn_guess}.")
print("   " + same(nn_guess, net.predict([x])[0])
      + f"  (probabilities match predict_proba(): {np.allclose(probs, net.predict_proba([x])[0])})")

# ---- STEP 7: sum up ----
# OUTPUT: the four answers side by side
print(f"\nThe real answer: {y_test[N]}.  KNN: {knn_guess}   Logistic regression: {lr_guess}   "
      f"K-means: {km_guess}   Neural network: {nn_guess}")
