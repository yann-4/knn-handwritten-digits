"""
why_not_if_statements.py: why not just write if-statements?
===========================================================
Yann Imanzi Shema · A-level Computer Science

Traditional programming = I write the rules. So could I recognise digits
with if-statements? This file tries, on the same 2,000 test digits.

  1. My own rules: 6 if-statements about simple measurements of the ink
     (how much ink, how wide, how much is in the top half or the left half).
  2. A decision tree: a computer chooses the if-statements for me,
     straight from the 784 pixels. More levels = more rules.

Handwriting varies too much (size, slant, thickness, style) for a short
list of rules, and nobody can write hundreds of pixel rules by hand.
That's why machine learning learns the rules from examples instead.
"""

# ---- STEP 1: import the libraries ----
import numpy as np                                 # numpy: fast maths on big grids of numbers
from sklearn.tree import DecisionTreeClassifier    # scikit-learn: a tree of if-statements chosen by the computer

from data import load_data                         # my own file: loads the digits from the CSV files

# ---- STEP 2: load the data ----
X_train, X_test, y_train, y_test = load_data()

# ---- STEP 3: my own if-statements ----
# Simple measurements a person might think of. A pixel counts as ink if it is over half dark.
ink_grid = X_test.reshape(-1, 28, 28) > 0.5
ink = ink_grid.sum(axis=(1, 2))                    # how many ink pixels
width = ink_grid.any(axis=1).sum(axis=1)           # how many columns have ink
top = ink_grid[:, :14].sum(axis=(1, 2))            # ink in the top half
left = ink_grid[:, :, :14].sum(axis=(1, 2))        # ink in the left half


def my_rules(i):
    if width[i] <= 7:                              # thin, so probably a 1
        return 1
    if ink[i] < 110:                               # not much ink, so maybe a 7
        return 7
    if top[i] > 0.6 * ink[i]:                      # top-heavy: a 9 or a 7
        return 9 if left[i] < 0.5 * ink[i] else 7
    if ink[i] > 180:                               # lots of ink: an 8 or a 0
        return 8 if width[i] < 17 else 0
    if left[i] > 0.55 * ink[i]:                    # left-heavy: maybe a 6
        return 6
    return 3                                       # otherwise guess 3


guesses = np.array([my_rules(i) for i in range(len(y_test))])
# OUTPUT: "My 6 if-statements: 20.7% correct", then how often each digit was right
print(f"My 6 if-statements: {(guesses == y_test).mean():.1%} correct")
print("  Right per digit: " + "  ".join(f"{d}: {(guesses[y_test == d] == d).mean():.0%}" for d in range(10)))

# ---- STEP 4: let the computer choose the if-statements (a decision tree) ----
# OUTPUT: one line per tree size, e.g. "  16 if-rules: 64.2% correct"
print("\nA decision tree: the computer picks the if-statements from the 784 pixels")
for levels in [2, 4, 6, 10]:
    tree = DecisionTreeClassifier(max_depth=levels, random_state=0)
    tree.fit(X_train, y_train)
    print(f"  {tree.get_n_leaves():4d} if-rules: {tree.score(X_test, y_test):.1%} correct")

# OUTPUT: the comparison with machine learning
print("\nFor comparison, learning from examples: KNN 94.4%, neural network (ReLU) 95.8%")
