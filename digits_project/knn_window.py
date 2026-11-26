"""
knn_window.py: the drawing window that knn.py opens at the end
==============================================================
Yann Imanzi Shema · A-level Computer Science

Run knn.py and, after its score, this window opens. (You can also run this
file on its own.) Draw a number in each box with the mouse, or with a finger
or a stylus on a touch screen or tablet (leave a small gap between digits),
choose + - x or /, then press "Read and solve". The top bar shows the accuracy.

For every digit you drew, the window shows:
  - what the computer sees: your drawing shrunk to 28 x 28 pixels, like MNIST
  - KNN's guess, and the probability of each digit 0 to 9
Then it works out the answer to the sum.

Where the probability comes from: KNN finds the 10 training digits that are
closest to your drawing. If 8 of them are 7s and 2 are 1s, it says
"7: 80%, 1: 20%". That is exactly what  knn.predict_proba()  gives back.

KNN only knows the digits 0 to 9, because that is all MNIST has. So the
+ - x / signs are buttons, not drawn.
"""

# ---- STEP 1: import the libraries ----
import tkinter as tk                                  # built into Python: windows, buttons and a canvas to draw on
import numpy as np                                    # numpy: fast maths on big grids of numbers
from PIL import Image, ImageDraw                      # Pillow (comes with matplotlib): a hidden copy of the drawing as pixels

PEN = 18          # pen thickness in the drawing box, in screen pixels
BOX_W, BOX_H = 380, 220


# ---- STEP 2: turn a drawing into an MNIST-style digit ----
def split_digits(ink):
    """Cut a drawing into separate digits, wherever there is an empty column between them."""
    has_ink = ink.max(axis=0) > 0                     # True for every column with some ink in it
    pieces, start = [], None
    for x, inked in enumerate(has_ink):
        if inked and start is None:
            start = x                                 # a digit starts here
        elif not inked and start is not None:
            pieces.append(ink[:, start:x])            # a digit ends here
            start = None
    if start is not None:
        pieces.append(ink[:, start:])
    return [p for p in pieces if p.sum() > 255 * 40]  # ignore tiny dots


def to_mnist(ink):
    """Make one drawn digit look like an MNIST digit: 28 x 28, the digit in a 20 x 20 box, centred."""
    rows = np.where(ink.max(axis=1) > 0)[0]
    cols = np.where(ink.max(axis=0) > 0)[0]
    crop = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]           # cut off the empty space
    h, w = crop.shape
    scale = 20 / max(h, w)                                           # MNIST digits fit in a 20 x 20 box
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.array(Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS), dtype=float)
    small = np.clip(small, 0, 255)
    ys, xs = np.indices(small.shape)                                 # centre of mass, like MNIST did
    cy, cx = (ys * small).sum() / small.sum(), (xs * small).sum() / small.sum()
    top = min(max(int(round(14 - cy)), 0), 28 - new_h)
    left = min(max(int(round(14 - cx)), 0), 28 - new_w)
    grid = np.zeros((28, 28))
    grid[top:top + new_h, left:left + new_w] = small
    return grid / 255.0                                              # 0 to 1, like load_data() gives


# ---- STEP 3: read a whole number with KNN ----
def read_number(knn, ink):
    """Returns the number, plus each digit's 28 x 28 picture and its 10 probabilities."""
    digits = []
    for piece in split_digits(ink):
        pixels = to_mnist(piece)
        probs = knn.predict_proba(pixels.reshape(1, 784))[0]          # share of the 10 neighbours per digit
        guess = int(knn.classes_[probs.argmax()])                     # the digit with the most votes
        digits.append((pixels, guess, probs))
    if not digits:
        return None, []
    number = int("".join(str(g) for _, g, _ in digits))               # e.g. digits 4 and 7 -> 47
    return number, digits


def solve(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "x":
        return a * b
    if b == 0:
        return "can't divide by 0"
    answer = a / b
    return int(answer) if answer == int(answer) else round(answer, 3)


# ---- STEP 4: the window ----
# The look: every colour and the font are set here once, so the whole window can be restyled in one place.
BG, CARD, LINE = "#EEF1F6", "#FFFFFF", "#D5DBE5"      # the page, the white cards, the card borders
INK, MUTED = "#1C2230", "#5B6472"                     # dark text (also the top bar) and grey text
BLUE, BLUE_DARK, BAR = "#2F6FD0", "#2458A8", "#C9D3E3"  # the guess, a button under the mouse, the other bars
GREEN, TIP, TIP_INK = "#7FDBA0", "#E3ECFB", "#24407A"  # the accuracy, the tablet tip and its text
FONT = "Segoe UI"                                     # Windows' font; on other computers Tk picks the closest one
SHOW = {"+": "+", "-": "−", "x": "×", "/": "÷"}       # how each sign is shown on screen


def flat_button(parent, text, command, colour, text_colour, hover_colour):
    """A flat coloured button: a label you can click. (Plain tk buttons ignore colours on a Mac.)"""
    button = tk.Label(parent, text=text, bg=colour, fg=text_colour, font=(FONT, 13, "bold"),
                      padx=26, pady=10, cursor="hand2")
    button.bind("<Button-1>", lambda event: command())  # a click, a tap or a stylus press
    button.bind("<Enter>", lambda event: button.config(bg=hover_colour))
    button.bind("<Leave>", lambda event: button.config(bg=colour))
    return button


class DrawBox:
    """A box you draw in with the mouse, a finger or a stylus. It keeps a hidden pixel copy for KNN."""

    def __init__(self, parent, title):
        self.frame = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE, padx=12, pady=10)
        tk.Label(self.frame, text=title, font=(FONT, 12, "bold"), bg=CARD, fg=INK).pack(anchor="w")
        tk.Label(self.frame, text="leave a small gap between digits", font=(FONT, 9),
                 bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(self.frame, width=BOX_W, height=BOX_H, bg="white",
                                highlightthickness=2, highlightbackground=LINE, cursor="pencil")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.pen_down)   # on a touch screen, a finger or stylus works like the mouse
        self.canvas.bind("<B1-Motion>", self.pen_move)
        self.canvas.bind("<Enter>", lambda event: self.canvas.config(highlightbackground=BLUE))
        self.canvas.bind("<Leave>", lambda event: self.canvas.config(highlightbackground=LINE))
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.canvas.create_text(BOX_W / 2, BOX_H / 2, text="draw here", fill="#B9C0CC", font=(FONT, 15), tags="hint")
        self.ink = Image.new("L", (BOX_W, BOX_H), 0)  # 0 = no ink, 255 = full ink (white on black, like MNIST)
        self.pen = ImageDraw.Draw(self.ink)
        self.last = None

    def dot(self, x, y):
        r = PEN / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=INK, outline=INK)
        self.pen.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def pen_down(self, event):
        self.canvas.delete("hint")                    # hide "draw here" once you start
        self.last = (event.x, event.y)
        self.dot(event.x, event.y)

    def pen_move(self, event):
        x0, y0 = self.last
        self.canvas.create_line(x0, y0, event.x, event.y, width=PEN, fill=INK, capstyle=tk.ROUND)
        self.pen.line([x0, y0, event.x, event.y], fill=255, width=PEN)
        self.dot(event.x, event.y)
        self.last = (event.x, event.y)

    def pixels(self):
        return np.array(self.ink)


def digit_panel(parent, pixels, guess, probs):
    """One card per digit: what the computer sees, its guess, and a bar for each digit's probability."""
    panel = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE, padx=10, pady=8)
    pic = tk.Canvas(panel, width=84, height=84, bg=INK, highlightthickness=0)
    for r in range(28):                               # draw the 28 x 28 pixels, 3 screen pixels each
        for c in range(28):
            v = pixels[r, c]
            if v > 0.04:
                shade = "#%02x%02x%02x" % tuple(int(dark + (255 - dark) * v) for dark in (28, 34, 48))
                pic.create_rectangle(c * 3, r * 3, c * 3 + 3, r * 3 + 3, fill=shade, width=0)
    pic.grid(row=0, column=0, rowspan=2)
    tk.Label(panel, text=f"It's a {guess}", font=(FONT, 15, "bold"), bg=CARD, fg=INK).grid(row=0, column=1, sticky="w", padx=10)
    tk.Label(panel, text=f"{probs[guess]:.0%} sure", font=(FONT, 10, "bold"), bg=CARD, fg=BLUE).grid(row=0, column=2, sticky="e")
    bars = tk.Canvas(panel, width=210, height=74, bg=CARD, highlightthickness=0)
    for d in range(10):
        x = 4 + d * 20.5
        top = 58 - probs[d] * 44
        bars.create_rectangle(x, top, x + 15, 58, fill=BLUE if d == guess else BAR, width=0)
        bars.create_text(x + 7.5, 68, text=str(d), font=(FONT, 9), fill=INK if d == guess else MUTED)
        if probs[d] > 0:
            bars.create_text(x + 7.5, top - 7, text=f"{probs[d]:.0%}", font=(FONT, 7), fill=MUTED)
    bars.grid(row=1, column=1, columnspan=2, padx=(10, 0))
    return panel


class App:
    def __init__(self, root, knn, accuracy):
        self.knn = knn                                 # the trained KNN model from knn.py
        root.title("Digit reader: KNN reads your sum and solves it")
        root.configure(bg=BG)

        # the top bar: what the window does on the left, the model's accuracy on the right
        bar = tk.Frame(root, bg=INK, padx=22, pady=12)
        bar.pack(fill="x")
        bar.columnconfigure(0, weight=1)
        tk.Label(bar, text="Draw a sum. KNN reads it and solves it.", font=(FONT, 17, "bold"),
                 bg=INK, fg="white").grid(row=0, column=0, sticky="w")
        tk.Label(bar, text=f"K-nearest neighbours · k = {knn.n_neighbors} · learnt from {knn.n_samples_fit_:,} MNIST digits",
                 font=(FONT, 10), bg=INK, fg="#AEB7C6").grid(row=1, column=0, sticky="w")
        if accuracy is not None:
            tk.Label(bar, text=f"{accuracy:.1%}", font=(FONT, 24, "bold"), bg=INK, fg=GREEN).grid(row=0, column=1, sticky="e")
            tk.Label(bar, text="accuracy on test digits it never saw", font=(FONT, 9),
                     bg=INK, fg="#AEB7C6").grid(row=1, column=1, sticky="e")

        # the tablet tip
        tip = tk.Frame(root, bg=TIP, padx=22, pady=7)
        tip.pack(fill="x")
        tk.Label(tip, text="Tablets and touch screens:", font=(FONT, 10, "bold"), bg=TIP, fg=TIP_INK).pack(side="left")
        tk.Label(tip, text="draw with a finger or a stylus. A touch works just like the mouse.",
                 font=(FONT, 10), bg=TIP, fg=TIP_INK).pack(side="left", padx=4)

        main = tk.Frame(root, bg=BG, padx=22, pady=16)
        main.pack(fill="both", expand=True)
        top = tk.Frame(main, bg=BG)
        top.pack()
        self.box1 = DrawBox(top, "Number 1")
        self.box1.frame.grid(row=0, column=0)
        signs = tk.Frame(top, bg=BG)
        signs.grid(row=0, column=1, padx=14)
        self.sign_buttons = {}
        for sign in ["+", "-", "x", "/"]:              # big enough to tap with a finger
            button = tk.Label(signs, text=SHOW[sign], width=3, font=(FONT, 20, "bold"), cursor="hand2",
                              highlightthickness=1, highlightbackground=LINE)
            button.bind("<Button-1>", lambda event, s=sign: self.choose_sign(s))
            button.pack(pady=4)
            self.sign_buttons[sign] = button
        self.choose_sign("+")
        self.box2 = DrawBox(top, "Number 2")
        self.box2.frame.grid(row=0, column=2)

        buttons = tk.Frame(main, bg=BG)
        buttons.pack(pady=14)
        flat_button(buttons, "Read and solve", self.read_and_solve, BLUE, "white", BLUE_DARK).pack(side="left", padx=6)
        flat_button(buttons, "Clear", self.clear, CARD, INK, "#E3E7EE").pack(side="left", padx=6)

        self.answer = tk.Label(main, text="", font=(FONT, 30, "bold"), bg=BG, fg=INK)
        self.answer.pack()
        self.details = tk.Frame(main, bg=BG)
        self.details.pack(pady=(6, 0))
        tk.Label(main, text=f"Each bar = the share of the {knn.n_neighbors} closest training digits that are that digit (knn.predict_proba).",
                 font=(FONT, 9), bg=BG, fg=MUTED).pack(side="bottom", pady=(12, 0))
        self.clear()

    def choose_sign(self, sign):
        self.op = sign                                 # remember the sign, and colour its button blue
        for s, button in self.sign_buttons.items():
            button.config(bg=BLUE if s == sign else CARD, fg="white" if s == sign else INK)

    def clear(self):
        self.box1.clear()
        self.box2.clear()
        self.answer.config(text="Your answer appears here", fg="#B0B8C6")
        for w in self.details.winfo_children():
            w.destroy()

    def show_digits(self, row, title, digits):
        tk.Label(self.details, text=title, font=(FONT, 11, "bold"), bg=BG, fg=MUTED).grid(row=row, column=0, sticky="ne", padx=8, pady=6)
        for i, (pixels, guess, probs) in enumerate(digits):
            digit_panel(self.details, pixels, guess, probs).grid(row=row, column=i + 1, padx=5, pady=6)

    def read_and_solve(self):
        for w in self.details.winfo_children():
            w.destroy()
        a, digits1 = read_number(self.knn, self.box1.pixels())
        b, digits2 = read_number(self.knn, self.box2.pixels())
        if a is None:
            self.answer.config(text="Draw a number in box 1 first", fg="#B54708")
            return
        self.show_digits(0, "Number 1", digits1)
        if b is None:                                  # only one number: just show what KNN read
            text = f"You wrote {a}"
            shown = text
        else:
            self.show_digits(1, "Number 2", digits2)
            text = f"{a} {self.op} {b} = {solve(a, self.op, b)}"
            shown = f"{a} {SHOW[self.op]} {b} = {solve(a, self.op, b)}"
        self.answer.config(text=shown, fg=INK)
        # OUTPUT: e.g. "47 + 5 = 52", then each digit's top probabilities
        print(text)
        for name, digits in [("Number 1", digits1), ("Number 2", digits2)]:
            for _, guess, probs in digits:
                best = np.argsort(probs)[::-1][:3]
                print(f"  {name}: read a {guess}   " + ", ".join(f"{d}: {probs[d]:.0%}" for d in best if probs[d] > 0))


# ---- STEP 5: open the window ----
def open_window(knn, accuracy=None):
    """knn.py calls this at the end with a trained KNN model and its accuracy on the test digits."""
    root = tk.Tk()
    App(root, knn, accuracy)
    root.mainloop()                                   # the window stays open until you close it


if __name__ == "__main__":                            # running this file on its own
    from sklearn.neighbors import KNeighborsClassifier
    from data import load_data
    X_train, X_test, y_train, y_test = load_data()
    knn = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
    open_window(knn, knn.score(X_test, y_test))       # score() = the share of test digits it gets right
