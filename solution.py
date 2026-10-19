import numpy as np


def load_data(filename):
    """Read the CSV (skipping the header) and return (age, hg, cls)."""
    data = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=int)
    age = data[:, 0]   # age_years
    hg = data[:, 1]    # blood_pressure_mmhg
    cls = data[:, 2]   # stroke, raw 0/1 labels
    return age, hg, cls


def perceptron(age, hg, cls, N=1_000_000):
    """Run the perceptron and return (w0, w_age, w_hg)."""
    rng = np.random.default_rng(42)  # random seed for reproducibility

    age = np.asarray(age, dtype=float)
    hg = np.asarray(hg, dtype=float)
    cls = np.asarray(cls)

    # Each point becomes [1, age, hg]; the 1 lets w0 act as the bias
    X = np.column_stack((np.ones(len(age)), age, hg))
    # Convert labels internally: 0 -> -1, 1 -> +1
    y = np.where(cls == 1, 1, -1)

    w = np.zeros(3)
    for _ in range(N):
        # Stop as soon as every point is classified correctly
        if np.all(np.sign(X @ w) == y):
            break
        i = rng.integers(len(y))          # pick a random point
        if np.sign(X[i] @ w) != y[i]:     # misclassified?
            w = w + y[i] * X[i]           # perceptron update

    w0, w_age, w_hg = w
    return w0, w_age, w_hg


if __name__ == "__main__":
    age, hg, cls = load_data("stroke_separable.csv")
    w0, w_age, w_hg = perceptron(age, hg, cls)
    pred = np.sign(w0 + w_age * np.array(age) + w_hg * np.array(hg))
    labels = np.where(np.array(cls) == 1, 1, -1)
    print("weights:", w0, w_age, w_hg)
    print("all correct:", np.all(pred == labels))