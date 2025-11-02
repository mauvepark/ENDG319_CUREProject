import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# task 1a
data = load_breast_cancer()
a = pd.DataFrame(data.data, columns=data.feature_names)
a["typeofcancer"] = data.target

print(a.columns.tolist())

# task 1(b)
cols = ["mean radius", "mean perimeter", "mean area"]
id = [list(data.feature_names).index(c) for c in cols]
df = a.iloc[:, id].copy()
df["typeofcancer"] = a["typeofcancer"]

print(df.head())

# task 1b first 2 rows
print(df.head(2))

# task 1b rows 17-21
print(df.loc[17:21])

# task 2
fig1, axes = plt.subplots(1, 3, figsize=(12, 3.5))

# subplot1 histogram
axes[0].hist(
    df[df["typeofcancer"] == 0]["mean radius"],
    bins=20, histtype="step", color="blue", label="c0 (Malignant)"
)
axes[0].hist(
    df[df["typeofcancer"] == 1]["mean radius"],
    bins=20, histtype="step", color="red", label="c1 (Benign)"
)
axes[0].set_title("Histogram of mean radius by class")
axes[0].set_xlabel("mean radius")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# subplot 2 mean radius vs mean perimeter
axes[1].scatter(
    df[df["typeofcancer"] == 0]["mean perimeter"],
    df[df["typeofcancer"] == 0]["mean radius"],
    color="blue", s=15, label="c0 (M)"
)
axes[1].scatter(
    df[df["typeofcancer"] == 1]["mean perimeter"],
    df[df["typeofcancer"] == 1]["mean radius"],
    color="orange", s=15, label="c1 (B)"
)
axes[1].set_xlabel("mean perimeter")
axes[1].set_ylabel("mean radius")
axes[1].legend()

# subplot 3 mean radius vs mean area
axes[2].scatter(
    df[df["typeofcancer"] == 0]["mean area"],
    df[df["typeofcancer"] == 0]["mean radius"],
    color="blue", s=15, label="c0 (M)"
)
axes[2].scatter(
    df[df["typeofcancer"] == 1]["mean area"],
    df[df["typeofcancer"] == 1]["mean radius"],
    color="orange", s=15, label="c1 (B)"
)
axes[2].set_xlabel("mean area")
axes[2].set_ylabel("mean radius")
axes[2].legend()

fig1.tight_layout()
plt.savefig("figure1.png", dpi=200, bbox_inches="tight")
plt.close(fig1)

# task 3

fig2, axes = plt.subplots(1, 4, figsize=(16, 3.5))

# subplot 1 histogram
axes[0].hist(
    a[a["typeofcancer"] == 0]["mean radius"],
    bins=20, histtype="step", color="blue", label="c0 (Malignant)"
)
axes[0].hist(
    a[a["typeofcancer"] == 1]["mean radius"],
    bins=20, histtype="step", color="red", label="c1 (Benign)"
)
axes[0].set_title("Histogram of mean radius by class")
axes[0].set_xlabel("mean radius")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# subplot 2 mean radius vs mean concavity
axes[1].scatter(
    a[a["typeofcancer"] == 0]["mean concavity"],
    a[a["typeofcancer"] == 0]["mean radius"],
    color="blue", s=15, label="c0 (M)"
)
axes[1].scatter(
    a[a["typeofcancer"] == 1]["mean concavity"],
    a[a["typeofcancer"] == 1]["mean radius"],
    color="orange", s=15, label="c1 (B)"
)
axes[1].set_xlabel("mean concavity")
axes[1].set_ylabel("mean radius")
axes[1].legend()

# subplot 3 mean radius vs mean concave points
axes[2].scatter(
    a[a["typeofcancer"] == 0]["mean concave points"],
    a[a["typeofcancer"] == 0]["mean radius"],
    color="blue", s=15, label="c0 (M)"
)
axes[2].scatter(
    a[a["typeofcancer"] == 1]["mean concave points"],
    a[a["typeofcancer"] == 1]["mean radius"],
    color="orange", s=15, label="c1 (B)"
)
axes[2].set_xlabel("mean concave points")
axes[2].set_ylabel("mean radius")
axes[2].legend()

# subplot 4 mean radius vs mean symmetry
axes[3].scatter(
    a[a["typeofcancer"] == 0]["mean symmetry"],
    a[a["typeofcancer"] == 0]["mean radius"],
    color="blue", s=15, label="c0 (M)"
)
axes[3].scatter(
    a[a["typeofcancer"] == 1]["mean symmetry"],
    a[a["typeofcancer"] == 1]["mean radius"],
    color="orange", s=15, label="c1 (B)"
)
axes[3].set_xlabel("mean symmetry")
axes[3].set_ylabel("mean radius")
axes[3].legend()

fig2.tight_layout()
plt.savefig("figure2.png", dpi=200, bbox_inches="tight")
plt.close(fig2)

# task 4

task4 = pd.DataFrame({
    "heart rate range(bpm)": [145 - 51, 159 - 60, 68 - 48, 118 - 57, 160 - 50, 136 - 48, 147 - 46, 143 - 45, 96 - 58, 124 - 48],
    "hours of sleep": [7.01, 7.06, 8.49, 5.50, 5.95, 5.33, 6.17, 5.16, 7.65, 5.15],
    "stress level": ["low","low","low","low","high","high","high","high","low","high"]
})
print(task4)