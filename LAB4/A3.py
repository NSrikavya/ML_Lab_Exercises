import numpy as np
import matplotlib.pyplot as plt

# Generate 20 random points (X, Y) between 1 and 10
np.random.seed(42)  # for reproducibility
X = np.random.randint(1, 11, 20)
Y = np.random.randint(1, 11, 20)

# Assign classes (simple rule: X+Y < 10 -> class 0, else class 1)
classes = np.where(X + Y < 10, 0, 1)

# Scatter Plot
plt.figure(figsize=(6, 6))
for i in range(len(X)):
    if classes[i] == 0:
        plt.scatter(X[i], Y[i], color="blue", label="Class 0 (Blue)" if i == 0 else "")
    else:
        plt.scatter(X[i], Y[i], color="red", label="Class 1 (Red)" if i == 1 else "")

plt.title("Scatter Plot of Training Data (20 Points)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.grid(True)
plt.show()
