# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.14.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Machine Learning - Session 5

# %% [markdown]
# ## Neural Networks

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# %%
data = load_breast_cancer()
X = data.data
y = data.target

print("Feature names:", data.feature_names)
print("Target names:", data.target_names)
print("Data shape:", X.shape)

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=1)

mlp.fit(X_train, y_train)

# %%
y_pred = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Accuracy:", accuracy_score(y_test, y_pred))

# %%
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# %% [markdown]
# ### Bonus: Activation Functions

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. Define activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    # Gaussian Error Linear Unit approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 2. Generate input range
x = np.linspace(-4, 4, 1000)

# 3. Plotting
plt.figure(figsize=(10, 6))

plt.plot(x, relu(x), label="ReLU", linewidth=2.5)
plt.plot(x, leaky_relu(x), label="LeakyReLU (α=0.1)", linewidth=2, linestyle="--")
plt.plot(x, elu(x), label="ELU (α=1.0)", linewidth=2, linestyle="-.")
plt.plot(x, gelu(x), label="GELU", linewidth=2.5, linestyle=":")

# 4. Styling and Layout
plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
plt.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
plt.title("Comparison of Activation Functions: ReLU vs. LeakyReLU vs. ELU vs. GELU", fontsize=12)
plt.xlabel("Input (x)", fontsize=10)
plt.ylabel("Output f(x)", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11, loc="upper left")
plt.ylim(-1.5, 4)

plt.show()
