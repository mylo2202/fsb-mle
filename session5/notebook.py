# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# ### Step 1: Library Imports
#
# * **`load_breast_cancer`**: Built-in benchmark dataset for binary medical classification.
# * **`train_test_split`**: Splits data into training and validation/test subsets.
# * **`StandardScaler`**: Scales features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
# * **`MLPClassifier`**: Scikit-learn's feedforward artificial neural network (Multi-Layer Perceptron).
# * **`classification_report`, `confusion_matrix`, `accuracy_score`**: Standard evaluation tools.
# * **`matplotlib.pyplot`**: Plotting training curves and activation functions.

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# %% [markdown]
# ### Step 2: Dataset Exploration
#
# * **Dataset Shape**: `(569, 30)` — $569$ samples and $30$ numerical features (e.g., mean radius, texture, perimeter).
# * **Target Classes**: `['malignant', 'benign']` (encoded as binary `0` and `1`).

# %%
# type: ignore
data = load_breast_cancer()
X = data.data
y = data.target

print("Feature names:", data.feature_names)
print("Target names:", data.target_names)
print("Data shape:", X.shape)

# %% [markdown]
# ### Step 3: Train-Test Split & Feature Scaling
#
# **Why Scaling is Critical for Neural Networks:**
#
# 1. **Gradient Descent Stability**: Unscaled features with large numerical ranges dominate weight updates and cause oscillations or exploding/vanishing gradients.
# 2. **Faster Convergence**: Normalized inputs allow gradient descent / Adam optimizer to converge much faster.
# 3. **Data Leakage Prevention**:
#    * `scaler.fit_transform(X_train)`: Computes $\mu$ and $\sigma$ strictly from the **training set** and transforms it.
#    * `scaler.transform(X_test)`: Uses the **training statistics** to transform the test set without learning from it.

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ### Step 4: Model Architecture & Training (`MLPClassifier`)
#
# * **Network Architecture**:
#   * **Input Layer**: $30$ neurons (matches the number of input features).
#   * **Hidden Layer 1**: $10$ neurons.
#   * **Hidden Layer 2**: $5$ neurons.
#   * **Output Layer**: Binary classification output.
# * **Hyperparameters**:
#   * `hidden_layer_sizes=(10, 5)`: Defines depth (2 hidden layers) and width (10 and 5 neurons).
#   * `max_iter=1000`: Number of training epochs allowed before stopping if convergence is not reached.
#   * `random_state=1`: Ensures reproducible weight initialization and shuffle orders.
# * **Default Scikit-Learn Settings Applied Behind the Scenes**:
#   * **Activation**: `activation='relu'` (Rectified Linear Unit for hidden layers).
#   * **Optimizer**: `solver='adam'` (Adaptive Moment Estimation).
#   * **Loss Function**: Binary Cross-Entropy (Log-loss).

# %%
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=1)

mlp.fit(X_train, y_train)

# %% [markdown]
# ### Step 5: Prediction & Model Evaluation
#
# * **Output Results**:
#   * **Accuracy**: $\approx 98.25\%$ ($112$ out of $114$ correct predictions).
#   * **Confusion Matrix**:
#     $$\begin{bmatrix} 42 & 1 \\ 1 & 70 \end{bmatrix}$$
#     * 42 True Malignant (Class 0), 1 False Positive.
#     * 70 True Benign (Class 1), 1 False Negative.
# * **Metrics Interpreted**:
#   * **Precision** ($\frac{TP}{TP + FP}$): Out of all positive predictions, how many were correct? ($\approx 98\%\text{--}99\%$)
#   * **Recall** ($\frac{TP}{TP + FN}$): Out of all actual positives, how many did the model detect? ($\approx 98\%\text{--}99\%$)
#   * **F1-Score**: Harmonic mean of Precision and Recall ($2 \times \frac{P \times R}{P + R}$).

# %%
# type: ignore
y_pred = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Accuracy:", accuracy_score(y_test, y_pred))

# %% [markdown]
# ### Step 6: Visualizing the Loss Curve
#
# * **`mlp.loss_curve_`**: Records the value of the loss function at each iteration (epoch).
# * **Takeaway**: A smoothly descending curve that plateaus near zero demonstrates healthy convergence without wild oscillations or early divergence.

# %%
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# %% [markdown]
# ### Bonus: Activation Functions
#
# | Activation | Formula | Pros | Cons / Notes |
# | :--- | :--- | :--- | :--- |
# | **ReLU** | $f(x) = \max(0, x)$ | Computationally efficient; reduces vanishing gradient for $x > 0$. | **Dying ReLU problem**: If $x \le 0$, gradient is 0 and neuron permanently deactivates. |
# | **Leaky ReLU** | $f(x) = x \text{ if } x > 0 \text{ else } \alpha x$ | Small slope ($\alpha = 0.1$) for $x < 0$ prevents dying neurons. | Requires selecting slope $\alpha$. |
# | **ELU** | $f(x) = x \text{ if } x > 0 \text{ else } \alpha(e^x - 1)$ | Smooth transition at 0; negative saturation brings mean activation closer to 0. | Slightly higher computational cost due to $\exp(x)$. |
# | **GELU** | $f(x) = x \cdot \Phi(x)$ | Smooth, non-monotonic; probabilistically retains inputs. | Industry standard in modern Transformers (BERT, GPT, ViT). |
#
# **Key Visual Differences to Note**
#
# - **ReLU:** Flat at zero for all negative values, creating a sharp angle at x=0.
#     
# - **LeakyReLU:** Shows a steady downward-sloping linear line for negative values, ensuring non-zero gradients.
#     
# - **ELU:** Smoothly curves down toward an asymptote at y=−1.0 for negative inputs.
#     
# - **GELU:** Features a subtle, non-monotonic dip below zero near x=−1.0 before smoothly flattening out.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Define activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return x * norm.cdf(x)

# 2. Generate input range
x = np.linspace(-4, 4, 1000)

# 3. Plotting
plt.figure(figsize=(10, 6))

plt.plot(x, relu(x), label="ReLU", linewidth=2.5)
plt.plot(x, leaky_relu(x), label="LeakyReLU (alpha=0.1)", linewidth=2, linestyle="--")
plt.plot(x, elu(x), label="ELU (alpha=1.0)", linewidth=2, linestyle="-.")
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

# %% [markdown]
# ### Bonus: MLPClassifier on Non-Linear Interlocking Moons

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

# 1. Generate non-linear interlocking moons with randomness/noise
X, y = make_moons(
    n_samples=800,
    noise=0.25,        # Adds random Gaussian dispersion
    random_state=42
)

# 2. Scale features (CRITICAL for neural network convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize MLPClassifier
# Architecture: 2 hidden layers with 32 and 16 neurons
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    random_state=42
)

# 4. Train across epochs using partial_fit
epochs = 120
classes = np.unique(y)
loss_curve = []
accuracy_curve = []

for epoch in range(epochs):
    # Shuffle each epoch for stochastic minibatch updates
    indices = np.random.permutation(len(X_scaled))
    X_shuffled = X_scaled[indices]
    y_shuffled = y[indices]
    
    # partial_fit updates weights incrementally
    mlp.partial_fit(X_shuffled, y_shuffled, classes=classes)
    
    # Evaluate performance
    probs = mlp.predict_proba(X_scaled)
    preds = mlp.predict(X_scaled)
    
    loss_curve.append(log_loss(y, probs))
    accuracy_curve.append(accuracy_score(y, preds))

# 5. Visualizations
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Non-Linear Decision Boundary
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8)
ax1.set_title("MLP Decision Boundary (32, 16 Architecture)")
ax1.set_xlabel("Scaled Feature 1")
ax1.set_ylabel("Scaled Feature 2")
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Cross-Entropy Loss Curve
ax2.plot(range(1, epochs + 1), loss_curve, color='crimson', linewidth=2)
ax2.set_title("Cross-Entropy Loss Decay")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Log Loss")
ax2.grid(True, linestyle=':', alpha=0.6)

# # Plot 3: Accuracy Curve
# ax3.plot(range(1, epochs + 1), accuracy_curve, color='forestgreen', linewidth=2)
# ax3.set_title("Training Accuracy Curve")
# ax3.set_xlabel("Epoch")
# ax3.set_ylabel("Accuracy")
# ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Bonus: Non-Linear Regression with MLPRegressor

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic non-linear data (Sinusoidal wave + noise)
np.random.seed(42)
n_samples = 600
X = np.sort(np.random.uniform(-3, 3, size=(n_samples, 1)), axis=0)
# Underlying true function: y = sin(2x) + 0.5x + noise
y_true = np.sin(2 * X.ravel()) + 0.5 * X.ravel()
y = y_true + np.random.normal(0, 0.35, size=n_samples)

# 2. Scale features (CRITICAL for neural network convergence)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# 3. Initialize MLPRegressor
# Architecture: 2 hidden layers with 64 and 32 neurons
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    alpha=0.001,          # L2 regularization weight decay
    random_state=42
)

# 4. Train across epochs using partial_fit
epochs = 150
loss_curve = []

for epoch in range(epochs):
    # Shuffle each epoch
    indices = np.random.permutation(len(X_scaled))
    X_shuffled = X_scaled[indices]
    y_shuffled = y[indices]
    
    # partial_fit incrementally updates network weights
    mlp_reg.partial_fit(X_shuffled, y_shuffled)
    
    # Predict and record full Mean Squared Error (MSE)
    y_pred = mlp_reg.predict(X_scaled)
    current_loss = mean_squared_error(y, y_pred)
    loss_curve.append(current_loss)

# 5. Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fitted Non-Linear Curve vs Raw Noisy Data
x_grid = np.linspace(-3.2, 3.2, 300).reshape(-1, 1)
x_grid_scaled = scaler_X.transform(x_grid)
y_grid_pred = mlp_reg.predict(x_grid_scaled)

ax1.scatter(X, y, color='steelblue', alpha=0.5, edgecolors='k', label='Noisy Training Data')
ax1.plot(x_grid, np.sin(2 * x_grid.ravel()) + 0.5 * x_grid.ravel(), 
         color='black', linestyle='--', linewidth=2, label='True Signal (No Noise)')
ax1.plot(x_grid, y_grid_pred, color='crimson', linewidth=2.5, label='MLP Fitted Curve')

ax1.set_title("MLPRegressor: Fitting Non-Linear Target Function")
ax1.set_xlabel("Feature X")
ax1.set_ylabel("Target y")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: MSE Loss Decay Curve
ax2.plot(range(1, epochs + 1), loss_curve, color='crimson', linewidth=2)
ax2.set_title("Training Loss Curve (MSE vs Epochs)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Squared Error (MSE)")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

print(f"Final Training MSE Loss: {loss_curve[-1]:.4f}")
