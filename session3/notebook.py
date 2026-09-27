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
#     display_name: .venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Machine Learning - Session 3

# %% [markdown]
# ## `LinearRegression`

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3

# %%
reg = LinearRegression().fit(X, y)

# %%
reg.score(X, y)

# %%
reg.coef_

# %%
reg.intercept_

# %%
reg.predict(np.array([[3, 5]]))

# %% [markdown]
# ## `SGDRegressor`

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# 1. Generate synthetic regression data with noise
X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=15.0,
    bias=10.0,
    random_state=42
)

# 2. Scale features (CRITICAL for SGD gradient stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize SGDRegressor
sgd = SGDRegressor(
    loss='squared_error',
    learning_rate='invscaling',  # Shrinks step size over time
    eta0=0.01,                    # Higher initial rate for faster early progress
    random_state=42,
    verbose=5
)

# 4. Train over multiple epochs using partial_fit to record Mean Squared Error (MSE)
epochs = 20
loss_curve = []

for epoch in range(epochs):
    # Shuffle each epoch
    indices = np.random.permutation(len(X_scaled))
    X_shuffled = X_scaled[indices]
    y_shuffled = y[indices]
    
    sgd.partial_fit(X_shuffled, y_shuffled)
    
    # Compute full-dataset MSE loss
    y_pred = sgd.predict(X_scaled)
    current_loss = np.mean((y_pred - y) ** 2) / 2.0
    loss_curve.append(current_loss)

# 5. Plot Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Regression Fit over Raw Data
ax1.scatter(X, y, color='steelblue', alpha=0.6, edgecolors='k', label='Data Points')
# Plot prediction line (unscale X for plotting)
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_line_scaled = scaler.transform(x_line)
y_line = sgd.predict(x_line_scaled)

ax1.plot(x_line, y_line, color='crimson', linewidth=2.5, label='SGD Fitted Line')
ax1.set_title("SGDRegressor: Data & Fitted Model")
ax1.set_xlabel("Feature X")
ax1.set_ylabel("Target y")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Training Loss Curve
ax2.plot(range(1, epochs + 1), loss_curve, color='darkorange', linewidth=2, marker='o', markersize=4)
ax2.set_title("SGD Training Loss Curve (MSE vs Epochs)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (MSE)")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# Print learned parameters
print(f"Final Learned Coefficient (Slope): {sgd.coef_[0]:.4f}")
print(f"Final Learned Intercept (Bias): {sgd.intercept_[0]:.4f}")

# %% [markdown]
# ## `LogisticRegression`

# %%
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(X.shape, y.shape)

# %%
clf = LogisticRegression(random_state=0).fit(X, y)
sample_index_min = 40
sample_index_max = 400
clf.predict(X[sample_index_min:sample_index_max, :])

# %%
clf.predict_proba(X[sample_index_min:sample_index_max, :])

# %%
clf.score(X, y)

# %% [markdown]
# ## `SGDClassifier`

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

# 1. Generate synthetic binary classification dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,          # 2D features for easy visualization
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=0.2,
    flip_y=0.10,           # 10% label noise (overlap between classes)
    random_state=42
)

# 2. Scale features (CRITICAL for SGD gradient stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize SGDClassifier as Logistic Regression
# loss='log_loss' fits a logistic regression model
sgd_clf = SGDClassifier(
    loss='log_loss',
    learning_rate='invscaling',
    eta0=0.3,
    power_t=0.25,
    random_state=42,
    verbose=10
)

# 4. Train over epochs using partial_fit
epochs = 100
classes = np.unique(y)
loss_curve = []
accuracy_curve = []

for epoch in range(epochs):
    # Shuffle dataset each epoch
    indices = np.random.permutation(len(X_scaled))
    X_shuffled = X_scaled[indices]
    y_shuffled = y[indices]
    
    # partial_fit requires specifying unique classes on first call
    sgd_clf.partial_fit(X_shuffled, y_shuffled, classes=classes)
    
    # Compute probability predictions and hard predictions
    probs = sgd_clf.predict_proba(X_scaled)
    preds = sgd_clf.predict(X_scaled)
    
    # Track metrics
    current_loss = log_loss(y, probs)
    current_acc = accuracy_score(y, preds)
    
    loss_curve.append(current_loss)
    accuracy_curve.append(current_acc)

# 5. Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

# Plot 1: Decision Boundary & Data Points
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = sgd_clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8)
ax1.set_title("Learned Decision Boundary")
ax1.set_xlabel("Scaled Feature 1")
ax1.set_ylabel("Scaled Feature 2")
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Binary Cross-Entropy Loss Curve
ax2.plot(range(1, epochs + 1), loss_curve, color='crimson', linewidth=2)
ax2.set_title("Log Loss (Cross-Entropy) vs Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Log Loss")
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
