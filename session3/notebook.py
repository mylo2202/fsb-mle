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
#     display_name: .venv (3.14.4.final.0)
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
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

# 1. Scaling features is mandatory for SGD
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Initialize SGDRegressor
model = SGDRegressor(
    loss="squared_error",
    penalty=None,          # Pure linear regression (no L2/L1 penalty)
    learning_rate="constant",
    eta0=0.01,             # Learning rate alpha
    random_state=42,
    max_iter=1000, tol=1e-3, verbose=1
)

# 3. Train epoch-by-epoch and track loss
epochs = 100
loss_curve = []

for epoch in range(epochs):
    # Perform one pass over the data
    model.partial_fit(X_scaled, y)
    
    # Calculate current Mean Squared Error
    y_pred = model.predict(X_scaled)
    current_loss = mean_squared_error(y, y_pred)
    loss_curve.append(current_loss)

# 4. Plot the recorded loss curve
plt.plot(range(1, epochs + 1), loss_curve)
plt.title("SGDRegressor Convergence Curve")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()

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
