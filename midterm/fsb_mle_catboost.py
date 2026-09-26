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

# %% [markdown] id="e59f2c36"
# # CatBoost - Group 6

# %% [markdown] id="Y3--4mEYX2-l"
# ## Install Libraries

# %% colab={"base_uri": "https://localhost:8080/"} id="rWMjOEQdIQXz" outputId="ae52257d-0733-4bf9-8490-fde19839f110"
# %pip install catboost

# %% [markdown] id="ed64f483"
# ## CatBoost in Regression
#
# We will use the **California Housing Dataset** from `scikit-learn`.

# %% id="8373b2fd"
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# %% colab={"base_uri": "https://localhost:8080/"} id="91df5770" outputId="7efb5dda-2432-481a-c7e9-13f5c1349ce2"
# 1. Load the California Housing Dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

# Display dataset info
print("California Housing Dataset Loaded:")
print(f"Samples: {len(data)}, Features: {len(data.columns) - 1}")
print(data.head())

# %% id="cc679a17"
# 2. Separate features (X) and target (y)
target = "MedHouseVal"
X = data.drop(columns=[target])
y = data[target]

# %% id="707932ae"
# 3. Identify categorical features (none in the California Housing dataset)
cat_features = []
# No categorical columns to cast to string, so we bypass that step.

# %% id="92abb08b"
# 4. Train-Test Split using Scikit-Learn (regression task, so we omit 'stratify')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% id="9b4c94ac"
# 5. Initialize CatBoostRegressor (for predicting continuous house values)
model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    random_seed=42,
    verbose=100
)

# %% colab={"base_uri": "https://localhost:8080/"} id="e564f64e" outputId="55d2a50a-c0ac-4121-c294-e8c839b231d0"
# 6. Fit the model on training data
model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),  # Optional validation set for early stopping
    early_stopping_rounds=30,  # Stop training if validation score doesn't improve for 30 rounds
)

# %% colab={"base_uri": "https://localhost:8080/"} id="a7327e52" outputId="51d2f309-e57c-469d-b023-a8d16c2c3b11"
# 7. Make predictions and evaluate using regression metrics
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# %% [markdown] id="84f11be6"
# ### CatBoost Feature Ranking
#
# Calculate and visualize feature importances directly from a trained model using the `get_feature_importance()` method.

# %% id="68760d2d"
import pandas as pd
import matplotlib.pyplot as plt

# %% id="0d444743"
# type: ignore
# 1. Calculate default feature importance scores (PredictionValuesChange)
feature_importances = model.get_feature_importance()
feature_names = X.columns

# %% colab={"base_uri": "https://localhost:8080/"} id="f5074af5" outputId="13143ed9-8cb4-4be3-9041-bd4f47286274"
# 2. Create a pandas DataFrame for neat display
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("--- Feature Importances ---")
print(importance_df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 410} id="473989f6" outputId="b456de66-1b62-4679-f353-e87dccce1585"
# 3. Plot the feature importances
plt.figure(figsize=(8, 4))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('CatBoost Feature Importances')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()

# %% [markdown] id="41bceac9"
# ### SHAP Values
#
# **SHAP** stands for **SHapley Additive exPlanations**. It is a widely used framework in machine learning for model interpretability, designed to explain individual predictions by measuring how much each feature contributed to a specific output.
#
# While standard feature importance tells you which features are generally important across an entire dataset, **SHAP values explain** **why** **a specific prediction was made for a single instance**.
#
# ### Interpreting SHAP Values
#
# * **Positive SHAP Value**: The feature value pushed the prediction **higher** than the baseline average.
# * **Negative SHAP Value**: The feature value pulled the prediction **lower** than the baseline average.
# * **Magnitude**: The absolute size of the value indicates how strongly that feature influenced the decision for that specific row.
#
# ### Calculating SHAP Values in CatBoost
#
# CatBoost provides built-in support for generating SHAP values, which can be combined with the external `shap` library for visualization

# %% colab={"base_uri": "https://localhost:8080/"} id="k2zunFppz_es" outputId="4946cbb7-b4ec-4b07-b23a-b52de1d2a4da"
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Returns Explanation object of shape (samples, features)
print(shap_values.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 476} id="pR2erDCv9I4q" outputId="b459f5f9-206e-4010-f753-8662774b3b31"
shap.summary_plot(shap_values, X_test)

# %% [markdown] id="WRxUi-t3aS0_"
# ## CatBoost in Classification
#
# We will use the **Iris Dataset** from `scikit-learn`.

# %% id="dU0SSfFSaS1C"
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# %% id="uF0VoGrdcTmD"
# 1. Load the Iris dataset
iris = load_iris(as_frame=True)
X_iris = iris.data
y_iris = iris.target

# %% id="Ty0UwxfVcRIS"
# 2. Split into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# %% id="SKjSBibEcPIh"
# 3. Initialize CatBoostClassifier
clf_model = CatBoostClassifier(
    iterations=150,
    learning_rate=0.05,
    depth=4,
    random_seed=42,
    verbose=50
)

# %% colab={"base_uri": "https://localhost:8080/"} id="zrA4PXDqcMeq" outputId="139630c9-b0ed-4b54-8b41-a368bb296c92"
# 4. Fit the classifier
clf_model.fit(
    X_train_iris,
    y_train_iris,
    eval_set=(X_test_iris, y_test_iris),
    early_stopping_rounds=15
)

# %% colab={"base_uri": "https://localhost:8080/"} id="kvsbW-rKcKY6" outputId="d535f228-be67-4c84-b7d4-305c5f004477"
# 5. Predict and evaluate
y_pred_iris = clf_model.predict(X_test_iris)
accuracy = accuracy_score(y_test_iris, y_pred_iris)

print("\n--- Iris Classification Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))

# %% [markdown] id="YnhbP0yMaS1L"
# ### CatBoost Feature Ranking
#
# Calculate and visualize feature importances directly from a trained model using the `get_feature_importance()` method.

# %% id="yxlSfY9NaS1N"
import pandas as pd
import matplotlib.pyplot as plt

# %% id="OAqZjK13aS1N"
# type: ignore
# 1. Calculate default feature importance scores for Iris classification
feature_importances = clf_model.get_feature_importance()
feature_names = X_iris.columns

# %% colab={"base_uri": "https://localhost:8080/"} id="xdm4SjPkaS1O" outputId="07f1181f-2301-4467-d9ff-a4575d8ed1bb"
# 2. Create a pandas DataFrame for neat display
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("--- Iris Feature Importances ---")
print(importance_df)

# %% colab={"base_uri": "https://localhost:8080/", "height": 410} id="8i03jYndaS1O" outputId="7194b4c3-be58-42be-a9f4-d84f56e3fb52"
# 3. Plot the feature importances for Iris dataset
plt.figure(figsize=(8, 4))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('CatBoost Feature Importances (Iris Classification)')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()

# %% [markdown] id="SQ34ROCMyxi7"
# ### SHAP Values with `shap.TreeExplainer()`

# %% colab={"base_uri": "https://localhost:8080/"} id="PFbvY0FQy2MB" outputId="01db368e-4f3b-4b65-e8c7-638341f0b934"
import shap

explainer = shap.TreeExplainer(clf_model)
shap_exp = explainer(X_test_iris)

# Shape: (samples, features, classes)
print(shap_exp.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 316} id="Wy_6h30Y8acP" outputId="f614094b-d97c-40ef-e0f7-015bf39198d5"
# Single-class summary plot
shap.summary_plot(shap_exp[:, :, 0], X_test_iris)

# %% colab={"base_uri": "https://localhost:8080/", "height": 317} id="nympY1yh8OeX" outputId="0329a1fb-0ff8-4ad0-e267-94f88ea4c871"
# Multi-class bar summary plot for all classes at once
shap.summary_plot(
    [shap_exp.values[:, :, i] for i in range(3)],
    X_test_iris,
    class_names=iris.target_names,
)
