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
# # Machine Learning - Session 6

# %% [markdown]
# ## Support Vector Machines with `SVC`
#
# The code below creates a simple 2D dataset with two classes.  
# We fit a Support Vector Classifier using a pipeline that includes `StandardScaler`, which normalizes the feature values before training. This is important because SVMs are sensitive to feature scale.
#
# `SVC(gamma='auto')` creates a support vector machine with an RBF kernel, which can separate non-linear patterns in the data.

# %%
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)

# %%
print(clf.predict([[-0.8, -1]]))

# %% [markdown]
# The prediction above applies the learned decision boundary to a new sample and returns the class label assigned to that point.
#
# In this example, the model is trying to separate the two groups based on the input coordinates.

# %% [markdown]
# ## K-means with `KMeans`
#
# The next example uses the `KMeans` clustering algorithm to partition data into groups based on similarity.  
# `n_clusters=2` tells the algorithm to find two centroids and assign each point to the nearest one.
#
# This is an unsupervised learning method, so there are no target labels during training.

# %%
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_

# %%
kmeans.predict([[0, 0], [12, 3]])

# %%
kmeans.cluster_centers_

# %% [markdown]
# The `labels_` array shows which cluster each training point belongs to.  
# `predict()` assigns new observations to the nearest cluster center, and `cluster_centers_` reports the final centroid positions learned by the model.
