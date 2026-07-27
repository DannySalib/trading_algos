# %% [markdown]
# # Optimizing Basket Selection via Independence

# %%
import pandas as pd 

data = pd.read_parquet('./sp500_stocks.paqruet')
data.head()

# %%
data.Close.dropna(how='all', axis=1, inplace=True)

# %%
data.Close.drop(columns=['SPY'], inplace=True)

# %%
import numpy as np 
returns = data.Close.apply(lambda x: np.log1p(x.pct_change())).dropna(how='all', axis=0)

# %%
corr = returns.corr()

# %%
dist = 1 - corr.abs()      # ignore sign

# %%
dist = dist.fillna(1)

# %%
X_dist = dist.values.copy()
np.fill_diagonal(X_dist, 0.0)

# %%
# %%
import kmedoids
from scipy.spatial.distance import squareform

X_dist_square = squareform(X_dist)  # not needed again if reused; keep square form (n x n) for KMedoids
D = dist.values.copy()
np.fill_diagonal(D, 0.0)

km = kmedoids.KMedoids(200, method='fasterpam', random_state=42)
result = km.fit(D)
labels = result.labels_

# medoid_indices are the actual "most representative / most independent" picks
medoid_idx = result.medoid_indices_
tickers = dist.columns.to_numpy()
independent_basket = tickers[medoid_idx]

print(independent_basket)


