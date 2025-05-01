import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

def load_features(file_path: str) -> pd.DataFrame:
    """
    Loads features from a csv file and returns a DataFrame.
    """
    X = pd.read_csv(file_path, header=0, sep=',', parse_dates=True)
    X.set_index('date', inplace=True)

    return X

def load_targets(file_path: str) -> pd.DataFrame:
    """
    Loads targets from a csv file and returns a DataFrame.
    """
    y = pd.read_csv(file_path, header=0, sep=',', parse_dates=True)
    y.set_index('date', inplace=True)

    return y

def calculate_mi(X: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses kNN-based MI estimator to compute the mutual information between features and targets.
    """
    mi_matrix = pd.DataFrame(index=X.columns)
    
    for target_col in target_df.columns:
        y = target_df[target_col]
        mi_scores = mutual_info_regression(X, y)
        mi_matrix[target_col] = mi_scores
    
    return mi_matrix

# Step 1: Loads features
features_path = 'data/X.csv'
target_path = 'data/y.csv'
output_path = 'data/X_pca.csv'

print(f"Loading features from {features_path}...")

start_time = time.time()
X = load_features(features_path)
end_time = time.time()

print(f"Features loaded in {end_time - start_time:.2f} seconds.")
print(f"Rows: {X.shape[0]}, Columns: {X.shape[1]}\n")

# Step 2: Loads targets
file_path = 'y.csv'

print(f"Loading targets from {target_path}...")

start_time = time.time()
y = load_targets(target_path)
end_time = time.time()

print(f"Targets loaded in {end_time - start_time:.2f} seconds.")
print(f"Rows: {y.shape[0]}, Columns: {y.shape[1]}\n")

# Step 3: Aligns features and targets
print("Aligning features and targets...\n")

X, y = X.align(y, axis=0, join='inner')
X.to_csv(features_path, index=True, header=True)
y.to_csv(target_path, index=True, header=True)

# Step 4: Calculates mutual information
print("Calculating mutual information...")

start_time = time.time()
mi_matrix = calculate_mi(X, y)
end_time = time.time()

print(f"Mutual information calculation completed in {end_time - start_time:.2f} seconds.")
# print(mi_matrix)

# Step 5: Group-wise PCA
print("Performing group-wise PCA...")

grouped_components = []
group_names = ['rsi', 'ema', 'atr', 'mfi']
explained_variances = {}

for group in group_names:
    group_cols = [col for col in X.columns if group in col]
    if not group_cols:
        print(f"Skipped '{group}' — no matching columns.")
        continue

    print(f"Applying PCA to group '{group}' with {len(group_cols)} features...")
    pca = PCA(n_components=0.95)  # keep 99% of variance
    group_data = X[group_cols]
    X_group_pca = pca.fit_transform(group_data)

    print(f" - Components kept: {pca.n_components_}")
    print(f" - Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    explained_variances[group] = np.sum(pca.explained_variance_ratio_)
    X_group_pca_df = pd.DataFrame(X_group_pca,
                                  columns=[f"{group}_pca_{i}" for i in range(X_group_pca.shape[1])],
                                  index=X.index)
    grouped_components.append(X_group_pca_df)

# Concatenate all PCA-reduced groups
X = pd.concat(grouped_components, axis=1)

# Also keeps MACD (not PCA-reduced)
# macd = [col for col in X.columns if ('macd_hist_std' in col) or ('macd_hist_fast' in col)]
# X_macd = X[macd]

# Concatenates PCA features with MACD
# X = pd.concat([X_pca, X_macd], axis=1)

# Step 6: Saves PCA features to csv
print("Saving features to csv...")
X.to_csv(output_path, index=True, header=True)