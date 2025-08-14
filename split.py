import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Defines data paths
market_data_path = Path("market_data")
derived_data_path = Path("derived_data")

# Reads all market data files
market_data_files = sorted(market_data_path.glob("*.csv"))

for asset in market_data_files:
    symbol = Path(asset).stem.upper()
    features_path = Path(f"derived_data/{symbol}/X_pca.csv")
    targets_path = Path(f"derived_data/{symbol}/y.csv")

    X = pd.read_csv(features_path, index_col='date')
    y = pd.read_csv(targets_path, index_col='date')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    X_train.to_csv(Path(f"{derived_data_path}/{symbol}/X_train.csv"), index=True, header=True)
    X_test.to_csv(Path(f"{derived_data_path}/{symbol}/X_test.csv"), index=True, header=True)
    y_train.to_csv(Path(f"{derived_data_path}/{symbol}/y_train.csv"), index=True, header=True)
    y_test.to_csv(Path(f"{derived_data_path}/{symbol}/y_test.csv"), index=True, header=True)