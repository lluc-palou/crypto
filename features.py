import time
import warnings
import talib as ta
import numpy as np
import pandas as pd

def read_market_data(file_path: str) -> pd.DataFrame:
    """
    Given a path to a file about an asset market data in csv format, returns a pandas dataframe with the data,
    sorted by time dimension in ascending order.
    """
    # Reads file
    df = pd.read_csv(file_path, header=0, sep=',', parse_dates=True)

    # Sets date as index
    df.set_index('date', inplace=True)

    # Sorts the dataframe ascending by date
    df.sort_index(inplace=True)
    
    return df

def calculate_features(market_data: pd.DataFrame, time_periods: list) -> pd.DataFrame:
    """
    Given a pandas dataframe with market data of a certain asset, returns a new dataframe with the calculated 
    features of the asset. Features are technical indicators of TA-Lib computed for different time periods.
    """
    # Initializes a new dataframe to store the derived features auxiliary data
    df = pd.DataFrame(index=market_data.index)

    # Trend indicator
    for p in time_periods:
        df[f'ema_{p}'] = ta.EMA(market_data['close'], timeperiod=p)
        
    # Momentum indicator
    for p in time_periods:
        df[f'rsi_{p}'] = ta.RSI(market_data['close'], timeperiod=p)

    # Volatility indicator
    for p in time_periods:
        df[f'atr_{p}'] = ta.ATR(high=market_data['high'], low=market_data['low'],
                                close=market_data['close'], timeperiod=p)
    
    # Volume indicator
    for p in time_periods:
        df[f'mfi_{p}'] = ta.MFI(high=market_data['high'], 
                                low=market_data['low'], 
                                close=market_data['close'], 
                                volume=market_data['volume'], 
                                timeperiod=p)
    
    # Drops NaN values
    df.dropna(inplace=True)
    
    return df

class SequentialStandardizer:
    """
    This class implements a sequential standardization of features in a DataFrame. It uses Welford's algorithm 
    to compute the mean and variance in a single pass. The standardization is done row-by-row, meaning that each 
    row is standardized using the statistics of the previous rows.
    """
    def __init__(self):
        self.stats = {}  # Stores {'feature': {'n', 'mean', 'M2'}}
        self.fitted_features = set()
    
    def _update_stats(self, feature: str, new_value: float) -> None:
        """
        Updates running stats for one sample.
        """
        if feature not in self.stats:
            self.stats[feature] = {'n': 0, 'mean': 0.0, 'M2': 0.0}
        
        stats = self.stats[feature]
        stats['n'] += 1
        delta = new_value - stats['mean']
        stats['mean'] += delta / stats['n']
        stats['M2'] += delta * (new_value - stats['mean'])
    
    def _get_standardized(self, feature: str, value: float) -> float:
        """
        Standardizes one value using current statatistics.
        """
        if feature not in self.stats or self.stats[feature]['n'] < 2:
            return 0.0  # No scaling for first sample
        
        stats = self.stats[feature]
        std = np.sqrt(stats['M2'] / (stats['n'] - 1))
        return (value - stats['mean']) / (std + 1e-8)
    
    def transform_sequentially(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes a time-ordered DataFrame row-by-row.
        Each row is standardized using statistics from previous rows only.
        """
        standardized = pd.DataFrame(index=df.index, columns=df.columns)
        
        for i in range(len(df)):
            # Standardizes current row using past data stats
            for feature in df.columns:
                standardized.iloc[i][feature] = self._get_standardized(
                    feature, df.iloc[i][feature]
                )
                # Update statistics with current row (for next samples standardization)
                self._update_stats(feature, df.iloc[i][feature])
        
        return standardized

if __name__ == "__main__":
    # Defines data paths
    input_path = 'data/market_data/BTC.csv'
    output_path = 'data/X.csv'

    # Step 1: Reads market data
    print(f"Reading market data from {input_path}...")

    start_time = time.time()
    market_data = read_market_data(input_path)
    end_time = time.time()

    print(f"Raw market data loaded in {end_time - start_time:.2f} seconds.")
    print(f"Rows: {market_data.shape[0]}, Columns: {market_data.shape[1]}")
    print(f"Columns: {list(market_data.columns)}")

    # Step 2: Calculates features
    print("Calculating features...")

    time_periods = [2, 3, 5, 7, 10, 14, 15, 21, 30, 42, 50, 60, 75, 90]

    print(f"Using time windows of: {time_periods} (days)")

    start_time = time.time()
    features = calculate_features(market_data, time_periods)
    end_time = time.time()

    print(f"Feature calculation completed in {end_time - start_time:.2f} seconds.")
    print(f"Rows: {features.shape[0]}, Columns: {features.shape[1]}")

    # Step 3: Standardizes features sequentially
    print("Standardizing features sequentially...")

    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        standardizer = SequentialStandardizer()
        features_norm = standardizer.transform_sequentially(features)
    end_time = time.time()

    print(f"Feature standardization completed in {end_time - start_time:.2f} seconds.")
    print(f"Rows: {features.shape[0]}, Columns: {features.shape[1]}")

    features_norm_df = pd.DataFrame(
        features_norm,
        columns=features.columns,
        index=features.index
    ).astype(np.float64)

    features_norm_df.iloc[:2, :] = np.nan        # Sets first two rows to NaN as they are not standardized
    features_norm_df = features_norm_df.dropna() # Drops NaN values

    # Step 4: Saves standardized features to csv
    print("Saving standardized features to csv...")
    features_norm_df.to_csv(output_path, index=True, header=True)