import os
import time
import numpy as np
import pandas as pd

def read_market_data(path: str) -> pd.DataFrame:
    """
    Given a path to a file about an asset market data in csv format, returns a pandas dataframe with the data,
    sorted by time dimension in ascending order.
    """
    # Reads file
    df = pd.read_csv(path, header=0, sep=',', parse_dates=True)

    # Sets date as index
    df.set_index('date', inplace=True)

    # Sorts the dataframe ascending by date
    df.sort_index(inplace=True)
    
    return df

def calculate_targets(market_data: pd.DataFrame, time_periods: list) -> pd.DataFrame:
    """
    Given a pandas dataframe with market data of a certain asset and a list of time periods, 
    returns a dataframe with the log-returns of the asset same periods shifted back.
    """
    # Initializes a new dataframe to store the log-returns auxiliary data
    df = pd.DataFrame(index=market_data.index)

    for period in time_periods:
        # Computes the log-returns for the given period
        df[f'log_returns_{period}'] = np.log(market_data['close'].shift(-period) / market_data['close'])
    
    # Drops NaN values
    df.dropna(inplace=True)

    return df

# Step 1: Reads raw market data
input_path = 'market_data/BTC.csv'
output_path = 'data/y.csv'
os.makedirs('data', exist_ok=True)

print(f"Reading market data from {input_path}...")

start_time = time.time()
market_data = read_market_data(input_path)
end_time = time.time()

print(f"Raw market data loaded in {end_time - start_time:.2f} seconds.")
print(f"Rows: {market_data.shape[0]}, Columns: {market_data.shape[1]}")
print(f"Columns: {list(market_data.columns)}\n")

# Step 2: Calculates targets
print("Calculating targets...")

time_periods = list(range(1, 30))

print(f"Using forward time periods: {time_periods} (days)")

start_time = time.time()
targets = calculate_targets(market_data, time_periods)
end_time = time.time()

print(f"Targets calculation completed in {end_time - start_time:.2f} seconds.")
print(f"Rows: {targets.shape[0]}, Columns: {targets.shape[1]}\n")

# Step 3: Saves targets to csv
print("Saving targets to csv...")
targets.to_csv(output_path, index=True, header=True)