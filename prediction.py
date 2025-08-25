import os
import json
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

def load_top_models(symbol: str, models_dir: str, architectures_dir: str, logs_dir: str):
    top_models_info = []

    for path in glob(os.path.join(Path(f"{logs_dir}/{symbol}/model_ranking"), "ranking_*.csv")):
        quantile = os.path.basename(path).replace("ranking_", "").replace(".csv", "")
        df = pd.read_csv(path)

        if df.empty:
            print(f"Skipping {quantile}: ranking is empty.")
            continue

        best_model = df.iloc[0].copy()
        model_id = str(best_model["model_id"])
        target = best_model["target"]

        model_path = os.path.join(Path(f"{models_dir}/{symbol}"), quantile, f"{model_id}_{target}.h5")
        architecture_path = os.path.join(Path(f"{architectures_dir}/{symbol}"), quantile, f"{model_id}.json")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        if not os.path.exists(architecture_path):
            print(f"Architecture file not found: {architecture_path}")
            continue
        
        with open(architecture_path, "r") as f:
            config = json.load(f)

        model = load_model(model_path)
        window_size = config.get("window size")

        top_models_info.append({
            "quantile": quantile,
            "model_id": model_id,
            "target": target,
            "model": model,
            "window_size": window_size,
            "ranking_row": best_model
        })

    return top_models_info

def reshape_data(X, y, window_size):
    """
    Reshapes the datasets into input sequences and corresponding targets required for the model.
    """
    # Converts the dataframes to numpy arrays
    if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
    if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values

    X_seq, y_seq = [], []

    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size])

    return np.array(X_seq), np.array(y_seq)

def get_last_window(X, window_size):
    """
    Extracts the last window of features from the test set for prediction.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    
    last_window = X[-window_size:]

    return np.expand_dims(last_window, axis=0)

if __name__ == '__main__':
    # Defines data paths
    market_data_path = Path("market_data")
    derived_data_path = Path("derived_data")
    models_dir = Path("models")
    architectures_dir = Path("architectures")
    logs_dir = Path("logs")

    # Reads all market data files
    market_data_files = sorted(market_data_path.glob("*.csv"))

    for asset in market_data_files:
        symbol = Path(asset).stem.upper()

        # Loads the selected models
        top_models_info = load_top_models(symbol, models_dir, architectures_dir, logs_dir)

        # Loads the test set features
        X_test = pd.read_csv(Path(f"{derived_data_path}/{symbol}/X_test.csv"))
        X_test = X_test.set_index('date')

        # Loads the test set targets
        y_test = pd.read_csv(Path(f"{derived_data_path}/{symbol}/y_test.csv"))
        y_test = y_test.set_index('date')

        symbol_results = []

        for model_info in top_models_info:
            quantile = model_info["quantile"]
            model_id = model_info["model_id"]
            target = model_info["target"]
            model = model_info["model"]
            window_size = model_info["window_size"]
            ranking_row = model_info["ranking_row"]

            if target not in y_test.columns:
                print(f"Target '{target}' not found in y_test. Skipping.")
                continue

            # Reshapes for time series
            try:
                X_test_seq, y_test_seq = reshape_data(X_test, y_test[target], int(window_size))

            except Exception as e:
                print(f"Error reshaping data for model {model_id}: {e}")
                continue

            # Predicts on full test set
            y_pred = model.predict(X_test_seq, verbose=0)
            y_true = pd.DataFrame(y_test_seq, columns=[target], index=y_test.index[int(window_size):])
            y_pred = pd.DataFrame(y_pred, columns=[target], index=y_true.index)

            # Calculates quantiles for thresholding
            lq_str, uq_str = quantile.split("_")
            lq, uq = float(f"0.{lq_str}"), float(f"0.{uq_str}")
            s_thresh = np.quantile(y_pred, lq)
            l_thresh = np.quantile(y_pred, uq)

            # Predicts last window
            X_last = get_last_window(X_test, int(window_size))
            y_last_pred = model.predict(X_last, verbose=0)[0][0]

            # Makes decision
            if y_last_pred < s_thresh:
                decision = "Short"

            elif y_last_pred > l_thresh:
                decision = "Long"

            else:
                decision = "Neutral"

            # Builds specific model result row
            row = {
                "symbol": symbol,
                "quantile": quantile,
                "model_id": model_id,
                "target": target,
                "predicted_value": y_last_pred,
                "short_threshold": s_thresh,
                "long_threshold": l_thresh,
                "decision": decision,
            }   

            # Attaches validation metrics
            val_metrics = [
                "val_sharpe_ewma", "val_sharpe_std", "val_drawdown_ewma",
                "val_hit_ratio_ewma", "val_pf_ewma", "val_trades_total",
                "test_sharpe", "test_drawdown", "test_hit_ratio",
                "test_pf", "test_trades_total", "test_perm_p_value"
            ]

            for metric in val_metrics:
                if metric in ranking_row:
                    row[metric] = ranking_row[metric]

            symbol_results.append(row)
        
        # Saves results
        results_dir = Path(f"{logs_dir}/{symbol}/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        results_path = results_dir / f"predictions_{today}.csv"
        pd.DataFrame(symbol_results).to_csv(results_path, index=False)

        


        
