import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

def aggregate_validation_logs(validation_log_dir):
    """
    Aggregates validation logs from  the specified directory.
    """
    summaries = []

    # Decay parameter (controls the smoothing)
    span = 3
    
    for file_path in glob(os.path.join(validation_log_dir, "*.csv")):
        filename = os.path.basename(file_path)
        model_id = filename.split("_")[0]
        target = "_".join(filename.split("_")[1:]).replace(".csv", "")
        df = pd.read_csv(file_path)

        summaries.append({
            "model_id": model_id,
            "target": target,
            "val_sharpe_ewma": df["sharpe"].ewm(span=span).mean().iloc[-1],
            "val_sharpe_std": df["sharpe"].std(),
            "val_drawdown_ewma": df["max_drawdown"].ewm(span=span).mean().iloc[-1],
            "val_hit_ratio_ewma": df["hit_ratio"].ewm(span=span).mean().iloc[-1],
            "val_pf_ewma": df["profit_factor"].ewm(span=span).mean().iloc[-1],
            "val_trades_total": df["n_long_trades"].sum() + df["n_short_trades"].sum()
        })
        
    return pd.DataFrame(summaries)

def load_test_logs(test_log_dir):
    """
    Loads and aggregates test (generalization) logs from the specified path.
    """
    summaries = []

    for file_path in glob(os.path.join(test_log_dir, "*.csv")):
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            summaries.append({
                "model_id": str(row["trial_id"]),
                "target": row["target"],
                "test_sharpe": row["sharpe"],
                "test_drawdown": row["max_drawdown"],
                "test_hit_ratio": row["hit_ratio"],
                "test_pf": row["profit_factor"],
                "test_trades_total": row["n_long_trades"] + row["n_short_trades"],
                "test_perm_p_value": row["perm_p_value"]
            })
            
    return pd.DataFrame(summaries)

def rank_for_quantile_pair(quantile_name, val_path, test_path, output_dir):
    val_df = aggregate_validation_logs(val_path)
    test_df = load_test_logs(test_path)

    # Drops NaN values from validation and test dataframes
    val_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Keeps only statistically significant test results
    test_df = test_df[test_df["test_perm_p_value"] <= 0.05]

    # Filters for positive sharpe ratios
    val_df = val_df[val_df["val_sharpe_ewma"] > 0]
    test_df = test_df[test_df["test_sharpe"] > 0]

    merged = pd.merge(val_df, test_df, on=["model_id", "target"], how="inner")
    
    merged["score"] = (
        merged["val_sharpe_ewma"] * 2.0 +
        np.log1p(merged["val_pf_ewma"]) * 1.5 +
        merged["val_hit_ratio_ewma"] * 1.0 -
        merged["val_drawdown_ewma"] * 0.5 -
        merged["val_sharpe_std"] * 1.0 +
        np.log1p(merged["val_trades_total"]) * 0.5
    )
    
    ranked = merged.sort_values(by="score", ascending=False).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    ranked.to_csv(os.path.join(output_dir, f"ranking_{quantile_name}.csv"), index=False)
    print(f"Saved ranking for {quantile_name} in {output_dir}/ranking_{quantile_name}.csv")

if __name__ == "__main__":
    # Defines data paths
    market_data_path = Path("market_data")

    # Reads all market data files
    market_data_files = sorted(market_data_path.glob("*.csv"))

    for asset in market_data_files:
        symbol = Path(asset).stem.upper()
        validation_base = Path(f"logs/{symbol}/performance/validation")
        test_base = Path(f"logs/{symbol}/performance/test")
        output_dir = Path(f"logs/{symbol}/model_ranking")

        quantile_folders = sorted([
            folder for folder in os.listdir(validation_base)
            if os.path.isdir(os.path.join(validation_base, folder))
        ])

        for q_folder in quantile_folders:
            val_path = os.path.join(validation_base, q_folder)
            test_path = os.path.join(test_base, q_folder)

            if os.path.exists(test_path):
                rank_for_quantile_pair(q_folder, val_path, test_path, output_dir)