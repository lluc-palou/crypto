import os
import numpy as np
import pandas as pd
from glob import glob

def aggregate_validation_logs(validation_log_dir):
    """
    Aggregates validation logs from  the specified directory.
    """
    summaries = []
    
    for file_path in glob(os.path.join(validation_log_dir, "*.csv")):
        filename = os.path.basename(file_path)
        model_id = filename.split("_")[0]
        target = "_".join(filename.split("_")[1:]).replace(".csv", "")
        df = pd.read_csv(file_path)

        summaries.append({
            "model_id": model_id,
            "target": target,
            "val_sharpe_mean": df["sharpe"].mean(),
            "val_sharpe_std": df["sharpe"].std(),
            "val_drawdown_mean": df["max_drawdown"].mean(),
            "val_hit_ratio_mean": df["hit_ratio"].mean(),
            "val_pf_mean": df["profit_factor"].mean(),
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
                "model_id": row["model_path"].split("/")[-1].split("_")[0],
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

    test_df = test_df[test_df["test_perm_p_value"] <= 0.05]
    merged = pd.merge(val_df, test_df, on=["model_id", "target"], how="inner")
    
    merged["score"] = (
        merged["val_sharpe_mean"] * 2.0 +
        merged["test_sharpe"] * 2.5 +
        np.log1p(merged["val_pf_mean"]) * 1.5 +
        np.log1p(merged["test_pf"]) * 2.0 +
        merged["val_hit_ratio_mean"] * 1.0 +
        merged["test_hit_ratio"] * 1.5 -
        merged["val_drawdown_mean"] * 0.5 -
        merged["test_drawdown"] * 0.5 -
        merged["val_sharpe_std"] * 1.0 +
        np.log1p(merged["val_trades_total"]) * 0.5 +
        np.log1p(merged["test_trades_total"]) * 0.5
    )
    
    ranked = merged.sort_values(by="score", ascending=False).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    ranked.to_csv(os.path.join(output_dir, f"ranking_{quantile_name}.csv"), index=False)
    print(f"Saved ranking for {quantile_name} in {output_dir}/ranking_{quantile_name}.csv")

if __name__ == "__main__":
    validation_base = "logs/performance/validation"
    test_base = "logs/performance/test"
    output_dir = "logs/model_ranking"

    quantile_folders = sorted([
        folder for folder in os.listdir(validation_base)
        if os.path.isdir(os.path.join(validation_base, folder))
    ])

    for q_folder in quantile_folders:
        val_path = os.path.join(validation_base, q_folder)
        test_path = os.path.join(test_base, q_folder)

        if os.path.exists(test_path):
            rank_for_quantile_pair(q_folder, val_path, test_path, output_dir)