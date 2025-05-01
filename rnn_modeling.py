import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------------------
def reshape_data(X, y, window_size):
    """
    Reshapes the datasets into input sequences and corresponding targets required for the
    model.
    """
    # Converts to numpy arrays
    if isinstance(X, (pd.DataFrame, pd.Series)): X = X.values
    if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values

    X_seq, y_seq = [], []

    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i + window_size])

    return np.array(X_seq), np.array(y_seq)

def get_model(window_size: int, input_shape: int) -> Sequential:
    """
    Given the input shape (number of features), this function initializes a regression model 
    with the below architecture:
    """
    # Initialize the model
    model = Sequential()

    # Adds the lstm layers
    model.add(LSTM(window_size, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(window_size // 2, return_sequences=False))

    # Adds the output layer
    model.add(Dense(1, activation='linear'))

    return model

def calculate_performance_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, lower_q: float, upper_q: float) -> dict:
    """
    Given the true, predicted values of test set, the lower and upper quantiles that define position
    direction, this function calculates performance metrics for a trading strategy such as: sharpe ratio, 
    probabilistic sharpe ratio (sharpe ratio statistical significance), precision, recall, F1 score, 
    confusion matrix, cumulative return, volatility, max drawdown, hit ratio, and profit factor.
    """
    # Calculates the quantiles of the strategy predicted values
    s_thresh = np.quantile(y_pred, lower_q)
    l_thresh = np.quantile(y_pred, upper_q)

    # Calculates the direction of the positions based on the quantile thresholds
    s_mask = y_pred <= s_thresh
    l_mask = y_pred >= l_thresh

    # Calculates the returns of the strategy based on the true values
    s_returns = -(np.exp(y_true[s_mask].dropna()) - 1).values
    l_returns = (np.exp(y_true[l_mask].dropna()) - 1).values
    all_returns = np.concatenate([s_returns, l_returns])

    # Calculates sharpe and probabilistic sharpe ratio
    sharpe = np.mean(all_returns) / np.std(all_returns, ddof=1)
    std_error = np.sqrt((1 + 0.5 * sharpe**2) / len(all_returns))
    psr = (sharpe / std_error)
    psr = norm.cdf(psr)

    # Calculates cumulative return
    cumulative_return = np.prod(1 + all_returns) - 1

    # Calculates volatility
    volatility = np.std(all_returns, ddof=1)

    # Calculates max drawdown
    equity_curve = np.cumprod(1 + all_returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown)

    # Calculates hit ratio
    hit_ratio = np.mean(all_returns > 0)

    # Calculates profit factor
    gross_profit = np.sum(all_returns[all_returns > 0])
    gross_loss = np.abs(np.sum(all_returns[all_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calculates classification metrics: positive direction = return > 0
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    conf_matrix = confusion_matrix(y_true_bin, y_pred_bin)

    return {
        "sharpe": sharpe,
        "psr": psr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "hit_ratio": hit_ratio,
        "profit_factor": profit_factor,
        "n_long": len(l_returns),
        "n_short": len(s_returns)
    }

def find_best_quantile_performance_metrics(y_true, y_pred, quantile_pairs):
    """
    Given the true, predicted values of test set and the quantile pairs that define position direction,
    this function calculates performance metrics for a trading strategy and returns the best pair of quantiles
    based on the highest sharpe ratio.
    """
    best_metrics = None
    best_pair = None

    for lq, uq in quantile_pairs:
        metrics = calculate_performance_metrics(y_true, y_pred, lq, uq)

        if (best_metrics is None) or (metrics["sharpe"] > best_metrics["sharpe"]):
            best_metrics = metrics
            best_pair = (lq, uq)

    best_metrics["quantiles"] = best_pair
    return best_metrics

def log_validation(trial_id: int, target: str, step: int, performance_metrics: dict) -> None:
    """
    Logs extended performance metrics for the validation set of a trading model to a CSV file and 
    saves model/config paths.
    """
    
    log_entry = {
        "step": step,
        "quantiles": performance_metrics["quantiles"],
        "sharpe": performance_metrics["sharpe"],
        "psr": performance_metrics["psr"],
        "precision": performance_metrics["precision"],
        "recall": performance_metrics["recall"],
        "f1_score": performance_metrics["f1"],
        "cumulative_return": performance_metrics["cumulative_return"],
        "volatility": performance_metrics["volatility"],
        "max_drawdown": performance_metrics["max_drawdown"],
        "hit_ratio": performance_metrics["hit_ratio"],
        "profit_factor": performance_metrics["profit_factor"],
        "n_long_trades": performance_metrics["n_long"],
        "n_short_trades": performance_metrics["n_short"]
    }

    log_file = f"logs/validation_performance/{trial_id}_{target}.csv"
    df_entry = pd.DataFrame([log_entry])

    if os.path.exists(log_file):
        existing_df = pd.read_csv(log_file)
        df_entry = pd.concat([existing_df, df_entry], ignore_index=True)

    df_entry.to_csv(log_file, index=False)

def log_evaluation(trial_id: int, target: str, performance_metrics: dict, config: dict) -> None:
    """
    Logs extended performance metrics for the test set of a trading model to a CSV file and 
    saves model/config paths.
    """
    log_entry = {
        "trial_id": trial_id,
        "target": target,
        "quantiles": performance_metrics["quantiles"],
        "sharpe": performance_metrics["sharpe"],
        "psr": performance_metrics["psr"],
        "precision": performance_metrics["precision"],
        "recall": performance_metrics["recall"],
        "f1_score": performance_metrics["f1"],
        "cumulative_return": performance_metrics["cumulative_return"],
        "volatility": performance_metrics["volatility"],
        "max_drawdown": performance_metrics["max_drawdown"],
        "hit_ratio": performance_metrics["hit_ratio"],
        "profit_factor": performance_metrics["profit_factor"],
        "n_long_trades": performance_metrics["n_long"],
        "n_short_trades": performance_metrics["n_short"],
        "conf_matrix": json.dumps(performance_metrics["confusion_matrix"]),
        "model_path": f"models/model_{trial_id}_{target}.h5",
        "config_path": f"logs/configuration_{trial_id}.json"
    }

    df = pd.DataFrame([log_entry])
    log_file = "logs/evaluation_performance.csv"

    if os.path.exists(log_file):
        df_existing = pd.read_csv(log_file)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(log_file, index=False)

    with open(log_entry["config_path"], 'w') as f:
        json.dump(config, f, indent=4)

# -----------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------
features_path = "data/X_pca.csv"
target_path = "data/y.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("logs/validation_performance", exist_ok=True)
os.makedirs("logs", exist_ok=True)

X = pd.read_csv(features_path, index_col='date')
y = pd.read_csv(target_path, index_col='date')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# X_train.to_csv('data/X_train.csv', index=True, header=True)
# X_test.to_csv('data/X_test.csv', index=True, header=True)
# y_train.to_csv('data/y_train.csv', index=True, header=True)
# y_test.to_csv('data/y_test.csv', index=True, header=True)

n_trials = 10
initial_lr = 0.001

for trial_id in range(n_trials):
    print(f"\nTrial {trial_id} — Starting new trial")

    # Randomized model configuration
    # dropout = np.round(np.random.uniform(0.0, 0.25), 2)
    window_size = np.random.randint(30, 45)
    delta = np.round(np.random.uniform(1.5, 3.5), 3)
    loss_fn = Huber(delta=delta)
    config = {"window size": float(window_size), "loss": f"Huber({delta})", "lr": float(initial_lr)}
    print(f"Model configuration: {config}")

    for target in y_train.columns:
        print(f"\nTraining model for target: {target}")

        # Reshapes data
        X_train_seq, y_train_seq = reshape_data(X_train, y_train[target], window_size)
        X_test_seq, y_test_seq = reshape_data(X_test, y_test[target], window_size)

        # Initializes model
        model = get_model(window_size, input_shape=(window_size, X.shape[1]))
        # model.summary()

        # Setsup optimizer and compiles model
        model.compile(optimizer=Adam(learning_rate=initial_lr), loss=loss_fn)

        # Online training (walk-forward validation style)
        train_loss, preds, truths = [], [], []
        training_mae, validation_mae = [], []

        for i in range(len(X_train_seq) - 1):
            history = model.train_on_batch(X_train_seq[i:i+1], y_train_seq[i:i+1])  
            train_loss.append(history)                                                       
            truths.append(y_train_seq[i+1])
            pred = model.predict(X_train_seq[i+1:i+2], verbose=0)[0][0]                    
            preds.append(pred)                                                                     

            # Analyzes training and validation error metrics
            if (i+1) % 200 == 0 or i == len(X_train) - 2:
                current_train_mae = np.mean(train_loss[-200:])
                current_val_mae = mean_absolute_error(truths[-200:], preds[-200:])
                training_mae.append(current_train_mae)
                validation_mae.append(current_val_mae)
                print(f"Samples {i+1}/{len(X_train)-1} - Training MAE: {current_train_mae:.4f} - Validation MAE: {current_val_mae:.4f}", end='\n')

            # Analyzes validation performance metrics
            if (i+1) % 365 == 0 and len(truths) >= 365:
                y_truths = pd.DataFrame(truths[-365:], columns=[target])
                y_preds = pd.DataFrame(preds[-365:], columns=[target])

                quantile_pairs = [(0.01, 0.99), (0.02, 0.98), (0.05, 0.95)]
                validation_performance_metrics = find_best_quantile_performance_metrics(y_truths, y_preds, quantile_pairs)

                log_validation(trial_id, target, i+1, validation_performance_metrics)

        # Evaluates generalization on test set
        y_pred = model.predict(X_test_seq, verbose=0)
        y_pred = pd.DataFrame(y_pred, columns=[target])
        y_pred.reset_index(drop=True, inplace=True)

        y_true = pd.DataFrame(y_test_seq, columns=[target])
        y_true.reset_index(drop=True, inplace=True)

        quantile_pairs = [(0.01, 0.99), (0.02, 0.98), (0.05, 0.95)]
        evalutation_performance_metrics = find_best_quantile_performance_metrics(y_true, y_pred, quantile_pairs)

        # Saves model and predictions
        model_path = f"models/model_{trial_id}_{target}.keras"
        model.save(model_path)
        log_evaluation(trial_id, target, evalutation_performance_metrics, config)