import numpy as np
import pandas as pd
import os
import json


def str_to_ndarray(str: str) -> np.ndarray:
    rows = str.split('\n')
    arr = np.ndarray((len(rows), len(rows[0].split())))
    
    for idx, row in enumerate(rows):
        arr[idx, :] = row.split()
    
    return arr


def read_config(config_filename: str, output_dir="output"):
    config = {}

    with open(f"config/{config_filename}.json", "r+") as f:
        config = json.load(f)
    config["experiment_name"] = config_filename

    os.makedirs(output_dir, exist_ok=True)
    
    return config

def populate_config(config: dict, X):
    """
    Populate config with data specific details
    """
    config["height"] = X[0][0].shape[0]
    config["width"] = X[0][0].shape[1]
    config["num_days"] = len(X[0])


def get_sharpe_ratio(daily_returns: list, factor=np.sqrt(252)) -> float:
    """ Given a list of daily returns, returns sharpe ratio

    Args:
        daily_returns (list): Daily returns
        factor (_type_, optional): Factor to annualize daily returns. There are
            252 trading days in a year. Defaults to np.sqrt(252).

    Returns:
        float: Sharpe ratio
    """
    mean_daily_returns = np.mean(daily_returns)
    std_daily_returns = np.std(daily_returns)

    return mean_daily_returns/std_daily_returns * factor

def get_anomalies(crypto_df: pd.DataFrame, 
                    columns=['Open', 'High', 'Low', 'Close'], window=14):
    df = crypto_df.copy()
    for column in columns:
        r = df[column].rolling(window)
        # TODO: This outlier calculation does not seem alright. What if its
        # smaller etc.?
        df[f"{column}_is_anomaly"] = df[column] > r.mean() + 3 * r.std()
    return df

def clean_anomalies(crypto_df: pd.DataFrame, 
                    columns=['Open', 'High', 'Low', 'Close'], window=14):
    df = get_anomalies(crypto_df, columns, window)
    for column in columns:
        df.loc[df[f"{column}_is_anomaly"], column] \
            = df[column].rolling(window).mean()[df[f"{column}_is_anomaly"]]
    return df
