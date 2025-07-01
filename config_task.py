import pandas as pd
import qlib
from qlib.data import D
from pathlib import Path
from qlib.utils import init_instance_by_config
from colors import print_green, print_yellow, warn_with_color, Colors
def get_date(day):
    # Calculate date ranges for training and validation
    start_date = pd.Timestamp("2019-01-01")  # Match data collector start date
    today = pd.Timestamp.now().normalize()  # Normalize to remove time component

    # --format='ISO8601'
    # Check if we have sufficient data range
    min_days = 100  # Minimum required days
    total_days = (today - start_date).days
    if total_days < min_days:
        print_yellow(f"Warning: Only {total_days} days of data available, adjusting date range...")
        start_date = today - pd.Timedelta(days=min_days)

    # Use 60% for training, 20% for validation, and 20% for testing
    train_ratio = 0.6
    train_days = int(total_days * train_ratio)
    train_end_date = start_date + pd.Timedelta(days=train_days)

    # Ensure we don't go into the future
    if train_end_date >= today:
        train_end_date = today - pd.Timedelta(days=day)

    # Ensure dates are properly formatted
    start_date_str = start_date.strftime("%Y-%m-%d")
    train_end_date_str = train_end_date.strftime("%Y-%m-%d")
    print_yellow(f"day: {day}")
    today_str = (today - pd.Timedelta(days=day)).strftime("%Y-%m-%d")
    valid_ratio = 0.2
    valid_days = int(total_days * valid_ratio)
    valid_end_date = train_end_date + pd.Timedelta(days=valid_days)
    valid_end_date_str = valid_end_date.strftime("%Y-%m-%d")
    print_green(f"Data range: {start_date_str} to {today_str}")
    print_green(f"Training: {start_date_str} to {train_end_date_str}")
    print_green(f"Validation: {train_end_date_str} to {valid_end_date_str}")
    print_green(f"Testing: {valid_end_date_str} to {today_str}")

    ###################################
    # train model
    ###################################

    return start_date_str, train_end_date_str, valid_end_date_str, today_str, today



def get_task(
    market=None,  # Market list to use, if None will try to load from existing data
    model="LGBModel",
    data_handler_config=None,  # Data handler configuration, if None will use default
    start_date_str="2021-01-01",
    train_end_date_str="2024-12-31",
    valid_end_date_str="2025-06-16",
    today_str="2025-06-18",  # Default to a future date for testing purposes
    today=None,  # If None, will use current date
):
    if market is None:
        raise ValueError("Market list cannot be None, please provide a valid market list")

    """
    Returns a task configuration for Qlib with specified date ranges.

    Parameters:
    - start_date_str: Start date for training data.
    - train_end_date_str: End date for training data.
    - today_str: Date for validation and test data.

    Returns:
    - A dictionary representing the task configuration.
    """
    if model == "EnsembleModel":
        task = {
            "model": {
                "class": "EnsembleModel",
                "module_path": "qlib.contrib.model.ensemble",
                "kwargs": {
                    # 多模型清單
                    "models": [
                        {
                            "class": "LGBModel",
                            "module_path": "qlib.contrib.model.gbdt",
                            "kwargs": {
                                "objective": "regression",
                                "metric": "l2",
                                "boosting_type": "gbdt",
                                "num_leaves": 64,
                                "learning_rate": 0.01,
                                "feature_fraction": 0.8,
                                "bagging_fraction": 0.8,
                                "bagging_freq": 5,
                                "min_child_samples": 20,
                                "lambda_l1": 0.1,
                                "lambda_l2": 0.1,
                                "min_split_gain": 0.01,
                                "max_depth": 8,
                                "n_estimators": 2**10,
                                "early_stopping_rounds": 2**8,
                                "random_state": 42,
                                "n_jobs": -1,
                                "verbosity": -1,
                            },
                        },
                        {
                            "class": "XGBModel",
                            "module_path": "qlib.contrib.model.gbdt",
                            "kwargs": {
                                "objective": "reg:linear",
                                "eval_metric": "rmse",
                                "max_depth": 8,
                                "learning_rate": 0.01,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "n_estimators": 2**10,
                                "early_stopping_rounds": 2**8,
                                "random_state": 42,
                                "n_jobs": -1,
                                "verbosity": 0,
                            },
                        },
                    ],
                    # 合併策略：平均 (也可改成 "median" 或用 weights 指定加權)
                    "method": "mean",
                    # "weights": [0.5, 0.5],
                },
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha360",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "train": (start_date_str, train_end_date_str),
                        "valid": (train_end_date_str, today_str),
                        "test": (
                            today_str,
                            (today + pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                        ),
                    },
                },
            },
        }
    elif model == "LGBModel":
        task = {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "objective": "regression",
                    "metric": "l2",
                    "boosting_type": "gbdt",
                    "num_leaves": 64,
                    "learning_rate": 0.01,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "min_child_samples": 20,
                    "lambda_l1": 0.1,
                    "lambda_l2": 0.1,
                    "min_split_gain": 0.01,
                    "max_depth": 8,
                    "n_estimators": 2**16,
                    "early_stopping_rounds": 2**14,
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbosity": -1,
                },
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha360",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "train": (start_date_str, train_end_date_str),
                        "valid": (train_end_date_str, today_str),
                        "test": (today_str, (today + pd.Timedelta(days=90)).strftime("%Y-%m-%d")),
                    },
                },
            },
        }
    elif model == 'TransformerModel_ts':

        transformer_dataset_cfg = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": (start_date_str, train_end_date_str),
                    "valid": (train_end_date_str, valid_end_date_str),
                    "test": (
                        valid_end_date_str,
                        today_str,
                    ),
                },
            },
        }


        """
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 8192,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        """
        
        task = {
            "model": {
                "class": "TransformerModel",
                "module_path": "qlib.contrib.model.pytorch_transformer_ts",
                "kwargs": {
                    # 特徵維度，通常是 handler 輸出特徵的數量
                    "d_feat": get_dfeat(transformer_dataset_cfg),   # now matches total_cols−1
                    "d_model": 1024,
                    # Attention heads 數量
                    "nhead": 8,
                    # Transformer 層數
                    "num_layers": 16,
                    # Feed-forward 隱藏層維度
                    "dim_feedforward": 4096,
                    # dropout 機率
                    "dropout": 0.1,
                    "use_amp": True,  # 是否使用自動混合精度
                    # 激活函數（可選 'relu'、'gelu'…）
                    "activation": "relu",
                    # 訓練相關超參數
                    "optimizer": "AdamW",
                    "learning_rate": 1,
                    "batch_size": 128,
                    "n_epochs": 3,
                    "loss": "mse_ic",
                    "alpha": 0.5,  # 用於 mse_ic 的正則化項
                    "beta" : 0.5,  # 用於 mse_ic 的正則化項
                    "metric": "loss",
                    # 如果有 GPU 可指定 "cuda"
                    "device": "cuda",
                    "seed": 42,
                    # 早停輪數
                    "early_stop": 64,
                },
            },
            "dataset": transformer_dataset_cfg,
        }
    elif model == "TransformerModel":
        """
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
"""
        transformer_dataset_cfg = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": (start_date_str, train_end_date_str),
                    "valid": (train_end_date_str, valid_end_date_str),
                    "test": (
                        valid_end_date_str,
                        today_str,
                    ),
                },
            },
        }

        task = {
            "model": {
                "class": "TransformerModel",
                "module_path": "new_transformer",
                "kwargs": {
                    # 特徵維度，通常是 handler 輸出特徵的數量
                    "d_feat": get_dfeat(transformer_dataset_cfg),   # now matches total_cols−1
                    "d_model": 256,
                    # Attention heads 數量
                    "nhead": 64,
                    # Transformer 層數
                    "num_layers": 256,
                    # Feed-forward 隱藏層維度
                    "dim_feedforward": 256,
                    # dropout 機率
                    "dropout": 0.001,
                    "use_amp": False,  # 是否使用自動混合精度
                    # 激活函數（可選 'relu'、'gelu'…）
                    "activation": "relu",
                    # 訓練相關超參數
                    "optimizer": "AdamW",
                    "learning_rate": 1e-7,
                    "batch_size": 128,
                    "n_epochs": 128,
                    "loss": "mse_ic2",
                    "alpha": 0.5,  # 用於 mse_ic 的正則化項
                    "beta" : 0.5,  # 用於 mse_ic 的正則化項
                    "metric": "loss",
                    # 如果有 GPU 可指定 "cuda"
                    "device": "cuda",
                    "seed": 19523,
                    # 早停輪數
                    "early_stop": 16,
                },
            },
            "dataset": transformer_dataset_cfg,
        }

    return task

def get_dfeat(transformer_dataset_cfg):
  # instantiate dataset to infer flat train‐window width
  ds_inst = init_instance_by_config(transformer_dataset_cfg)
  train_df = ds_inst.prepare("train")
  # train_df.columns = [ F₁(t₁),…,F_d(t₁),L(t₁),…,F₁(tₙ),…,F_d(tₙ),L(tₙ) ]
  # so number of features per step = (total_cols ÷ window_steps) − 1 (minus the label)
  total_cols = train_df.shape[1]
  # if you know your sequence length is 1, simply subtract label:
  actual_d_feat = total_cols - 1
  return actual_d_feat

def get_data_handler_config(
        market=None,  # Market list to use, if None will try to load from existing data
        start_date_str="2021-01-01",
        train_end_date_str="2024-12-31",
        valid_end_date_str="2025-06-16",
        today_str="2025-06-18",  # Default to a future date for testing purposes
):
    try:
       # Keep the original broader date range - don't narrow it down for fallback

        # If verification succeeds, use the current configuration
        data_handler_config = {
            "start_time": start_date_str,
            "end_time": today_str,
            "fit_start_time": train_end_date_str,
            "fit_end_time": valid_end_date_str,
            # "instruments": market,  # 確保`market`包含有效的股票代碼
            "instruments": market,  # 確保`market`包含有效的股票代碼
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"clip_outlier": True, "fit_start_time": start_date_str}
                },
                {
                    "class": "Fillna",
                    "kwargs": {"fill_value": 0}
                }
            ],
            "learn_processors": [
                {
                    "class": "DropnaLabel"
                },
                {
                    "class": "CSRankNorm",
                }
            ],
        }

    except Exception as e:
        warn_with_color(f"Verification failed: {e}")
        # Create a minimal fallback but keep reasonable date range
        print_yellow("Creating minimal fallback configuration...")
        market = ["AAPL"]  # Single stock
        # Keep the original broader date range even for fallback
        data_handler_config = {
            "start_time": start_date_str,
            "end_time": train_end_date_str,
            "fit_start_time": start_date_str,
            "fit_end_time": train_end_date_str,
            "instruments": market,  # 確保`market`包含有效的股票代碼
            # "instruments": 'instruments',  # 確保`market`包含有效的股票代碼
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"clip_outlier": True, "fit_start_time": start_date_str}
                },
                {
                    "class": "Fillna",
                    "kwargs": {"fill_value": 0}
                }
            ],
            "learn_processors": [
                {
                    "class": "DropnaLabel"
                },
                {
                    "class": "CSRankNorm",
                }
            ],
        }
    return data_handler_config

def test_data(market=None,
         start_date_str="2021-01-01",
         today_str="2025-06-16",  # Default to a future date for testing purposes
         ):
            # Verify data one more time with the handler configuration
    print_green("Verifying data with handler configuration...")
    verification_data = D.features(
        market,
        ["$close", "$open", "$high", "$low", "$volume"],
        start_time=start_date_str,
        end_time=today_str,
        freq="day"
    )
    print_green(f"Keeping original broader date range: {start_date_str} to {today_str}")
    print_green(f"Verification data shape: {verification_data.shape}")
    if verification_data.empty:
        warn_with_color("Verification failed: No data available")
        raise ValueError("Verification failed: No data available")

    # Check data quality
    non_null_rows = verification_data.dropna().shape[0]
    print_green(f"Non-null rows: {non_null_rows} out of {verification_data.shape[0]}")
