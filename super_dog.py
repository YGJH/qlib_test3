import os
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from config_task import get_date, get_data_handler_config, get_task
from transformer import TransformerModel  # 确保 module_path="transformer" 能被找到
from colors import * 
from pathlib import Path
from symbo import symbols  # 确保 symbols.py 在同一目录下
import random


def get_market(market_num):
    try:
        # Read instruments directly from the file
        instruments_file = Path(".qlib/qlib_data/us_data/instruments/all.txt")
        if instruments_file.exists():
            with open(instruments_file, 'r') as f:
                lines = f.readlines()
            
            # Parse the instrument names (first column)
            available_instruments = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments and empty lines
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        symbol = parts[0].strip()
                        available_instruments.append(symbol)
            
        
        # Filter to common US stocks (remove indices that start with ^ and _)
        us_stocks = available_instruments

        if len(us_stocks) > 0:
            market = us_stocks
            if market_num is not None:
                print_green(f"Limiting market to {market_num} instruments")
                avaliable_sym = set(us_stocks)&set(symbols['US'])
                market = list(avaliable_sym)[:min(len(market), market_num)]  # Limit to specified number
            else:
                print_green("No market limit specified, using all available US stocks")
                avaliable_sym = set(us_stocks)&set(symbols['US'])
                market = list(avaliable_sym)
            print_yellow(f"Selected {len(market)} instruments from US stocks: {market[:min(10, len(market))]}...")
            return market
        else:
            warn_with_color("No valid US stocks found in instruments file.")
            return []

    except Exception as e:
        warn_with_color(f"Error reading instruments from file: {e}")
        print_yellow("Using hardcoded fallback configuration...")
        market = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def evaluate_rmse(model, dataset):
    """在 valid 上预测并计算 RMSE"""
    val_feat = dataset
    val_label = dataset.prepare("valid", col_set="label")
    pred = model.predict(val_feat)
    # 对齐
    idx = pred.index.intersection(val_label.index)
    y_pred, y_true = get_y_pred_and_y_true(idx, pred, val_label)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def get_y_pred_and_y_true(common_index , val_pred, val_label):
    if len(common_index) > 0:
        aligned_pred = val_pred.loc[common_index]
        aligned_label = val_label.loc[common_index]
        
        # Extract values - 正確處理 Series 和 DataFrame
        # 處理標籤 (DataFrame)
        if isinstance(aligned_label, pd.DataFrame):
            y_true = aligned_label.iloc[:, 0].values  # DataFrame 可以使用 iloc[:, 0]
        else:
            y_true = aligned_label.values  # Series 直接使用 .values
        
        # 處理預測 (Series)
        if isinstance(aligned_pred, pd.Series):
            y_pred = aligned_pred.values  # Series 直接使用 .values
        elif isinstance(aligned_pred, pd.DataFrame):
            y_pred = aligned_pred.iloc[:, 0].values  # DataFrame 使用 iloc[:, 0]
        else:
            y_pred = aligned_pred.values
    return y_pred, y_true

def evaluate_ic(model, dataset):
    y_label = dataset.prepare("valid", col_set="label")
    y_pred = model.predict(dataset)
    common_index = y_pred.index.intersection(y_label.index)
    y_pred, y_true = get_y_pred_and_y_true(common_index, y_pred, y_label)
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_true_rank = y_true_series.rank()
    y_pred_rank = y_pred_series.rank()
    rank_ic_corr = np.corrcoef(y_true_rank, y_pred_rank)
    rank_ic = rank_ic_corr[0, 1] if rank_ic_corr.shape == (2, 2) and not np.isnan(rank_ic_corr[0, 1]) else 0
    return rank_ic


def eval_model_performance(dataset, model):
    print_green("Evaluating model performance...")
    model_metrics = {}
    
    try:
        print_green("Calculating model performance metrics...")
        
        # Get validation data and labels using dataset object properly
        try:
            val_data = dataset
            val_label = dataset.prepare("valid", col_set="label")
            # print_green(f"Validation data shape: {val_data.shape}")
            print_green(f"Validation label shape: {val_label.shape}")
        except Exception as dataset_error:
            warn_with_color(f"Dataset prepare error: {dataset_error}")
            # Try alternative approach if prepare fails
            raise ValueError(f"Cannot prepare dataset: {dataset_error}")
        
        if not val_label.empty:
            print_green("Generating predictions for evaluation...")
            
            # Use the trained model to predict on validation data
            val_pred = model.predict(val_data)
            print_green(f"Evaluation predictions shape: {val_pred.shape}")
            if not val_pred.empty and len(val_pred) > 0:
                # Align predictions and labels by index
                common_index = val_pred.index.intersection(val_label.index)
                
                if len(common_index) > 0:
                    aligned_pred = val_pred.loc[common_index]
                    aligned_label = val_label.loc[common_index]
                    
                    # Extract values - 正確處理 Series 和 DataFrame
                    # 處理標籤 (DataFrame)
                    if isinstance(aligned_label, pd.DataFrame):
                        y_true = aligned_label.iloc[:, 0].values  # DataFrame 可以使用 iloc[:, 0]
                    else:
                        y_true = aligned_label.values  # Series 直接使用 .values
                    
                    # 處理預測 (Series)
                    if isinstance(aligned_pred, pd.Series):
                        y_pred = aligned_pred.values  # Series 直接使用 .values
                    elif isinstance(aligned_pred, pd.DataFrame):
                        y_pred = aligned_pred.iloc[:, 0].values  # DataFrame 使用 iloc[:, 0]
                    else:
                        y_pred = aligned_pred.values
                    
                    print_green(f"Aligned data points: {len(y_true)}")
                    
                    # Calculate metrics using numpy
                    mse = np.mean((y_true - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    
                    # Calculate R² score manually
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calculate Information Coefficient (IC)
                    ic_corr = np.corrcoef(y_true, y_pred)
                    ic = ic_corr[0, 1] if ic_corr.shape == (2, 2) and not np.isnan(ic_corr[0, 1]) else 0
                    
                    # Calculate rank IC
                    y_true_series = pd.Series(y_true)
                    y_pred_series = pd.Series(y_pred)
                    y_true_rank = y_true_series.rank()
                    y_pred_rank = y_pred_series.rank()
                    rank_ic_corr = np.corrcoef(y_true_rank, y_pred_rank)
                    rank_ic = rank_ic_corr[0, 1] if rank_ic_corr.shape == (2, 2) and not np.isnan(rank_ic_corr[0, 1]) else 0
                    
                    # Calculate accuracy metrics
                    direction_accuracy = np.mean((y_true > 0) == (y_pred > 0))
                    
                    model_metrics = {
                        "validation_samples": len(y_true),
                        "mse": round(float(mse), 6),
                        "rmse": round(float(rmse), 6),
                        "r2_score": round(float(r2), 4),
                        "information_coefficient": round(float(ic), 4),
                        "rank_ic": round(float(rank_ic), 4),
                        "direction_accuracy": round(float(direction_accuracy), 4),
                        "prediction_mean": round(float(np.mean(y_pred)), 6),
                        "prediction_std": round(float(np.std(y_pred)), 6),
                        "label_mean": round(float(np.mean(y_true)), 6),
                        "label_std": round(float(np.std(y_true)), 6)
                    }
                    
                    print_green("Model Performance Metrics:")
                    print_green(f"  - Validation Samples: {model_metrics['validation_samples']}")
                    print_green(f"  - R² Score: {model_metrics['r2_score']:.4f}")
                    print_green(f"  - RMSE: {model_metrics['rmse']:.6f}")
                    print_green(f"  - Information Coefficient: {model_metrics['information_coefficient']:.4f}")
                    print_green(f"  - Rank IC: {model_metrics['rank_ic']:.4f}")
                    print_green(f"  - Direction Accuracy: {model_metrics['direction_accuracy']:.4f}")
                else:
                    warn_with_color("No common index between predictions and labels")
                    model_metrics = {"error": "No common index", "status": "alignment_failed"}
            else:
                warn_with_color("Empty validation predictions")
                model_metrics = {"error": "Empty predictions", "status": "prediction_failed"}
        else:
            warn_with_color("Empty validation data or labels")
            model_metrics = {"error": "Empty validation data", "status": "data_unavailable"}
            
        return model_metrics
    except Exception as metric_error:
        warn_with_color(f"Error calculating model metrics: {metric_error}")
        
        # Skip fallback metric calculation to avoid further DataFrame/dataset confusion
        print_yellow("Skipping fallback metric calculation due to data structure issues...")
        model_metrics = {
            "error": f"Metric calculation failed: {str(metric_error)}",
        "status": "metrics_unavailable"
    }
    return model_metrics
def get_lr():
    now = 10
    while now > 1e-10:
        now *= 0.5
        yield now


def get_seeds():
    now = 616
    while now % 13 != 0:
        now = (now * 5654233  + 1233131) % 231123
        yield now
def get_market_num():
    for i in range(10 , 500 , 20):
        yield i

def main():
    # 1. 初始化 qlib
    provider_uri = ".qlib/qlib_data/us_data"
    qlib.init(provider_uri=provider_uri, region=REG_US)
    market_num = 100
    # 2. 统一拿好日期 & 市场（或从 main.py 复制）
    days = 8
    start_str, train_end_str, today_str, today = get_date(days)
    # 你这里直接 hardcode 或复用 main.py 里的 market 逻辑
    print_green("Checking available instruments...")

    # 3. 定义要遍历的超参列表（这里只示例不同 seed）
    # seeds = [0, 42, 100, 2025]
    best_rank_ic = float("-inf")
    best_cfg = None
    best_model = None

    for market_num in get_market_num():
        # 3.1 获取市场数据
        market = get_market(market_num)
        if not market:
            raise ValueError("No valid market instruments found. Please check your configuration or data.")
        
        data_handler_cfg = get_data_handler_config(
            market=market,
            start_date_str=start_str,
            train_end_date_str=train_end_str,
            today_str=today_str,
        )

        # print(f"\n>>> Training with lr = {lr}")
        print_green(f"\n>>> Training with market_num = {market_num}")
        # 在 get_task 中总是采用 TransformerModel
        task = get_task(
            market=market,
            model="TransformerModel",
            data_handler_config=data_handler_cfg,
            start_date_str=start_str,
            train_end_date_str=train_end_str,
            today_str=today_str,
            today=today,
        )
        import json
        with open("instruments.json", "w", encoding="utf-8") as fp:
            json.dump(market, fp, ensure_ascii=False, indent=2)

        # 3.1 构造模型 & 数据集
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        # 3.2 训练
        import mlflow
        mlflow.log_artifact("instruments.json", artifact_path="metadata")

        model.fit(dataset)
        # 3.3 验证
        rank_ic = evaluate_ic(model, dataset)
        print(f"→ market_num {market_num} validation Rank IC = {rank_ic:.6f}")
        if rank_ic > best_rank_ic:
            best_rank_ic = rank_ic
            best_cfg = market_num
            best_model = model

    # 4. 保存最优模型
    save_path = "best_transformer.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\n🏆 最佳 market_num = {best_cfg}, Rank IC = {best_rank_ic:.6f}")
    print(f"最佳模型已保存到 {save_path}")


    # 5. 评估模型性能
    dataset = init_instance_by_config(task["dataset"])
    eval_model_performance(dataset, best_model)




if __name__ == "__main__":
    main()