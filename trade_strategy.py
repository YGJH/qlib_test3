import pickle
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from config_task import *


def main():
    # 1. 初始化 Qlib（请确保已下载美股数据到 .qlib/qlib_data/us_data）
    qlib.init(provider_uri=".qlib/qlib_data/us_data", region=REG_US)
    days = 10
    start_date_str, train_end_date_str, valid_end_date_str, today_str, today = get_date(days)

    # 2. 加载已训练好模型
    with open("best_transformer.pkl", "rb") as f:
        model = pickle.load(f)

    # 3. 构造数据集配置（直接写在脚本里，不依赖 config_task）
    instruments = get # 示例标的
    data_handler_config = {
        "start_time": "2019-01-01",
        "end_time": "2025-06-18",
        "fit_start_time": "2019-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": instruments,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"clip_outlier": True, "fit_start_time": "2019-01-01"}},
            {"class": "Fillna", "kwargs": {"fill_value": 0}}
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm"}
        ],
    }
    dataset_cfg = {
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
                "test":  (valid_end_date_str, today_str),
            },
        },
    }
    dataset = init_instance_by_config(dataset_cfg)

    # 4. 配置策略和执行器
    strategy = TopkDropoutStrategy(model=model, dataset=dataset, topk=50, n_drop=5)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    backtest_config = {
        "start_time": valid_end_date_str,
        "end_time":   today_str,
        "account":    1e8,
        "benchmark":  "SPY",
        "exchange_kwargs": {
            "freq":         "day",
            "codes":        instruments,
            "open_cost":    0.0005,
            "close_cost":   0.0015,
            "min_cost":     5,
        },
    }

    # 5. 运行回测并保存信号/组合表现
    with R.start(experiment_name="trade_strategy"):
        # 生成交易信号
        sr = SignalRecord(
            model=model,
            dataset=dataset,
            strategy=strategy,
            executor=executor,
            backtest=backtest_config
        )
        sr.generate()  # sr.data["trade"] 包含逐笔买卖信号
        # 生成投资组合评估
        pr = PortAnaRecord(record=sr)
        pr.generate()  # pr.data["portfolio"] 包含净值曲线、指标等
        R.save_objects()  # 存档模型、信号、回测结果

    # 6. 导出并查看买卖信号
    trades = sr.data["trade"].reset_index()
    trades.to_csv("trades.csv", index=False)
    print("=== 前 10 条交易信号 ===")
    print(trades.head(10))

    # 7. 导出并查看组合表现
    portfolio = pr.data["portfolio"]
    portfolio.to_csv("portfolio_performance.csv")
    print("组合净值走势和指标已保存到 portfolio_performance.csv")

if __name__ == "__main__":
    main()