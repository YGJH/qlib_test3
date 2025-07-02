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
from main import get_market


def main():
    # 1. 初始化 Qlib（请确保已下载美股数据到 .qlib/qlib_data/us_data）
    qlib.init(provider_uri=".qlib/qlib_data/us_data", region=REG_US)
    days = 10
    start_date_str, train_end_date_str, valid_end_date_str, today_str, today = get_date(days)

    # 2. 加载已训练好模型
    with open("best_transformer.pkl", "rb") as f:
        model = pickle.load(f)

    # 3. 构造数据集配置（直接写在脚本里，不依赖 config_task）
    market , benchmark = get_market(market_num=50)

    start_date_str, train_end_date_str, valid_end_date_str, today_str, today = get_date(days)

    data_handler_config = get_data_handler_config(market=market,
                                                start_date_str=start_date_str,
                                                train_end_date_str=train_end_date_str,
                                                valid_end_date_str=valid_end_date_str,
                                                today_str=today_str)
    test_data(
        market=market,
        start_date_str=start_date_str,
        today_str=today_str,
    )
    task = get_task(
        start_date_str=start_date_str,
        train_end_date_str=train_end_date_str,
        valid_end_date_str=valid_end_date_str,
        today_str=today_str,
        data_handler_config=data_handler_config,
        today=today,
        market=market,
        model="TransformerModel"
    )
    dataset = init_instance_by_config(task["dataset"])

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
            "codes":        market,
            "open_cost":    0.0005,
            "close_cost":   0.0015,
            "min_cost":     5,
        },
    }

    # 5. 运行回测并保存信号/组合表现
    with R.start(experiment_name="trade_strategy"):
        recorder = R.get_recorder()

        # 1) 先做 SignalRecord
        sr = SignalRecord(
            model=model,      # 已训练好的模型
            dataset=dataset,  # Qlib DatasetH 实例
            recorder=recorder
        )
        sr.generate()  # 保存 pred.pkl, label.pkl 等

        # 2) 再做 Portfolio Analysis
        #    需要传入 recorder 和完整的 config dict
        pa_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {
                    "signal": sr.load("pred.pkl"),  # 用刚存的预测
                    "topk": 50,
                    "n_drop": 5,
                },
            },
            "backtest": backtest_config,
        }
        pr = PortAnaRecord(
            recorder=recorder,
            config=pa_config
        )
        pr.generate()  # 保存 report_normal_...、positions_normal_...、port_analysis_... 等

        # 3) 最后落盘
        recorder.save_objects()


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