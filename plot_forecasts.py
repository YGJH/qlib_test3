import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from qlib.data import D
from qlib.constant import REG_US
import qlib

def plot_stock_forecast(stock: str, future_json: dict, end_date: str):
    """
    1) 从 Qlib 拿到 stock 的历史收盘价（到预测前一天）
    2) 从 future_json 里提取 daily_forecast 累积收益率，重建未来价格
    3) 画历史+预测价格曲线
    """
    # 初始化 Qlib（仅第一次需要）
    qlib.init(provider_uri=".qlib/qlib_data/us_data", region=REG_US)

    # 1) 历史收盘价
    hist_df = D.features(
        instruments=[stock],
        fields=["$close"],
        start_time=None,
        end_time=end_date
    )
    # hist_df 的 index 是 (instrument, datetime)，value 在第一列
    hist = hist_df.iloc[:, 0]
    # 去掉第一层 instrument，保留 datetime 作为索引
    if isinstance(hist.index, pd.MultiIndex):
        hist.index = pd.to_datetime(hist.index.get_level_values(1))
    else:
        hist.index = pd.to_datetime(hist.index)

    last_price = hist.iloc[-1]

    # 2) 解析累积收益率
    daily = future_json["predictions"][stock]["daily_forecast"]
    df = pd.Series(
        {pd.to_datetime(d): info["cumulative_return"] for d, info in daily.items()}
    )
    # 重建未来价格
    forecast_price = last_price * (1 + df)

    # 3) 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist.values, label="历史收盘价")
    plt.plot(forecast_price.index, forecast_price.values, "--", label="预测价格")
    plt.title(f"{stock} 价格走势预测")
    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.legend()
    plt.grid(True)
    out_html = f"image/{stock}_forecast.html"
    # 如果想要保存为静态图片： plt.savefig(f"{stock}_forecast.png")
    # 也可以用 mpld3.export_html(...) 保存成网页交互图
    plt.savefig(f"image/{stock}_forecast.png", dpi=150)
    plt.show()

def main():
    # 加载 future.json
    fp = Path("future.json")
    if not fp.exists():
        raise FileNotFoundError("请先生成 future.json")
    future_json = json.loads(fp.read_text(encoding="utf-8"))
    import os
    if os.path.exists("image"):
        print("image 目录已存在，跳过创建")
    else:
        os.makedirs("image", exist_ok=True)
        print("创建 image 目录")

    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    script_dir = Path(__file__).parent
    font_path = script_dir / "ttc" / "GenKiGothic2JP-B-03.ttf"
    font_manager.fontManager.addfont(str(font_path))

    # 2) Query the internal font name
    prop = font_manager.FontProperties(fname=str(font_path))
    font_name = prop.get_name()
    print(f"Loaded font, internal name = {font_name}")

    # 3) Tell Matplotlib to use that name
    plt.rcParams["font.family"] = [font_name]




    end_date = future_json['metadata']["prediction_start"]

    # 对每只股票分别画图
    for stock in future_json["instruments"][:10]:  # 只画前10只示例
        print(f"绘制 {stock} …")
        plot_stock_forecast(stock, future_json, end_date)

if __name__ == "__main__":
    main()