import pandas as pd
from pathlib import Path
from yahooquery import Ticker
import fire

class FXCollector:
    """
    抓取主要货币对汇率数据（日线）并保存为 CSV，
    方便后续用 dump_bin.py 转成 Qlib bin 格式。
    """

    def __init__(
        self,
        output_dir: str = None,
        start: str = None,
        end: str = None,
        interval: str = "1d",
    ):
        # 输出目录，默认当前脚本同级 fx_data
        self.output_dir = (
            Path(output_dir) if output_dir else Path(__file__).parent / "fx_data"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start = start  # e.g. "2021-01-01"
        self.end = end      # e.g. "2025-06-18"
        self.interval = interval

        # Yahoo 格式的 ticker
        # USD/TWD, USD/CNY, EUR/USD, CHF/USD
        self.pairs = {
            "USDTWD": "USDTWD=X",
            "USDCNY": "USDCNY=X",
            "EURUSD": "EURUSD=X",
            "CHFUSD": "CHFUSD=X",
        }

    def download(self):
        """
        下载指定时间区间内的汇率历史（OHLCV），
        按 symbol 拆分并保存为多份 CSV，方便后续用 dump_bin.py 转成 Qlib bin 格式。
        """
        symbols = list(self.pairs.values())
        print(f"Fetching FX rates for: {symbols}")
        tk = Ticker(symbols, asynchronous=False)
        df = tk.history(
            interval=self.interval,
            start=self.start,
            end=self.end,
            adj_ohlc=True,
        )
        if df is None or df.empty:
            print("❌ 未获取到任何汇率数据")
            return

        df = df.reset_index()[["symbol", "date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 按 symbol 拆分
        for sym, subdf in df.groupby("symbol"):
            out_csv = self.output_dir / f"{sym}.csv"
            subdf.to_csv(out_csv, index=False)
            print(f"✅ {sym}: saved {len(subdf)} rows to {out_csv}")

        print(f"✅ 全部拆分完成，共 {len(df['symbol'].unique())} 个文件保存在 {self.output_dir}")
    def normalize(self):
        """
        调用现有 normalize 脚本，把 CSV 转为 Qlib Bin
        """
        from dump_bin import DumpDataUpdate

        # 假设 qlib 存储目录 ~/.qlib/qlib_data/fx
        qlib_fx_dir = Path(".qlib/qlib_data/us_data")
        qlib_fx_dir.mkdir(parents=True, exist_ok=True)

        dumper = DumpDataUpdate(
            csv_path=str(self.output_dir),
            qlib_dir=str(qlib_fx_dir),
            exclude_fields="symbol,date",
            freq="day",
        )
        dumper.dump()
        print(f"✅ 已将 FX CSV 转成 Qlib 数据，存放在 {qlib_fx_dir}")

if __name__ == "__main__":
    fire.Fire(FXCollector)