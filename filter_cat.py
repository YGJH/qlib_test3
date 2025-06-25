import pandas as pd
import os
import random
from colors import Colors, print_green, print_yellow, warn_with_color, print_red
from qlib.data import D
import qlib
from qlib.constant import REG_US
import sys
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

def get_data_config(
        start_date_str: str = "2021-01-01",
        train_end_date_str: str = "2023-01-01",
        today_str: str = "2023-01-01",
        today: pd.Timestamp = pd.Timestamp.now(),
        market: list = None,
    ):
    try:
        data_handler_config = {
            "start_time": start_date_str,
            "end_time": train_end_date_str,
            "fit_start_time": start_date_str,
            "fit_end_time": train_end_date_str,
            "instruments": market,  # 確保`market`包含有效的股票代碼
        }

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
                    "valid": (train_end_date_str, today_str),
                    "test": (
                        today_str,
                        (today + pd.Timedelta(days=9)).strftime("%Y-%m-%d"),
                    ),
                },
            },
        }
        return transformer_dataset_cfg
    except Exception as e:
        raise ValueError(f"Failed to get data config: {e}")

def analyze_label_quality(dataset):
    """分析標籤質量"""
    try:
        labels = dataset.prepare("train", col_set="label")
        print_green(f"標籤統計信息:")
        print(f"  - 標籤形狀: {labels.shape}")
        print(f"  - 標籤均值: {labels.mean().iloc[0]:.6f}")
        print(f"  - 標籤標準差: {labels.std().iloc[0]:.6f}")
        print(f"  - 標籤範圍: [{labels.min().iloc[0]:.6f}, {labels.max().iloc[0]:.6f}]")
        print(f"  - 缺失值數量: {labels.isnull().sum().iloc[0]}")
        
        # 檢查標籤分佈
        label_values = labels.iloc[:, 0].values
        print(f"  - 正值比例: {(label_values > 0).mean():.4f}")
        print(f"  - 零值比例: {(label_values == 0).mean():.4f}")
        print(f"  - 負值比例: {(label_values < 0).mean():.4f}")
        return labels
    except Exception as e:
        print(f"標籤分析失敗: {e}")


def check_instrument_data(market_folder, start_date_str, today_str, features_dir):
    """
    檢查單個股票的資料是否存在且無NaN值
    
    Args:
        market_folder: 市場代碼資料夾名稱
        start_date_str: 開始日期字串
        today_str: 結束日期字串  
        features_dir: 特徵資料夾路徑
        
    Returns:
        str or None: 如果資料有效則返回市場代碼，否則返回None
    """
    try:
        market = market_folder.split('.')[0]
        print_green(f"Processing market: {market}")
        
        stock_data = D.features(
            instruments=[market],
            fields=['$close', '$factor', '$high', '$low', '$open', '$volume'],
            start_time=start_date_str,
            end_time=today_str,
            freq="day"
        )
        
        if stock_data.empty:
            warn_with_color(f"No data available for market: {market}")
            shutil.rmtree(os.path.join(features_dir, market_folder), ignore_errors=True)
            return None
        
        print_green(stock_data)
        # 檢查是否有NaN值
        has_nan = False
        for data in stock_data.index:
            if "NaN" in str(data):
                has_nan = True
                break
        
        # 也檢查資料值是否有NaN
        if stock_data.isnull().any().any():
            has_nan = True
        
        if has_nan:
            shutil.rmtree(os.path.join(features_dir, market_folder), ignore_errors=True)
            # warn_with_color(f"NaN data found for market: {market}")
            return None
        
        print_green(f"Found valid data for market: {market}")
        return market
        
    except Exception as e:
        # warn_with_color(f"Error processing market {market_folder}: {str(e)}")
        try:
            pass
            # shutil.rmtree(os.path.join(features_dir, market_folder), ignore_errors=True)
        except:
            pass
        return None

def process_instruments_parallel(market_folders, start_date_str, today_str, features_dir, max_workers=None):
    """
    並行處理多個股票的資料檢查
    
    Args:
        market_folders: 市場代碼資料夾列表
        start_date_str: 開始日期字串
        today_str: 結束日期字串
        features_dir: 特徵資料夾路徑
        max_workers: 最大工作進程數，None為自動檢測
        
    Returns:
        list: 有效的市場代碼列表
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(market_folders))
    
    # 創建部分應用函數，固定部分參數
    check_func = partial(
        check_instrument_data,
        start_date_str=start_date_str,
        today_str=today_str,
        features_dir=features_dir
    )
    
    valid_instruments = []
    
    with Pool(processes=max_workers) as pool:
        print_green(f"Starting parallel processing with {max_workers} workers...")
        results = pool.map(check_func, market_folders)
        
        # 過濾掉None結果
        valid_instruments = [result for result in results if result is not None]
    
    return valid_instruments

def filter(chunk_size: int=None, regen: bool=False, max_workers: int=None):
    print(pd.Timestamp.now().strftime("%Y-%m-%d"))

    # 定義文件路徑
    all_txt_path = '.qlib/qlib_data/us_data/instruments/all.txt'
    features_dir = '.qlib/qlib_data/us_data/features'
    start_date = pd.Timestamp("2021-01-01")
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    today_str = (pd.Timestamp.now() - pd.Timedelta(days=4)).strftime("%Y-%m-%d")

    print_green(f"Data range: {start_date_str} to {today_str}")

    # 初始化qlib
    qlib.init(provider_uri=".qlib/qlib_data/us_data", region=REG_US)
    
    # 獲取所有市場資料夾
    market_folders = os.listdir(features_dir)
    
    # 如果指定了chunk_size，限制處理數量
    if chunk_size is not None:
        target_count = 20 * chunk_size
        market_folders = market_folders[:target_count]
    
    print_green(f"Processing {len(market_folders)} market folders...")
    
    # 並行處理所有股票
    new_instruments = process_instruments_parallel(
        market_folders, 
        start_date_str, 
        today_str, 
        features_dir,
        max_workers
    )
    
    print_green(f"Found {len(new_instruments)} valid instruments")
    new_instruments = list(map(lambda x: x + '\t' + start_date_str + '\t' + today_str, new_instruments))
    # 隨機打亂順序
    random.shuffle(new_instruments)
    
    # 如果指定了chunk_size，限制最終結果數量
    if chunk_size is not None:
        new_instruments = new_instruments[:chunk_size]
    
    # 寫入文件
    with open(all_txt_path, 'w') as f:
        f.write('\n'.join(new_instruments))
    with open('.qlib/qlib_data/us_data/instruments/instruments.txt', 'w') as f:
        f.write('\n'.join(new_instruments))
    
    print_green(f"Final selection: {len(new_instruments)} instruments")

def main(chunk_size: int=None, regen: bool=False, max_workers: int=None):
    print(f"Regenerate: {regen}")
    filter(chunk_size, regen=regen, max_workers=max_workers)

if __name__ == "__main__":
    chunk_size = None
    regen = False
    max_workers = None
    
    if len(sys.argv) > 1:
        chunk_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        regen = sys.argv[2] == 'regen'
    if len(sys.argv) > 3:
        max_workers = int(sys.argv[3])
    
    main(chunk_size=chunk_size, regen=regen, max_workers=max_workers)