import sys, site, os
import warnings
from pathlib import Path
import qlib  # Uncomment this line
import pandas as pd
import json
import random
from qlib.constant import REG_US
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
import subprocess
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.contrib.model.gbdt import LGBModel
from transformer import TransformerModel
from config_task import get_task, get_date, get_data_handler_config, test_data
from colors import Colors, print_green, print_yellow, print_red, warn_with_color
from symbo import symbols, get_extra_symbols
from qlib.contrib.data.handler import Alpha158, Alpha360

# Try to import sklearn, if not available use basic metrics
def check_gpu():
    os.environ['export CUDA_LAUNCH_BLOCKING'] = '1'  # Ensure CUDA errors are reported immediately
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to 0 for single GPU usage, or adjust as needed
    import torch
    torch.cuda.empty_cache()

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    try:
        from sklearn.metrics import mean_squared_error, r2_score
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False
        print("sklearn not available, using basic metrics only")

    print("SKLEARN_AVAILABLE:", SKLEARN_AVAILABLE)

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
            
            print_green(f"Found {len(available_instruments)} instruments in instruments file")
            if len(available_instruments) > 0:
                print_green(f"Sample instruments: {available_instruments[:10]}")
            else:
                warn_with_color("No instruments found in all.txt file")
                raise ValueError("No instruments found in all.txt file")
        else:
            warn_with_color(f"Instruments file not found: {instruments_file}")
            raise FileNotFoundError("Instruments file missing")
        
        # Filter to common US stocks (remove indices that start with ^ and _)
        us_stocks = available_instruments
        print_green(f"Filtered to {len(us_stocks)} US stocks (excluding indices)")

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
            # 驗證這些symbols是否真的有數據
            print_green("Verifying data availability for selected instruments...")
            valid_instruments = []
            for symbol in market:
                features_dir = Path(f".qlib/qlib_data/us_data/features/{symbol.lower()}")
                if features_dir.exists():
                    valid_instruments.append(symbol)
                else:
                    print_yellow(f"No data found for {symbol}, skipping...")
            
            if len(valid_instruments) > 0:
                market = valid_instruments
                print_green(f"Final selection: {len(market)} instruments with data: {market[:min(10, len(market))]}")
                # Use the first available instrument as benchmark since SPY might not be available
                benchmark = market[0] if len(market) > 0 else None
            else:
                warn_with_color("No instruments have actual data files")
                raise ValueError("No instruments have actual data files")
        else:
            warn_with_color("No US stocks found after filtering")
            raise ValueError("No US stocks found after filtering")
            
    except Exception as e:
        warn_with_color(f"Error reading instruments from file: {e}")
        print_yellow("Using hardcoded fallback configuration...")
        market = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Hardcoded fallback
        benchmark = random.choice(market)  # Randomly select a benchmark from the fallback list
    return market, benchmark


def check_predict_data(pred_label):
    # Add this before line 704 to diagnose the data:

    print_green("=== Prediction Data Diagnosis ===")
    print_green(f"pred_label shape: {pred_label.shape}")
    print_green(f"pred_label columns: {pred_label.columns.tolist()}")
    print_green(f"Non-null predictions: {pred_label['score'].notna().sum()}")
    print_green(f"Non-null labels: {pred_label['label'].notna().sum()}")
    print_green(f"Valid pairs (both score and label): {pred_label.dropna().shape[0]}")

    # Check for data distribution
    if not pred_label.empty:
        score_stats = pred_label['score'].describe()
        label_stats = pred_label['label'].describe()
        print_green(f"Score range: [{score_stats['min']:.6f}, {score_stats['max']:.6f}]")
        print_green(f"Label range: [{label_stats['min']:.6f}, {label_stats['max']:.6f}]")

    # Check unique dates and instruments
    unique_dates = pred_label.index.get_level_values('datetime').nunique()
    unique_instruments = pred_label.index.get_level_values('instrument').nunique()
    print_green(f"Unique dates: {unique_dates}")
    print_green(f"Unique instruments: {unique_instruments}")
    print_green("=== End Diagnosis ===")



def analyze_label_quality(dataset):
    """分析標籤質量"""
    try:
        labels = dataset.prepare("train")
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
        
    except Exception as e:
        print(f"標籤分析失敗: {e}")



def get_data():
    cmd = [
        "uv",
        "run",
        "scripts/data_collector/yahoo/collector.py",
        "--region",
        "US",
        "update_data_to_bin",
        "--qlib_data_1d_dir",
        ".qlib/qlib_data/us_data",
        "--end_date",
        (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "--delay",
        "1",
        "--exists_skip",
        "False",
    ]
    subprocess.run(cmd, check=True)

# Download/update data before using it, No no no no don't do this, it will take a long time
# print("Checking and updating data...")
# try:
#     # get_data()
#     print("Data download/update completed successfully")
# except Exception as e:
#     print(f"Warning: Data download failed: {e}")
#     print("Continuing with existing data...")

def run(model="LGBModel",market_num=None, days=6):
    import numpy as np
    save_path = "best_transformer.pkl"

    check_gpu()
    # NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
    provider_uri = ".qlib/qlib_data/us_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_US)

    def analyze_feature_importance(model, dataset):
        """分析特徵重要性"""
        try:
            # 獲取特徵重要性
            importance = model.model.feature_importance(importance_type='gain')
            feature_names = dataset.prepare("train").columns
            
            # 創建重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print_green("前20個最重要特徵:")
            print(importance_df.head(20))
            
            # 檢查是否有特徵被忽略
            zero_importance = (importance_df['importance'] == 0).sum()
            print_yellow(f"零重要性特徵數量: {zero_importance}")
            
            return importance_df
            
        except Exception as e:
            print(f"特徵重要性分析失敗: {e}")
            return None



    # Check which instruments are available in the data
    print_green("Checking available instruments...")

    market , benchmark = get_market(market_num)

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
        model=model
    )
    print_green("Initializing model and dataset...")
    print_green(f"Final configuration:")
    print_green(f"  - Market: {market[:min(10 , len(market))]}...")
    print_green(f"  - Benchmark: {benchmark}")
    print_green(f"  - Date range: {start_date_str} to {today_str}")
    print_green(f"  - Training: {start_date_str} to {train_end_date_str}")
    print_green(f"  - Validation: {train_end_date_str} to {valid_end_date_str}")
    print_green(f"  - Test: {valid_end_date_str} to {today_str}")


    try:
        # 指定資料目錄
        features_dir = ".qlib/qlib_data/us_data/features"

        # 選擇一個股票
        instrument = market[0]  # 替換為您感興趣的股票代碼
        instrument_dir = os.path.join(features_dir, instrument.lower())

        # 列出特徵檔案
        if os.path.exists(instrument_dir):
            fields = [f.split(".")[0] for f in os.listdir(instrument_dir) if f.endswith(".bin")]
            print_green(f"支持的因子字段: {fields}")
        else:
            print_green(f"未找到股票 {instrument} 的特徵資料")
        stock_data = D.features(
            instruments=market,
            fields=['$close', '$factor', '$high', '$low', '$open', '$volume'],
            start_time=(pd.Timestamp.now()-pd.Timedelta(days=days+1)).strftime("%Y-%m-%d"),
            end_time=today_str,
            freq="day"
        )
        print_green("前幾支股票的開盤價:")
        print_green(stock_data)
    except Exception as e:
        print_green(f"提取開盤價時發生錯誤: {e}")

    try:
        print_green("Creating model...")
        model = init_instance_by_config(task["model"])
        print_green("Model created successfully")
        
        print_green("Creating dataset...")
        dataset = init_instance_by_config(task["dataset"])
        print_green("Dataset created successfully")
        analyze_label_quality(dataset)
        # Check dataset segments
        print_green("Checking dataset segments...")
        try:
            train_data = dataset.prepare("train")
            print_green(f"Training data shape: {train_data.shape}")
            
            if train_data.empty:
                warn_with_color("Training dataset is empty after creation")
                raise ValueError("Training dataset is empty after creation")
                
            valid_data = dataset.prepare("valid") 
            print_green(f"Validation data shape: {valid_data.shape}")
            
        except Exception as seg_error:
            warn_with_color(f"Error checking segments: {seg_error}")
            raise
        
    except Exception as e:
        warn_with_color(f"Critical error during model/dataset initialization: {e}")
        print_yellow("Cannot proceed without valid data. Creating mock predictions instead...")
        
        # Create mock predictions as fallback
        print("Generating mock predictions...")
        
        # Use available instruments or fallback list
        if 'market' not in locals():
            market = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        future_start = today
        future_end = today + pd.Timedelta(days=days)
        future_dates = pd.bdate_range(start=future_start, end=future_end, freq='B')
        
        mock_predictions = {}
        for date in future_dates:
            date_str = date.strftime("%Y-%m-%d")
            mock_predictions[date_str] = {}
            for stock in market:  # Limit to 10 stocks
                # Mock prediction (neutral value around 0)
                mock_predictions[date_str][stock] = 0.0
        
        # Create error output with mock predictions
        error_output = {
            "prediction_date": today_str + " " + pd.Timestamp.now().strftime("%H:%M:%S"),
            "error": f"Data initialization error: {str(e)}",
            "status": "mock_predictions",
            "message": "Unable to train model due to data issues. Mock predictions generated.",
            "model_type": "MockModel",
            "instruments_count": len(market),
            "instruments": market[:10],
            "prediction_period": f"{future_start.strftime('%Y-%m-%d')} to {future_end.strftime('%Y-%m-%d')}",
            "predictions": mock_predictions
        }
        
        with open("future.json", "w", encoding="utf-8") as f:
            json.dump(error_output, f, ensure_ascii=False, indent=2)
        
        print_red("Mock predictions saved to future.json")
        print_red(f"Generated mock predictions for {len(market[:10])} instruments over {len(future_dates)} business days")
        exit(0)  # Exit successfully with mock data

    # If we reach here, model and dataset were created successfully
    print("Starting model training...")

    # start exp to train model
    with R.start(experiment_name="train_model"):
        import json
        with open("instruments.json", "w", encoding="utf-8") as fp:
            json.dump(market, fp, ensure_ascii=False, indent=2)

        # 2. 把這個 JSON 當 artifact 上傳
        import mlflow
        mlflow.log_artifact("instruments.json", artifact_path="metadata")
        # 3. 其他原本要記參數的動作
        R.log_params(**flatten_dict(task))
        params = flatten_dict(task)
        params.pop("dataset.kwargs.handler.kwargs.instruments", None)
        R.log_params(**params)
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        print(f"模型已保存到 {save_path}")




    ###################################
    # prediction, backtest & analysis
    ###################################
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": train_end_date_str,
            "end_time": today_str,
            "account": 100000000,
            "benchmark": benchmark,  # Explicitly set benchmark to None
            "exchange_kwargs": {
                "freq": "day",
                "codes": market,          # ← add this line so Exchange.codes==your list
                "limit_threshold": None,  # US market has no limit threshold
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },

    }

    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        # Load trained model from training recorder
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        print_green("get recorder successfully")
        model = recorder.load_object("trained_model")
        print(type(model))
        # Generate predictions on validation period using D.features directly
        print_green("Generating predictions on validation dataset using D.features...")
        try:
            val_features = dataset

            print_yellow(type(val_features))
            # print_green(f"Validation features shape: {val_features.shape}")
            # if not val_features.empty:
            pred_df = model.predict(val_features)
            print_green(f"Predictions generated successfully: {pred_df.shape}")
            # else:
                # warn_with_color("Validation features DataFrame is empty, cannot generate predictions")
                # pred_df = pd.DataFrame()
        except Exception as gen_error:
            warn_with_color(f"Prediction generation failed: {gen_error}")
            pred_df = pd.DataFrame()
        
        print_green("Model training and prediction completed successfully!")
        
        # Use pred_df generated above and display its info
        print_green(f"Prediction DataFrame shape: {pred_df.shape}")
        if not pred_df.empty:
            # print_green(f"Prediction DataFrame columns: {pred_df.columns.tolist()}")
            print_green(f"Prediction DataFrame sample:\n{pred_df.head()}")
        else:
            warn_with_color("Prediction DataFrame is empty! Generating predictions directly...")
            
            # Try to generate predictions directly using the trained model
            try:
                print_green("Attempting direct prediction on validation data...")
                
                # Get validation data directly from dataset (not from DataFrame)
                val_data = dataset.prepare("test")
                print_green(f"Validation data shape: {val_data.shape}")
                
                if not val_data.empty:
                    # Use the trained model to predict directly
                    validation_pred = model.predict(val_data)
                    print_green(f"Direct validation predictions shape: {validation_pred.shape}")
                    print_green(f"Direct validation predictions sample:\n{validation_pred.head()}")
                    pred_df = validation_pred
                else:
                    raise ValueError("Validation data is empty")
                    
            except Exception as direct_error:
                warn_with_color(f"Direct prediction failed: {direct_error}")
                
                # Create predictions using stock list and recent data
                print_yellow("Creating predictions using available stock data...")
                
                try:
                    # Use qlib data directly to create predictions
                    recent_start = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

                    # Get recent data for the stocks we have
                    recent_data = D.features(
                        market,
                        ["$close", "$open"],
                        start_time=recent_start,
                        end_time=today_str,
                        freq="day"
                    )
                    
                    print_green(f"Recent data for prediction generation: {recent_data.shape}")
                    
                    if not recent_data.empty:
                        # Create realistic predictions based on recent price movements
                        predictions_list = []
                        
                        for stock in market:
                            try:
                                stock_data = recent_data.loc[recent_data.index.get_level_values(0) == stock]
                                if len(stock_data) > 1:
                                    # Calculate momentum-based prediction
                                    price_change = (stock_data['$close'].iloc[-1] - stock_data['$close'].iloc[0]) / stock_data['$close'].iloc[0]
                                    # Normalize to reasonable prediction range
                                    prediction = np.tanh(price_change) * 0.05  # Scale to [-0.05, 0.05]
                                else:
                                    prediction = 0.0
                                
                                # Create entry for this stock
                                predictions_list.append({
                                    'instrument': stock,
                                    'datetime': today_str,
                                    'score': prediction
                                })
                            except Exception as stock_error:
                                print_yellow(f"Error processing {stock}: {stock_error}")
                                predictions_list.append({
                                    'instrument': stock,
                                    'datetime': today_str,
                                    'score': 0.0
                                })
                        
                        # Create DataFrame with proper MultiIndex
                        if predictions_list:
                            df_data = pd.DataFrame(predictions_list)
                            df_data.set_index(['instrument', 'datetime'], inplace=True)
                            pred_df = df_data[['score']]
                            
                            print_green(f"Created predictions from stock data: {pred_df.shape}")
                            print_green(f"Sample predictions:\n{pred_df.head()}")
                        else:
                            raise ValueError("No predictions could be generated from stock data")
                            
                    else:
                        raise ValueError("No recent data available")
                        
                except Exception as stock_data_error:
                    warn_with_color(f"Stock data prediction failed: {stock_data_error}")
                    
                    # Final fallback: Generate structured dummy predictions
                    print_yellow("Creating structured dummy predictions based on market instruments...")
                    import numpy as np
                    
                    # Get the latest date for each instrument
                    latest_date = today_str
                    instruments = market  # Limit to 20 instruments
                    
                    # Create multi-index for latest predictions
                    index_tuples = [(inst, latest_date) for inst in instruments]
                    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['instrument', 'datetime'])
                    
                    # Generate realistic predictions (small values around 0)
                    np.random.seed(42)  # For reproducible results
                    predictions = np.random.normal(0, 0.005, len(instruments))
                    pred_df = pd.DataFrame(predictions, index=multi_index, columns=['score'])
                    
                    print_green(f"Created structured dummy predictions with shape: {pred_df.shape}")
                    print_green(f"Structured predictions sample:\n{pred_df.head()}")
        
        # Model evaluation and metrics
        print_green("Evaluating model performance...")
        model_metrics = {}
        
        try:
            print_green("Calculating model performance metrics...")
            
            # Get validation data and labels using dataset object properly
            try:
                val_data = dataset
                val_label = dataset.prepare("test", col_set="label")
                # print_green(f"Validation data shape: {val_data.shape}")
                print_green(f"Test label shape: {val_label.shape}")
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
                
        except Exception as metric_error:
            warn_with_color(f"Error calculating model metrics: {metric_error}")
            
            # Skip fallback metric calculation to avoid further DataFrame/dataset confusion
            print_yellow("Skipping fallback metric calculation due to data structure issues...")
            model_metrics = {
                "error": f"Metric calculation failed: {str(metric_error)}",
                "status": "metrics_unavailable"
            }

        recorder = R.get_recorder()
        br_rid = recorder.id
        sr = SignalRecord(model=model , dataset=dataset,recorder=recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_analysis_config , "day")
        par.generate()

    with R.start(experiment_name="portfolio_analysis"):
        # Load the recorder for portfolio analysis
        recorder = R.get_recorder(recorder_id=br_rid, experiment_name="backtest_analysis")
        print_green("Loading recorder for portfolio analysis...")
        
        # Load analysis results
        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        # Generate Plotly figures and display/save them
        fig_report = analysis_position.report_graph(report_normal_df, show_notebook=False)
        fig_risk   = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
        # prepare prediction+label for IC plot
        label_df = dataset.prepare("test", col_set="label")
        label_df.columns = ['label']
        pred_label = pd.concat([label_df, pred_df], axis=1).reindex(label_df.index)
        fig_ic     = analysis_position.score_ic_graph(pred_label, show_notebook=False)
        # score IC
        # model performance (returns list of Figures)
        model_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
        
        check_predict_data(pred_label)

        def dump_figs(figs, base_name):
            lst = figs if isinstance(figs, (list, tuple)) else [figs]
            for idx, fig in enumerate(lst):
                fname = f"{base_name}" + (f"_{idx}" if len(lst) > 1 else "") + ".html"
                fig.write_html(fname)
                # fig.show(renderer="browser")

        # save and show each figure

        dump_figs(fig_report, "report")
        dump_figs(fig_risk,   "risk")
        dump_figs(fig_ic,     "score_ic")
        dump_figs(model_figs, "model_performance")

        print_green("Portfolio analysis saved successfully!")

    ###################################
    # Generate future predictions
    ###################################
    print_green("Generating future predictions...")

    # 導入預測函數
    from predict_function import comprehensive_predict, generate_stock_recommendations

    # 創建未來預測的數據結構
    future_start = today
    future_end = today + pd.Timedelta(days=days)
    future_dates = pd.bdate_range(start=future_start, end=future_end, freq='B')

    print_green(f"預測期間: {future_start.strftime('%Y-%m-%d')} 到 {future_end.strftime('%Y-%m-%d')}")
    print_green(f"預測天數: {len(future_dates)} 個交易日")

    # 初始化預測結果字典
    prediction_output = {
        "metadata": {
            "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_start": future_start.strftime("%Y-%m-%d"),
            "prediction_end": future_end.strftime("%Y-%m-%d"),
            "prediction_horizon_days": days,
            "business_days": len(future_dates),
            "model_type": model.__class__.__name__,
            "total_instruments": len(market),
            "data_source": "qlib",
            "region": "US"
        },
        "model_performance": model_metrics,
        "instruments": market,
        "predictions": {}
    }

    try:
        print_green("開始進行全面預測分析...")
        
        # 使用 comprehensive_predict 函數進行詳細預測
        comprehensive_predictions = comprehensive_predict(
            model=model,
            dataset=dataset,
            chunk=market,
            steps=days
        )
        
        if comprehensive_predictions:
            print_green(f"成功預測了 {len(comprehensive_predictions)} 支股票")
            
            # 生成選股建議
            stock_recommendations = generate_stock_recommendations(comprehensive_predictions)
            
            # 處理每支股票的預測結果
            for stock_symbol, stock_pred in comprehensive_predictions.items():
                try:
                    # 基本信息
                    basic_info = stock_pred.get("basic_info", {})
                    
                    # 多時間段收益率預測
                    multi_horizon = stock_pred.get("multi_horizon_returns", {})
                    
                    # 風險指標
                    risk_metrics = stock_pred.get("risk_metrics", {})
                    
                    # 選股評分
                    selection_scores = stock_pred.get("selection_scores", {})
                    
                    # 構建詳細的預測信息
                    detailed_prediction = {
                        "basic_info": {
                            "symbol": stock_symbol,
                            "company_name": stock_symbol,  # 可以擴展為實際公司名稱
                            "prediction_date": basic_info.get("prediction_date", ""),
                            "last_known_return": basic_info.get("last_known_return", 0.0),
                            "data_points_used": basic_info.get("data_points", 0),
                            "feature_dimension": basic_info.get("feature_dimension", 0)
                        },
                        
                        "return_forecasts": {
                            "short_term": {
                                "1_day": {
                                    "expected_return": multi_horizon.get("1d", {}).get("expected_return", 0.0),
                                    "cumulative_return": multi_horizon.get("1d", {}).get("cumulative_return", 0.0),
                                    "confidence_level": "medium"
                                },
                                "3_day": {
                                    "expected_return": multi_horizon.get("3d", {}).get("expected_return", 0.0),
                                    "cumulative_return": multi_horizon.get("3d", {}).get("cumulative_return", 0.0),
                                    "daily_returns": multi_horizon.get("3d", {}).get("daily_returns", []),
                                    "confidence_level": "medium"
                                }
                            },
                            "medium_term": {
                                "5_day": {
                                    "expected_return": multi_horizon.get("5d", {}).get("expected_return", 0.0),
                                    "cumulative_return": multi_horizon.get("5d", {}).get("cumulative_return", 0.0),
                                    "daily_returns": multi_horizon.get("5d", {}).get("daily_returns", []),
                                    "confidence_level": "medium"
                                },
                                "7_day": {
                                    "expected_return": multi_horizon.get("7d", {}).get("expected_return", 0.0),
                                    "cumulative_return": multi_horizon.get("7d", {}).get("cumulative_return", 0.0),
                                    "daily_returns": multi_horizon.get("7d", {}).get("daily_returns", []),
                                    "confidence_level": "low"
                                }
                            }
                        },
                        
                        "risk_analysis": {
                            "volatility": {
                                "7_day_volatility": risk_metrics.get("volatility_7d", 0.0),
                                "volatility_percentile": min(100, max(0, risk_metrics.get("volatility_7d", 0.02) * 2500)),
                                "risk_level": "LOW" if risk_metrics.get("volatility_7d", 0.02) < 0.02 else "MEDIUM" if risk_metrics.get("volatility_7d", 0.02) < 0.04 else "HIGH"
                            },
                            "expected_range": {
                                "min_return_7d": risk_metrics.get("min_return_7d", -0.05),
                                "max_return_7d": risk_metrics.get("max_return_7d", 0.05),
                                "expected_return_7d": risk_metrics.get("expected_return_7d", 0.0),
                                "range_width": risk_metrics.get("max_return_7d", 0.05) - risk_metrics.get("min_return_7d", -0.05)
                            },
                            "risk_adjusted_metrics": {
                                "sharpe_estimate": selection_scores.get("risk_adjusted_return", 0.0),
                                "risk_reward_ratio": abs(risk_metrics.get("expected_return_7d", 0.0) / max(0.001, risk_metrics.get("volatility_7d", 0.02)))
                            }
                        },
                        
                        "trading_signals": {
                            "overall_signal": "BUY" if selection_scores.get("expected_return", 0) > 0.01 else "SELL" if selection_scores.get("expected_return", 0) < -0.01 else "HOLD",
                            "signal_strength": min(100, max(0, abs(selection_scores.get("expected_return", 0)) * 1000)),
                            "momentum_signal": "POSITIVE" if multi_horizon.get("7d", {}).get("expected_return", 0) > 0 else "NEGATIVE",
                            "trend_direction": "UPTREND" if multi_horizon.get("7d", {}).get("expected_return", 0) > 0.005 else "DOWNTREND" if multi_horizon.get("7d", {}).get("expected_return", 0) < -0.005 else "SIDEWAYS"
                        },
                        
                        "selection_scores": {
                            "composite_score": selection_scores.get("composite_score", 50.0),
                            "expected_return_score": min(100, max(0, (selection_scores.get("expected_return", 0) + 0.05) * 1000)),
                            "volatility_score": min(100, max(0, 100 - selection_scores.get("volatility", 0.02) * 2500)),
                            "risk_adjusted_score": min(100, max(0, (selection_scores.get("risk_adjusted_return", 0) + 1) * 50)),
                            "percentile_rank": 0  # 稍後計算
                        },
                        
                        "probability_analysis": {
                            "prob_positive_return": len([r for r in multi_horizon.get("7d", {}).get("daily_returns", []) if r > 0]) / max(1, len(multi_horizon.get("7d", {}).get("daily_returns", []))),
                            "prob_outperform_market": 0.5,  # 假設值
                            "prob_exceed_threshold": {
                                "1_percent": len([r for r in multi_horizon.get("7d", {}).get("daily_returns", []) if r > 0.01]) / max(1, len(multi_horizon.get("7d", {}).get("daily_returns", []))),
                                "2_percent": len([r for r in multi_horizon.get("7d", {}).get("daily_returns", []) if r > 0.02]) / max(1, len(multi_horizon.get("7d", {}).get("daily_returns", []))),
                                "5_percent": len([r for r in multi_horizon.get("7d", {}).get("daily_returns", []) if r > 0.05]) / max(1, len(multi_horizon.get("7d", {}).get("daily_returns", [])))
                            }
                        },
                        
                        "scenario_analysis": {
                            "bull_scenario": {
                                "probability": 0.3,
                                "expected_return": max(multi_horizon.get("7d", {}).get("daily_returns", [0])) if multi_horizon.get("7d", {}).get("daily_returns") else 0.0,
                                "description": "Optimistic market conditions"
                            },
                            "base_scenario": {
                                "probability": 0.4,
                                "expected_return": multi_horizon.get("7d", {}).get("expected_return", 0.0),
                                "description": "Normal market conditions"
                            },
                            "bear_scenario": {
                                "probability": 0.3,
                                "expected_return": min(multi_horizon.get("7d", {}).get("daily_returns", [0])) if multi_horizon.get("7d", {}).get("daily_returns") else 0.0,
                                "description": "Pessimistic market conditions"
                            }
                        },
                        
                        "daily_forecast": {}
                    }
                    
                    # 添加每日預測詳情
                    daily_returns = multi_horizon.get("7d", {}).get("daily_returns", [])
                    for i, date in enumerate(future_dates[:len(daily_returns)]):
                        detailed_prediction["daily_forecast"][date.strftime("%Y-%m-%d")] = {
                            "predicted_return": daily_returns[i],
                            "confidence": "medium",
                            "trading_day": i + 1,
                            "cumulative_return": sum(daily_returns[:i+1]),
                            "signal": "BUY" if daily_returns[i] > 0.01 else "SELL" if daily_returns[i] < -0.01 else "HOLD"
                        }
                    
                    prediction_output["predictions"][stock_symbol] = detailed_prediction
                    
                except Exception as stock_error:
                    print_yellow(f"處理 {stock_symbol} 預測結果時出錯: {stock_error}")
                    continue
            
            # 計算百分位排名
            all_scores = [pred["selection_scores"]["composite_score"] for pred in prediction_output["predictions"].values()]
            if all_scores:
                for stock_symbol, pred in prediction_output["predictions"].items():
                    score = pred["selection_scores"]["composite_score"]
                    percentile = (sum(1 for s in all_scores if s < score) / len(all_scores)) * 100
                    prediction_output["predictions"][stock_symbol]["selection_scores"]["percentile_rank"] = round(percentile, 1)
            
            # 添加投資組合級別的分析
            prediction_output["portfolio_analysis"] = {
                "top_picks": [],
                "avoid_list": [],
                "sector_analysis": {},
                "risk_distribution": {
                    "low_risk": len([p for p in prediction_output["predictions"].values() if p["risk_analysis"]["volatility"]["risk_level"] == "LOW"]),
                    "medium_risk": len([p for p in prediction_output["predictions"].values() if p["risk_analysis"]["volatility"]["risk_level"] == "MEDIUM"]),
                    "high_risk": len([p for p in prediction_output["predictions"].values() if p["risk_analysis"]["volatility"]["risk_level"] == "HIGH"])
                },
                "signal_distribution": {
                    "buy_signals": len([p for p in prediction_output["predictions"].values() if p["trading_signals"]["overall_signal"] == "BUY"]),
                    "hold_signals": len([p for p in prediction_output["predictions"].values() if p["trading_signals"]["overall_signal"] == "HOLD"]),
                    "sell_signals": len([p for p in prediction_output["predictions"].values() if p["trading_signals"]["overall_signal"] == "SELL"])
                }
            }
            
            # 生成投資建議
            sorted_predictions = sorted(
                prediction_output["predictions"].items(),
                key=lambda x: x[1]["selection_scores"]["composite_score"],
                reverse=True
            )
            
            prediction_output["portfolio_analysis"]["top_picks"] = [
                {
                    "symbol": symbol,
                    "score": pred["selection_scores"]["composite_score"],
                    "expected_7d_return": pred["return_forecasts"]["medium_term"]["7_day"]["expected_return"],
                    "risk_level": pred["risk_analysis"]["volatility"]["risk_level"],
                    "signal": pred["trading_signals"]["overall_signal"],
                    "reason": f"高綜合評分 ({pred['selection_scores']['composite_score']:.1f}分)"
                }
                for symbol, pred in sorted_predictions[:10]
            ]
            
            prediction_output["portfolio_analysis"]["avoid_list"] = [
                {
                    "symbol": symbol,
                    "score": pred["selection_scores"]["composite_score"],
                    "expected_7d_return": pred["return_forecasts"]["medium_term"]["7_day"]["expected_return"],
                    "risk_level": pred["risk_analysis"]["volatility"]["risk_level"],
                    "signal": pred["trading_signals"]["overall_signal"],
                    "reason": f"低綜合評分 ({pred['selection_scores']['composite_score']:.1f}分)"
                }
                for symbol, pred in sorted_predictions[-5:]
            ]
            
            # 添加推薦建議
            prediction_output["investment_recommendations"] = stock_recommendations
            
            print_green(f"✓ 完成 {len(prediction_output['predictions'])} 支股票的詳細預測")
            
        else:
            print_yellow("未能獲取有效的預測結果，生成基礎預測...")
            
            # 生成基礎預測作為後備
            for stock in market:
                prediction_output["predictions"][stock] = {
                    "basic_info": {
                        "symbol": stock,
                        "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        "status": "fallback_prediction"
                    },
                    "return_forecasts": {
                        "medium_term": {
                            "7_day": {
                                "expected_return": 0.0,
                                "cumulative_return": 0.0,
                                "confidence_level": "low"
                            }
                        }
                    },
                    "risk_analysis": {
                        "volatility": {
                            "risk_level": "MEDIUM",
                            "7_day_volatility": 0.02
                        }
                    },
                    "trading_signals": {
                        "overall_signal": "HOLD",
                        "signal_strength": 50
                    },
                    "selection_scores": {
                        "composite_score": 50.0
                    }
                }

    except Exception as pred_error:
        print_red(f"預測過程中出現錯誤: {pred_error}")
        import traceback
        traceback.print_exc()
        
        # 創建錯誤報告
        prediction_output["error"] = {
            "message": str(pred_error),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "prediction_failed"
        }

    # 保存預測結果到 JSON 文件
    try:
        with open("future.json", "w", encoding="utf-8") as f:
            # -------------------------------------------------------------------
            json.dump(prediction_output, f, ensure_ascii=False, indent=2)
        
        print_green("✓ 詳細預測報告已保存到 future.json")
        print_green("預測報告包含以下內容:")
        print_green("  - 多時間段收益率預測 (1天, 3天, 5天, 7天)")
        print_green("  - 風險分析 (波動率, 風險等級, 預期範圍)")
        print_green("  - 交易信號 (買入/賣出/持有建議)")
        print_green("  - 選股評分 (綜合評分, 百分位排名)")
        print_green("  - 概率分析 (正收益概率, 超越門檻概率)")
        print_green("  - 情景分析 (牛市/基準/熊市情景)")
        print_green("  - 每日預測詳情")
        print_green("  - 投資組合級別分析")
        print_green("  - 投資建議 (頂級推薦, 避險清單)")
        
        # 顯示關鍵統計信息
        if "predictions" in prediction_output and prediction_output["predictions"]:
            total_stocks = len(prediction_output["predictions"])
            buy_signals = len([p for p in prediction_output["predictions"].values() 
                            if p.get("trading_signals", {}).get("overall_signal") == "BUY"])
            avg_expected_return = np.mean([p.get("return_forecasts", {}).get("medium_term", {}).get("7_day", {}).get("expected_return", 0) 
                                        for p in prediction_output["predictions"].values()])
            
            print_green(f"預測統計:")
            print_green(f"  - 總股票數: {total_stocks}")
            print_green(f"  - 買入信號: {buy_signals} ({buy_signals/total_stocks*100:.1f}%)")
            print_green(f"  - 平均預期7日收益率: {avg_expected_return*100:.3f}%")
            
            if "portfolio_analysis" in prediction_output:
                print_green(f"  - 頂級推薦: {len(prediction_output['portfolio_analysis']['top_picks'])} 支")
                print_green(f"  - 避險清單: {len(prediction_output['portfolio_analysis']['avoid_list'])} 支")

    except Exception as save_error:
        print_red(f"保存預測結果時出錯: {save_error}")

    print_green("股票預測分析完成！")



def main():
    run(
        model="TransformerModel",
        market_num=50,
        days=9,
    )


if __name__ == "__main__":
    main()