import numpy as np
from datetime import datetime
import pandas as pd
import qlib
from qlib.data.dataset.handler import DataHandlerLP
import torch


def generate_stock_recommendations(predictions):
    """生成選股建議"""
    recommendations = {
        "top_picks": {},
        "avoid_list": {},
        "category_leaders": {},
        "risk_adjusted_picks": {}
    }
    
    # 按綜合評分排序
    scored_stocks = [(stock, data["selection_scores"]["composite_score"]) 
                    for stock, data in predictions.items() 
                    if "selection_scores" in data]
    scored_stocks.sort(key=lambda x: x[1], reverse=True)
    
    # 頂級推薦（前10）
    recommendations["top_picks"] = {
        stock: {
            "score": score,
            "expected_7d_return": predictions[stock]["multi_horizon_returns"].get("7d", {}).get("expected_return", 0),
            "risk_level": "LOW" if predictions[stock]["risk_metrics"]["volatility_7d"] < 0.02 else "MEDIUM" if predictions[stock]["risk_metrics"]["volatility_7d"] < 0.04 else "HIGH"
        }
        for stock, score in scored_stocks[:10]
    }
    
    # 避開清單（後10）
    recommendations["avoid_list"] = {
        stock: {
            "score": score,
            "reason": "Low expected return" if predictions[stock]["multi_horizon_returns"].get("7d", {}).get("expected_return", 0) < 0 else "High risk"
        }
        for stock, score in scored_stocks[-10:]
    }
    
    return recommendations


def comprehensive_predict(model, dataset, chunk, steps: int = 7):
    """
    全面預測：提供多種預測標的
    """
    predictions = {}
    
    try:
        # 1. 獲取最新的歷史數據作為基準（优先 valid，若不存在则用 test）
        print("Preparing input data for predictions...")
        seg_name = "valid" if "valid" in getattr(dataset, "segments", {}) else "test"
        print(f"→ using segment `{seg_name}`")
        latest_data = dataset.prepare(seg_name, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)        
        if latest_data is None or len(latest_data) == 0:
            print("No training data available for predictions")
            return predictions
            
        # 檢查數據結構
        print(f"Data shape: {latest_data.shape}")
        print(f"Data index: {latest_data.index.names}")
        
        # 獲取可用的股票列表
        if hasattr(latest_data.index, 'get_level_values'):
            available_instruments = set(latest_data.index.get_level_values('instrument'))
        else:
            available_instruments = set(latest_data.index)
            
        print(f"Available instruments in data: {len(available_instruments)} stocks")
        print(f"Sample instruments: {list(available_instruments)[:10]}")
        print(f"Chunk instruments: {chunk[:5]}...")
        
        # 檢查有多少chunk中的股票在數據中
        valid_stocks = [stock for stock in chunk if stock in available_instruments]
        missing_stocks = [stock for stock in chunk if stock not in available_instruments]
        
        print(f"Valid stocks in chunk: {len(valid_stocks)}")
        print(f"Missing stocks in chunk: {len(missing_stocks)}")
        if missing_stocks:
            print(f"Missing stocks sample: {missing_stocks[:5]}")
        
        for stock in valid_stocks:  # 只處理有效的股票
            try:
                print(f"Processing stock: {stock}")
                
                # 安全地獲取股票數據
                try:
                    if hasattr(latest_data.index, 'get_level_values'):
                        # MultiIndex 情況
                        stock_data = latest_data.loc[latest_data.index.get_level_values('instrument') == stock]
                    else:
                        # 單層索引情況
                        stock_data = latest_data.loc[stock:stock] if stock in latest_data.index else pd.DataFrame()
                    
                    if len(stock_data) == 0:
                        print(f"No data rows for stock {stock}, skipping...")
                        continue
                        
                    # 取最近20天數據，如果不足20天就取全部
                    stock_data = stock_data.tail(min(20, len(stock_data)))
                    print(f"Got {len(stock_data)} data points for {stock}")
                    
                except Exception as e:
                    print(f"Error getting data for stock {stock}: {e}")
                    continue
                
                # 檢查必要的列是否存在
                if 'feature' not in stock_data.columns:
                    print(f"No feature data for stock {stock}, skipping...")
                    continue
                
                # 獲取最新特徵和標籤
                try:
                    latest_features = stock_data['feature'].iloc[-1]
                    latest_returns = stock_data['label'].values if 'label' in stock_data.columns else None
                    
                    # 確保 latest_features 是 numpy array
                    if hasattr(latest_features, 'values'):
                        latest_features = latest_features.values
                    elif isinstance(latest_features, list):
                        latest_features = np.array(latest_features)
                    
                    # 檢查特徵維度
                    if len(latest_features.shape) == 0 or latest_features.shape[0] == 0:
                        print(f"Invalid feature shape for {stock}: {latest_features.shape}")
                        continue
                        
                    print(f"Feature shape for {stock}: {latest_features.shape}")
                    
                except Exception as e:
                    print(f"Error extracting features for {stock}: {e}")
                    continue
                
                # 初始化預測結果
                stock_predictions = {
                    "basic_info": {
                        "symbol": stock,
                        "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                        "last_known_return": float(latest_returns[-1]) if latest_returns is not None and len(latest_returns) > 0 else None,
                        "data_points": len(stock_data),
                        "feature_dimension": len(latest_features)
                    },
                    "multi_horizon_returns": {},
                    "risk_metrics": {},
                    "technical_signals": {},
                    "trend_analysis": {},
                    "probability_distributions": {}
                }
                
                # 2. 多時間段收益率預測
                print(f"Predicting multi-horizon returns for {stock}...")
                for horizon in [1, 3, 5, 7]:
                    if horizon <= steps:
                        try:
                            predicted_returns = predict_multi_step_returns(model, latest_features, horizon)
                            if predicted_returns is not None and len(predicted_returns) > 0:
                                stock_predictions["multi_horizon_returns"][f"{horizon}d"] = {
                                    "expected_return": float(np.mean(predicted_returns)),
                                    "cumulative_return": float(np.sum(predicted_returns)),
                                    "daily_returns": [float(x) for x in predicted_returns]
                                }
                            else:
                                raise ValueError(f"Empty predictions for {horizon}d")
                        except Exception as e:
                            print(f"Error predicting {horizon}d returns for {stock}: {e}")
                            stock_predictions["multi_horizon_returns"][f"{horizon}d"] = {
                                "expected_return": 0.0,
                                "cumulative_return": 0.0,
                                "daily_returns": [0.0] * horizon,
                                "error": str(e)
                            }
                
                # 3. 風險指標預測（簡化版本）
                print(f"Calculating risk metrics for {stock}...")
                try:
                    returns_7d = predict_multi_step_returns(model, latest_features, 7)
                    if returns_7d is not None and len(returns_7d) > 0:
                        stock_predictions["risk_metrics"] = {
                            "volatility_7d": float(np.std(returns_7d)),
                            "expected_return_7d": float(np.mean(returns_7d)),
                            "min_return_7d": float(np.min(returns_7d)),
                            "max_return_7d": float(np.max(returns_7d))
                        }
                    else:
                        raise ValueError("Empty risk predictions")
                except Exception as e:
                    print(f"Error calculating risk metrics for {stock}: {e}")
                    stock_predictions["risk_metrics"] = {
                        "volatility_7d": 0.02,
                        "expected_return_7d": 0.0,
                        "min_return_7d": -0.05,
                        "max_return_7d": 0.05,
                        "error": str(e)
                    }
                
                # 4. 簡化的選股評分
                try:
                    expected_7d = stock_predictions["multi_horizon_returns"].get("7d", {}).get("expected_return", 0)
                    volatility = stock_predictions["risk_metrics"].get("volatility_7d", 0.02)
                    
                    # 簡單的風險調整收益評分
                    if volatility > 0:
                        risk_adjusted_return = expected_7d / volatility
                    else:
                        risk_adjusted_return = 0
                    
                    stock_predictions["selection_scores"] = {
                        "expected_return": expected_7d,
                        "volatility": volatility,
                        "risk_adjusted_return": risk_adjusted_return,
                        "composite_score": max(0, min(100, (risk_adjusted_return + 1) * 50))
                    }
                except Exception as e:
                    print(f"Error calculating selection scores for {stock}: {e}")
                    stock_predictions["selection_scores"] = {
                        "composite_score": 50.0,
                        "error": str(e)
                    }
                
                predictions[stock] = stock_predictions
                print(f"✓ Successfully processed {stock}")
                
            except Exception as e:
                print(f"Error processing stock {stock}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Error in comprehensive_predict: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"Successfully processed {len(predictions)} out of {len(chunk)} stocks")
    return predictions


def predict_multi_step_returns(model, initial_features, steps):
    """簡化版多步預測收益率"""
    try:
        returns = []
        current_features = initial_features.copy()
        
        for step in range(steps):
            # 確保特徵是正確的形狀
            if len(current_features.shape) == 1:
                feature_tensor = torch.from_numpy(current_features.reshape(1, -1)).float().to(model.device)
            else:
                feature_tensor = torch.from_numpy(current_features).float().to(model.device)
            
            with torch.no_grad():
                pred = model.model(feature_tensor)
                if pred.dim() == 2 and pred.size(1) >= 2:
                    predicted_return = float(pred[0, 0].cpu())
                elif pred.dim() == 2 and pred.size(1) == 1:
                    predicted_return = float(pred[0, 0].cpu())
                else:
                    predicted_return = float(pred[0].cpu() if pred.dim() == 1 else pred.cpu())
            
            returns.append(predicted_return)
            
            # 簡化的特徵更新（不更新，保持原始特徵）
            # 這樣避免了特徵更新可能導致的錯誤
        
        return np.array(returns)
        
    except Exception as e:
        print(f"Error in predict_multi_step_returns: {e}")
        return np.array([0.0] * steps)  # 返回零收益作為默認值


def calculate_max_drawdown(returns):
    """計算最大回撤"""
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))


def calculate_sharpe_estimate(returns):
    """估算夏普比率"""
    if np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # 年化


def generate_technical_signals(historical_data, predicted_returns):
    """生成技術指標信號"""
    signals = {}
    
    if len(historical_data) >= 5:
        recent_returns = historical_data['label'].values[-5:] if 'label' in historical_data.columns else []
        
        # 動量信號
        signals["momentum_5d"] = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
        signals["momentum_signal"] = "BUY" if signals["momentum_5d"] > 0.01 else "SELL" if signals["momentum_5d"] < -0.01 else "HOLD"
        
        # 預測動量
        signals["predicted_momentum_7d"] = float(np.mean(predicted_returns))
        signals["predicted_signal"] = "BUY" if signals["predicted_momentum_7d"] > 0.02 else "SELL" if signals["predicted_momentum_7d"] < -0.02 else "HOLD"
        
        # 反轉信號
        signals["mean_reversion_signal"] = "BUY" if signals["momentum_5d"] < -0.03 else "SELL" if signals["momentum_5d"] > 0.03 else "HOLD"
    
    return signals


def analyze_trends(predicted_returns, historical_returns):
    """趨勢分析"""
    analysis = {}
    
    # 預測趨勢
    analysis["predicted_trend"] = "UPTREND" if np.mean(predicted_returns) > 0.005 else "DOWNTREND" if np.mean(predicted_returns) < -0.005 else "SIDEWAYS"
    analysis["trend_strength"] = float(abs(np.mean(predicted_returns)))
    analysis["trend_consistency"] = float(np.mean([1 if r > 0 else 0 for r in predicted_returns]))
    
    # 趨勢變化
    if historical_returns is not None and len(historical_returns) >= 5:
        hist_trend = np.mean(historical_returns[-5:])
        pred_trend = np.mean(predicted_returns)
        analysis["trend_change"] = "ACCELERATING" if pred_trend > hist_trend else "DECELERATING" if pred_trend < hist_trend else "STABLE"
    
    return analysis


def calculate_outperform_probability(returns):
    """計算跑贏市場的概率（假設市場收益率為0.001）"""
    market_return = 0.001  # 假設日均市場收益率
    return float(np.mean(returns > market_return))
        
        
"""
        chunk_comprehensive = comprehensive_predict(model, dataset, chunk, steps)
        all_comprehensive_predictions.update(chunk_comprehensive)
"""
"""
    recommendations = generate_stock_recommendations(all_comprehensive_predictions)
    # 保存結果
    final_result = {
        "comprehensive_predictions": all_comprehensive_predictions,
        "stock_recommendations": recommendations,
        "training_history": evals_result,
        "metadata": {
            "prediction_date": datetime.now().isoformat(),
            "prediction_horizon_days": steps,
            "total_stocks_analyzed": len(all_comprehensive_predictions)
        }
    }

    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"✓ Comprehensive predictions saved to {out_json}")
    
    return final_result
"""

def calculate_selection_scores(predictions):
    """計算選股評分"""
    scores = {}
    
    # 收益評分 (0-100)
    expected_7d = predictions["multi_horizon_returns"].get("7d", {}).get("expected_return", 0)
    scores["return_score"] = max(0, min(100, (expected_7d + 0.05) * 1000))
    
    # 風險評分 (0-100, 越低越好)
    volatility = predictions["risk_metrics"].get("volatility_7d", 0.1)
    scores["risk_score"] = max(0, min(100, 100 - volatility * 1000))
    
    # 夏普評分
    sharpe = predictions["risk_metrics"].get("sharpe_estimate", 0)
    scores["sharpe_score"] = max(0, min(100, (sharpe + 2) * 25))
    
    # 概率評分
    prob_positive = predictions["probability_distributions"].get("prob_positive_7d", 0.5)
    scores["probability_score"] = prob_positive * 100
    
    # 綜合評分
    scores["composite_score"] = (
        scores["return_score"] * 0.3 +
        scores["risk_score"] * 0.2 +
        scores["sharpe_score"] * 0.3 +
        scores["probability_score"] * 0.2
    )
    
    return scores
