# core/backtester.py
import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm
from typing import Dict

def perform_rolling_forecast(returns: pd.Series, best_model_spec: Dict, config: Dict) -> pd.DataFrame:
    """
    执行滚动窗口预测，以评估模型的样本外预测能力。

    Args:
        returns (pd.Series): 完整的收益率时间序列。
        best_model_spec (Dict): 在模型选择阶段确定的最优模型的参数字典。
        config (Dict): 包含所有配置的字典。

    Returns:
        pd.DataFrame: 包含预测波动率和实际收益率的回测结果。
    """
    print("\n--- 步骤3: 滚动预测回测 ---")
    split_point = int(len(returns) * config['model_settings']['train_split_ratio'])
    test_size = len(returns) - split_point
    
    predictions = pd.Series(index=returns.index[split_point:])
    
    model_config = config['model_settings']
    # 从最优模型规格中提取参数
    best_model_params = best_model_spec.get('params', {})

    for i in tqdm(range(test_size), desc=f"滚动预测 ({best_model_spec.get('name')})", unit="天"):
        train_data = returns[0:split_point+i]
        
        # 使用 ** (字典解包) 来构建模型，确保参数正确
        model = arch_model(
            train_data,
            mean='ARX', lags=model_config['ar_lags'],
            dist=model_config['distribution'],
            **best_model_params
        )
        result = model.fit(disp='off')
        
        forecast = result.forecast(horizon=1, reindex=False)
        predictions.iloc[i] = np.sqrt(forecast.variance.iloc[0, 0])
        
    print("滚动预测完成。")
    
    backtest_results = pd.DataFrame({
        'Predicted_Volatility': predictions,
        'Actual_Return': returns.iloc[split_point:]
    })
    backtest_results['Realized_Volatility'] = backtest_results['Actual_Return'].abs()
    
    return backtest_results

# ... simulate_strategy 函数保持不变 ...
def simulate_strategy(backtest_results: pd.DataFrame, strategy_config: Dict) -> pd.DataFrame:
    """
    基于波动率预测，模拟波动率目标策略的表现。

    Args:
        backtest_results (pd.DataFrame): 滚动回测的结果。
        strategy_config (Dict): 包含策略参数的配置字典。

    Returns:
        pd.DataFrame: 包含策略和基准的详细表现数据。
    """
    print("\n--- 步骤4: 模拟波动率目标策略 ---")
    
    daily_vol_forecast = backtest_results['Predicted_Volatility'] / 100
    annualized_forecast_vol = daily_vol_forecast * np.sqrt(252)
    
    positions = strategy_config['target_annual_vol'] / annualized_forecast_vol
    positions = positions.clip(0, strategy_config['max_leverage'])
    
    strategy_returns = (positions.shift(1) * (backtest_results['Actual_Return'] / 100)).dropna()
    
    benchmark_returns = (backtest_results['Actual_Return'] / 100).loc[strategy_returns.index]
    
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    
    performance_df = pd.DataFrame({
        'Strategy_Returns': strategy_returns,
        'Benchmark_Returns': benchmark_returns,
        'Strategy_Equity': strategy_equity,
        'Benchmark_Equity': benchmark_equity,
        'Position': positions.shift(1).dropna()
    })
    
    print("策略模拟完成。")
    return performance_df