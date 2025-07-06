# csi300_volatility_trader/main.py

import yaml
from typing import Dict
from pathlib import Path
import sys

from core.data_handler import fetch_data, prepare_features
from core.model_fitter import fit_and_select_model, diagnose_model
from core.backtester import perform_rolling_forecast, simulate_strategy
from utils.visualizer import plot_backtest_results, plot_strategy_performance, plot_model_diagnostics

def load_config(config_path: Path) -> Dict:
    """从YAML文件中加载配置。"""
    print(f"--- 步骤0: 加载配置文件 ---")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            print(f"配置文件 '{config_path}' 加载成功。")
            return config
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_path}' 未找到。请确保它存在于项目根目录中。")
        sys.exit(1)
    except Exception as e:
        print(f"加载或解析配置文件时出错: {e}")
        sys.exit(1)

def main():
    """主函数，负责协调整个量化分析流程。"""
    
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    config = load_config(config_path)
    
    # --- 步骤1: 数据处理 ---
    data = fetch_data(config['data_settings'])
    if data is None:
        print("数据获取失败，工作流程终止。")
        return
    returns = prepare_features(data)

    # --- 步骤2: 模型选择与诊断 ---
    split_idx = int(len(returns) * config['model_settings']['train_split_ratio'])
    train_returns = returns.iloc[:split_idx]
    
    # 函数现在返回最优模型的规格字典
    best_model_result, best_model_spec = fit_and_select_model(train_returns, config['model_settings'])
    
    diagnose_model(best_model_result)
    plot_model_diagnostics(best_model_result, config['output_settings'])
    
    # --- 步骤3: 回测与评估 ---
    # 将最优模型规格字典传递给回测函数
    backtest_results = perform_rolling_forecast(returns, best_model_spec, config)
    plot_backtest_results(backtest_results, config['output_settings'])

    # --- 步骤4: 策略模拟与评估 ---
    performance_df = simulate_strategy(backtest_results, config['strategy_settings'])
    plot_strategy_performance(performance_df, config)
    
    print("\n🎉 量化分析工作流全部完成！")

if __name__ == "__main__":
    main()