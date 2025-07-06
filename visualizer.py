# utils/visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from typing import Dict, Any
import statsmodels.api as sm
from pathlib import Path
import sys

# 导入时使用相对路径
from .kpi_calculator import calculate_kpis

# --- 全局设置matplotlib以支持中文 (最终解决方案) ---
try:
    # 构建字体文件的绝对路径
    # Path(__file__) 是当前脚本 visualizer.py 的路径
    # .resolve() 获取绝对路径
    # .parent.parent 获取项目根目录 (csi300_volatility_trader/)
    font_path = Path(__file__).resolve().parent.parent / 'fonts' / 'NotoSansCJKsc-Regular.otf'

    if font_path.exists():
        # 将字体文件添加到matplotlib的字体管理器
        fm.fontManager.addfont(font_path)
        # 从字体文件创建FontProperties对象，并获取字体名称
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        # 设置全局字体
        plt.rcParams['font.family'] = 'sans-serif' # 首先设置字体族
        plt.rcParams['font.sans-serif'] = [font_name] # 指定我们新添加的字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("✅ 成功加载并设置自定义中文字体 'Noto Sans CJK SC'。")
    else:
        print(f"⚠️ 警告: 字体文件未找到于 {font_path}。图表中的中文可能无法正常显示。")
        print("请按照README指导下载字体文件。")

except Exception as e:
    print(f"⚠️ 警告: 加载自定义字体时出错: {e}。将使用系统默认字体。")
# ----------------------------------------------------


def _ensure_dir_exists(dir_path: Path):
    """确保目录存在，如果不存在则创建。"""
    if not dir_path.exists():
        print(f"创建目录: {dir_path}")
        dir_path.mkdir(parents=True)

def plot_model_diagnostics(model_result: Any, output_config: Dict):
    """对拟合好的模型进行可视化诊断，包括残差分析。"""
    print("\n--- 模型可视化诊断 ---")
    std_resid = model_result.std_resid

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("模型标准化残差诊断图", fontsize=16)

    axes[0].plot(std_resid, linewidth=0.8)
    axes[0].set_title('Standardized Residuals Time Series')
    axes[0].grid(True, alpha=0.4)

    axes[1].hist(std_resid, bins=50, density=True, alpha=0.7)
    axes[1].set_title('Histogram of Standardized Residuals')
    axes[1].grid(True, alpha=0.4)
    
    sm.qqplot(std_resid, line='s', ax=axes[2], lw=1)
    axes[2].set_title('Q-Q Plot of Standardized Residuals')
    axes[2].grid(True, alpha=0.4)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    diag_dir = Path(output_config['diagnostics_dir'])
    _ensure_dir_exists(diag_dir)
    
    fig_path = diag_dir / "model_residuals_analysis.png"
    plt.savefig(fig_path)
    print(f"模型诊断图已保存至: {fig_path}")
    
    plt.close(fig)

def plot_backtest_results(results: pd.DataFrame, output_config: Dict):
    """可视化滚动回测的结果，对比预测波动率与实现波动率。"""
    fig = plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['Realized_Volatility'], label='已实现波动率 (|日收益率|)', color='grey', alpha=0.7, linewidth=1)
    plt.plot(results.index, results['Predicted_Volatility'], label='GARCH预测波动率', color='blue', alpha=0.8, linewidth=1.5)
    plt.title('样本外滚动预测 vs. 已实现波动率', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('波动率 (x100)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    results_dir = Path(output_config['results_dir'])
    _ensure_dir_exists(results_dir)
        
    fig_path = results_dir / "rolling_forecast_vs_realized.png"
    plt.savefig(fig_path)
    print(f"回测结果图已保存至: {fig_path}")
    
    plt.close(fig)

    rmse = np.sqrt(np.mean((results['Predicted_Volatility'] - results['Realized_Volatility'])**2))
    print(f"\n预测波动率与实现波动率的均方根误差 (RMSE): {rmse:.4f}")

def plot_strategy_performance(performance_df: pd.DataFrame, config: Dict):
    """可视化策略表现并打印关键绩效指标 (KPIs)。"""
    
    output_config = config['output_settings']
    strategy_config = config['strategy_settings']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('策略表现分析', fontsize=16)

    axes[0].plot(performance_df.index, performance_df['Strategy_Equity'], label='波动率目标策略')
    axes[0].plot(performance_df.index, performance_df['Benchmark_Equity'], label='买入持有基准', color='grey', alpha=0.8)
    axes[0].set_title('策略净值曲线 vs. 基准')
    axes[0].set_ylabel('净值 (对数坐标)')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].plot(performance_df.index, performance_df['Position'], label='策略仓位', color='orange', alpha=0.8)
    axes[1].set_title('策略仓位变化')
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('仓位/杠杆')
    axes[1].grid(True, alpha=0.4)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    results_dir = Path(output_config['results_dir'])
    _ensure_dir_exists(results_dir)
        
    fig_path = results_dir / "strategy_performance.png"
    plt.savefig(fig_path)
    print(f"\n策略表现图已保存至: {fig_path}")

    plt.close(fig)

    risk_free_rate = strategy_config.get('risk_free_rate', 0.0)
    strategy_kpis = calculate_kpis(performance_df['Strategy_Returns'], risk_free_rate)
    benchmark_kpis = calculate_kpis(performance_df['Benchmark_Returns'], risk_free_rate)

    kpi_df = pd.DataFrame([strategy_kpis, benchmark_kpis], index=['策略 (Strategy)', '基准 (Benchmark)'])
    print("\n--- 关键绩效指标 (KPIs) 对比 ---")
    print(kpi_df.to_string())