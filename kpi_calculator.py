# utils/kpi_calculator.py
import pandas as pd
import numpy as np
from typing import Dict

def calculate_kpis(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, str]:
    """
    根据日收益率序列计算一系列关键绩效指标 (KPIs)。

    Args:
        returns (pd.Series): 策略或基准的日收益率序列。
        risk_free_rate (float): 年化无风险利率，用于计算夏普比率。

    Returns:
        Dict[str, str]: 包含格式化后KPIs的字典。
    """
    if returns.empty or returns.isnull().all():
        return {
            "累计收益率 (Total Return)": "N/A", "年化收益率 (Annual Return)": "N/A",
            "年化波动率 (Annual Volatility)": "N/A", "夏普比率 (Sharpe Ratio)": "N/A",
            "最大回撤 (Max Drawdown)": "N/A"
        }

    days = len(returns)
    trading_days_per_year = 252
    
    # 累积净值曲线
    equity_curve = (1 + returns).cumprod()
    # 总收益率
    total_return = equity_curve.iloc[-1] - 1
    
    # 年化收益率
    annual_return = (1 + total_return)**(trading_days_per_year / days) - 1
    
    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(trading_days_per_year)
    
    # 夏普比率
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    
    # 最大回撤
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()
    
    return {
        "累计收益率 (Total Return)": f"{total_return:.2%}",
        "年化收益率 (Annual Return)": f"{annual_return:.2%}",
        "年化波动率 (Annual Volatility)": f"{annual_volatility:.2%}",
        "夏普比率 (Sharpe Ratio)": f"{sharpe_ratio:.2f}",
        "最大回撤 (Max Drawdown)": f"{max_drawdown:.2%}"
    }