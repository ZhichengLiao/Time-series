# core/data_handler.py
import pandas as pd
import akshare as ak
from typing import Dict, Optional
import numpy as np

def fetch_data(config: Dict) -> Optional[pd.DataFrame]:
    """使用akshare获取并清理指数历史数据。"""
    print(f"\n--- 步骤1: 获取数据 ---")
    print(f"目标: {config['ticker']}, 时间范围: {config['start_date']} to {config['end_date']}")
    try:
        data_ak = ak.stock_zh_index_daily(symbol=config['ticker'])
        if data_ak.empty:
            raise ValueError("Akshare未返回任何数据。请检查ticker代码是否正确。")

        data_ak['date'] = pd.to_datetime(data_ak['date'])
        data_ak.set_index('date', inplace=True)
        
        data = data_ak.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)
        
        # 截取配置中指定的时间范围
        data = data.loc[config['start_date']:config['end_date']]
        
        print("数据获取成功！")
        return data
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

def prepare_features(data: pd.DataFrame) -> pd.Series:
    """计算对数收益率并进行缩放以便于模型拟合。"""
    # 计算对数收益率，这是金融时间序列建模的标准做法
    returns = np.log(data['Close']).diff().dropna()
    
    # 乘以100是为了提高数值稳定性，这是GARCH模型拟合时的常见技巧
    # 结果的单位不再是百分比，而是一个放大了100倍的数值
    print("特征工程完成：已计算对数收益率。")
    return returns * 100