# core/model_fitter.py
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, Tuple, Any

def fit_and_select_model(returns: pd.Series, model_config: Dict) -> Tuple[Any, Dict]:
    # ... 此函数保持不变 ...
    print("\n--- 步骤2: 模型选择与拟合 ---")
    best_model_result = None
    best_aic = np.inf
    best_model_spec = {}

    for model_spec in model_config['models_to_test']:
        model_name = model_spec['name']
        model_params = model_spec['params']
        
        print(f"正在拟合 {model_name} 模型...")
        
        try:
            model = arch_model(
                returns,
                mean='ARX',
                lags=model_config['ar_lags'],
                dist=model_config['distribution'],
                **model_params
            )
            result = model.fit(update_freq=0, disp='off')
            
            print(f"    {model_name} 模型 - AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")

            if result.aic < best_aic:
                best_aic = result.aic
                best_model_result = result
                best_model_spec = model_spec
        except Exception as e:
            print(f"    ❌ {model_name} 模型拟合失败: {e}")
            continue # 如果拟合失败，跳过这个模型
            
    if not best_model_spec:
        print("\n所有模型均未能成功拟合，无法继续。")
        return None, {}
        
    print(f"\n✅ 基于AIC的最优模型选择: {best_model_spec.get('name', 'N/A')} (AIC: {best_aic:.2f})")
    print(best_model_result.summary())
    return best_model_result, best_model_spec


def diagnose_model(model_result: Any):
    """
    对拟合好的模型进行Ljung-Box检验，以检查残差自相关性。
    (已更新，逻辑更稳健)
    """
    print("\n--- 模型统计诊断 ---")
    std_resid = model_result.std_resid

    # --- 新增：检查残差的有效性 ---
    if not np.isfinite(std_resid).all():
        print("    ⚠️ 警告: 模型标准化残差中包含NaN或无穷大值。")
        print("    这通常意味着模型存在数值不稳定性。Ljung-Box检验无法执行。")
        print("    建议：即使此模型的AIC较低，也应优先考虑使用其他诊断正常的模型。")
        return # 直接退出诊断函数

    # 对标准化残差的平方进行Ljung-Box检验
    lb_test = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
    print("\n对“平方标准化残差”进行Ljung-Box检验:")
    print(lb_test)
    
    # --- 修改：完善p值的判断逻辑 ---
    p_value = lb_test.iloc[0]['lb_pvalue']
    
    if pd.isna(p_value):
        # 这种情况现在被上面的有效性检查覆盖了，但保留以增加稳健性
        print(f"❌ 检验无法完成 (p-value is NaN): 检验的输入数据可能存在问题。")
    elif p_value > 0.05:
        print(f"✅ 检验通过 (p-value = {p_value:.3f} > 0.05): 残差的平方不存在显著的自相关性。模型成功捕捉了ARCH效应。")
    else:
        print(f"❌ 检验失败 (p-value = {p_value:.3f} <= 0.05): 残差的平方仍然存在自相关性，模型可能需要调整。")