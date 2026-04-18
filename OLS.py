import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 读取处理后的数据
df = pd.read_csv('../data/CHN_sample_data_processed.csv')

print("=" * 80)
print("OLS回归分析")
print("=" * 80)
print(f"数据样本量: {len(df)} 条观测值")

# 检查并处理无穷值和缺失值
print("\n检查数据质量...")
for col in df.columns:
    inf_count = np.isinf(df[col]).sum()
    nan_count = df[col].isna().sum()
    if inf_count > 0:
        print(f"  {col}: {inf_count} 个无穷值")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    if nan_count > 0:
        print(f"  {col}: {nan_count} 个缺失值")

# 删除包含缺失值的行
df_clean = df.dropna()
print(f"清理后样本量: {len(df_clean)} 条观测值")

# 定义因变量
y = df_clean['y']

# 获取所有自变量（排除Dates, stkcd, y）
all_vars = [col for col in df_clean.columns if col not in ['Dates', 'stkcd', 'y']]

print(f"\n可用变量数: {len(all_vars)} 个")

# ==================== 模型A：基础模型 ====================
print("\n" + "=" * 80)
print("模型A：基础模型（CFP, OCFP, Lsize, BM）")
print("=" * 80)

model_a_vars = ['CFP', 'OCFP', 'Lsize', 'BM']
# 检查变量是否存在
model_a_vars = [v for v in model_a_vars if v in df_clean.columns]

print(f"模型A包含变量: {model_a_vars}")

X_a = df_clean[model_a_vars]
X_a = sm.add_constant(X_a)  # 添加常数项

model_a = sm.OLS(y, X_a).fit()

print("\n回归结果:")
print("-" * 80)
print(model_a.summary().tables[1])
print("-" * 80)
print(f"R-squared: {model_a.rsquared:.4f}")
print(f"Adjusted R-squared: {model_a.rsquared_adj:.4f}")
print(f"F-statistic: {model_a.fvalue:.4f}")
print(f"Prob (F-statistic): {model_a.f_pvalue:.4e}")

# 模型A结果说明
print("\n" + "=" * 80)
print("模型A结果说明:")
print("=" * 80)

for var in model_a_vars:
    coef = model_a.params[var]
    pvalue = model_a.pvalues[var]
    significance = "显著" if pvalue < 0.05 else "不显著"
    direction = "正" if coef > 0 else "负"
    
    if pvalue < 0.01:
        sig_level = "在1%水平"
    elif pvalue < 0.05:
        sig_level = "在5%水平"
    else:
        sig_level = ""
    
    print(f"{var}对y呈现{direction}相关，系数为{coef:.4f}，{sig_level}{significance}(p={pvalue:.4f})")

# ==================== 模型B：全变量模型 ====================
print("\n" + "=" * 80)
print("模型B：全变量模型（所有自变量）")
print("=" * 80)

# 排除完全相关的变量（OCFP与CFP完全相关，size与mv完全相关）
exclude_vars = ['OCFP', 'mv', 'size']  # 排除与CFP、Lsize、size完全相关的变量
model_b_vars = [v for v in all_vars if v not in exclude_vars]

print(f"参与回归的变量数: {len(model_b_vars)} 个")

X_b = df_clean[model_b_vars]
X_b = sm.add_constant(X_b)  # 添加常数项

model_b = sm.OLS(y, X_b).fit()

print("\n回归结果（显著变量）:")
print("-" * 80)
print(f"{'变量':<15} {'系数':>12} {'标准误':>12} {'t值':>10} {'p值':>12} {'显著性':>8}")
print("-" * 80)

significant_vars = []
for var in model_b_vars:
    coef = model_b.params[var]
    std_err = model_b.bse[var]
    t_val = model_b.tvalues[var]
    pvalue = model_b.pvalues[var]
    
    if pvalue < 0.05:
        significance = "***" if pvalue < 0.01 else "**"
        significant_vars.append((var, coef, pvalue))
        print(f"{var:<15} {coef:>12.4f} {std_err:>12.4f} {t_val:>10.2f} {pvalue:>12.4f} {significance:>8}")

print("-" * 80)
print(f"R-squared: {model_b.rsquared:.4f}")
print(f"Adjusted R-squared: {model_b.rsquared_adj:.4f}")
print(f"F-statistic: {model_b.fvalue:.4f}")
print(f"Prob (F-statistic): {model_b.f_pvalue:.4e}")

# 模型B结果说明
print("\n" + "=" * 80)
print("模型B结果说明:")
print("=" * 80)

if significant_vars:
    print(f"\n共有 {len(significant_vars)} 个变量对收益率(y)有显著影响:\n")
    for var, coef, pvalue in significant_vars:
        direction = "正" if coef > 0 else "负"
        if pvalue < 0.01:
            sig_level = "在1%水平"
        else:
            sig_level = "在5%水平"
        print(f"{var}对y呈现{direction}相关，系数为{coef:.4f}，{sig_level}显著(p={pvalue:.4f})")
else:
    print("未发现对y有显著影响的变量")

# 其他发现
print("\n" + "=" * 80)
print("其他发现:")
print("=" * 80)

# 找出系数最大的变量（绝对值）
coef_abs = model_b.params.drop('const').abs().sort_values(ascending=False)
top_vars = coef_abs.head(5)
print("\n影响最大的5个变量（按系数绝对值）:")
for i, (var, abs_coef) in enumerate(top_vars.items(), 1):
    actual_coef = model_b.params[var]
    pval = model_b.pvalues[var]
    sig = "显著" if pval < 0.05 else "不显著"
    print(f"{i}. {var}: 系数={actual_coef:.4f}, |系数|={abs_coef:.4f}, {sig}")

# 模型比较
print("\n" + "=" * 80)
print("模型比较:")
print("=" * 80)
print(f"模型A (4个变量): R² = {model_a.rsquared:.4f}, 调整R² = {model_a.rsquared_adj:.4f}")
print(f"模型B ({len(model_b_vars)}个变量): R² = {model_b.rsquared:.4f}, 调整R² = {model_b.rsquared_adj:.4f}")

if model_b.rsquared_adj > model_a.rsquared_adj:
    improvement = model_b.rsquared_adj - model_a.rsquared_adj
    print(f"模型B的解释力更强，调整R²提高了 {improvement:.4f}")
else:
    print("模型A更简洁，但模型B包含更多信息")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
