import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import sys
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置stdout编码为utf-8
sys.stdout.reconfigure(encoding='utf-8')

# 读取处理后的数据
df = pd.read_csv('../data/CHN_sample_data_processed.csv')

print("=" * 80)
print("LASSO回归模型分析 - 特征选择与收益率预测")
print("=" * 80)
print(f"原始数据样本量: {len(df)} 条观测值")

# 检查并处理无穷值和缺失值
for col in df.columns:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# 删除包含缺失值的行
df_clean = df.dropna()
print(f"清理后样本量: {len(df_clean)} 条观测值")

# 定义因变量和自变量
y = df_clean['y'].values

# 获取所有自变量（排除Dates, stkcd, y, 以及完全相关的变量）
exclude_vars = ['Dates', 'stkcd', 'y', 'OCFP', 'mv', 'size']
feature_vars = [col for col in df_clean.columns if col not in exclude_vars]

X = df_clean[feature_vars].values

print(f"\n特征变量数量: {len(feature_vars)} 个")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\n训练集样本量: {len(X_train)}")
print(f"测试集样本量: {len(X_test)}")

# ==================== 1. LASSO交叉验证选择最优alpha ====================
print("\n" + "=" * 80)
print("1. LASSO交叉验证选择最优alpha")
print("=" * 80)

# 使用LassoCV进行交叉验证
print("正在进行5折交叉验证选择最优alpha...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=2000, n_alphas=50)
lasso_cv.fit(X_train, y_train)

best_alpha = lasso_cv.alpha_
print(f"\n最优alpha值: {best_alpha:.6f}")



# ==================== 2. 使用最优alpha训练LASSO模型 ====================
print("\n" + "=" * 80)
print("2. LASSO模型训练与评估")
print("=" * 80)

# 使用最优alpha训练LASSO模型
lasso = Lasso(alpha=best_alpha, max_iter=2000, random_state=42)
lasso.fit(X_train, y_train)

# 预测
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# 计算评估指标
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = lasso.score(X_train, y_train)
test_r2 = lasso.score(X_test, y_test)

print(f"\n模型评估指标:")
print(f"{'指标':<20} {'训练集':>15} {'测试集':>15}")
print("-" * 50)
print(f"{'MSE':<20} {train_mse:>15.6f} {test_mse:>15.6f}")
print(f"{'RMSE':<20} {train_rmse:>15.6f} {test_rmse:>15.6f}")
print(f"{'R-squared':<20} {train_r2:>15.4f} {test_r2:>15.4f}")

# 统计被选中的特征数量
selected_features = np.sum(lasso.coef_ != 0)
print(f"\n被选中的特征数量: {selected_features} / {len(feature_vars)}")
print(f"特征选择比例: {selected_features/len(feature_vars)*100:.2f}%")

# ==================== 3. 特征重要性分析（LASSO系数）====================
print("\n" + "=" * 80)
print("3. 特征重要性分析（LASSO系数）")
print("=" * 80)

# 获取系数（只显示非零系数）
coef_df = pd.DataFrame({
    'Feature': feature_vars,
    'Coefficient': lasso.coef_,
    'Abs_Coefficient': np.abs(lasso.coef_)
})

# 筛选非零系数
nonzero_coef_df = coef_df[coef_df['Coefficient'] != 0].copy()
nonzero_coef_df = nonzero_coef_df.sort_values('Abs_Coefficient', ascending=False)

print(f"\n被选中的特征（按|系数|排序）:")
print("-" * 60)
print(f"{'排名':<6} {'特征':<15} {'系数':>12} {'|系数|':>12}")
print("-" * 60)

for i, (_, row) in enumerate(nonzero_coef_df.iterrows(), 1):
    print(f"{i:<6} {row['Feature']:<15} {row['Coefficient']:>12.4f} {row['Abs_Coefficient']:>12.4f}")

# 保存所有系数到CSV
coef_df.to_csv('../data/lasso_coefficients.csv', index=False)
print(f"\n所有特征系数已保存到 data/lasso_coefficients.csv")

# 保存被选中的特征
nonzero_coef_df.to_csv('../data/lasso_selected_features.csv', index=False)
print(f"被选中的特征已保存到 data/lasso_selected_features.csv")



# ==================== 5. 与OLS和Ridge对比 ====================
print("\n" + "=" * 80)
print("5. LASSO vs OLS vs Ridge 对比")
print("=" * 80)

from sklearn.linear_model import Ridge, LinearRegression

# 训练OLS模型
ols = LinearRegression()
ols.fit(X_train, y_train)
y_test_pred_ols = ols.predict(X_test)

# 训练Ridge模型
ridge = Ridge(alpha=11.514, random_state=42)  # 使用之前Ridge的最优alpha
ridge.fit(X_train, y_train)
y_test_pred_ridge = ridge.predict(X_test)

# 计算各模型评估指标
test_mse_ols = np.mean((y_test - y_test_pred_ols) ** 2)
test_rmse_ols = np.sqrt(test_mse_ols)
test_r2_ols = ols.score(X_test, y_test)

test_mse_ridge = np.mean((y_test - y_test_pred_ridge) ** 2)
test_rmse_ridge = np.sqrt(test_mse_ridge)
test_r2_ridge = ridge.score(X_test, y_test)

# 统计各模型的特征数量
ols_features = len(feature_vars)
ridge_features = len(feature_vars)
lasso_features = selected_features

comparison_data = {
    'Model': ['OLS', 'Ridge', 'LASSO'],
    'Test R²': [test_r2_ols, test_r2_ridge, test_r2],
    'Test RMSE': [test_rmse_ols, test_rmse_ridge, test_rmse],
    'Test MSE': [test_mse_ols, test_mse_ridge, test_mse],
    'Features Used': [ols_features, ridge_features, lasso_features],
    'Regularization': ['None', f'L2 (α=11.51)', f'L1 (α={best_alpha:.4f})']
}

comparison_df = pd.DataFrame(comparison_data)
print("\n三模型对比:")
print(comparison_df.to_string(index=False))

# ==================== 6. 可视化对比 ====================
print("\n" + "=" * 80)
print("6. LASSO vs OLS vs Ridge 可视化对比")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1: 模型性能对比 - 条形图
metrics = ['R²', 'RMSE']
ols_metrics = [test_r2_ols, test_rmse_ols]
ridge_metrics = [test_r2_ridge, test_rmse_ridge]
lasso_metrics = [test_r2, test_rmse]

x_pos = np.arange(len(metrics))
width = 0.25

axes[0, 0].bar(x_pos - width, ols_metrics, width, label='OLS', color='coral', alpha=0.8)
axes[0, 0].bar(x_pos, ridge_metrics, width, label='Ridge', color='steelblue', alpha=0.8)
axes[0, 0].bar(x_pos + width, lasso_metrics, width, label='LASSO', color='forestgreen', alpha=0.8)

axes[0, 0].set_ylabel('Value', fontsize=11)
axes[0, 0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(metrics)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (ols_val, ridge_val, lasso_val) in enumerate(zip(ols_metrics, ridge_metrics, lasso_metrics)):
    axes[0, 0].text(i - width, ols_val, f'{ols_val:.4f}', ha='center', va='bottom', fontsize=8)
    axes[0, 0].text(i, ridge_val, f'{ridge_val:.4f}', ha='center', va='bottom', fontsize=8)
    axes[0, 0].text(i + width, lasso_val, f'{lasso_val:.4f}', ha='center', va='bottom', fontsize=8)

# 图2: 特征数量对比
axes[0, 1].bar(['OLS', 'Ridge', 'LASSO'], [ols_features, ridge_features, lasso_features], 
               color=['coral', 'steelblue', 'forestgreen'], alpha=0.8)
axes[0, 1].set_ylabel('Number of Features', fontsize=11)
axes[0, 1].set_title('Feature Selection Comparison', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, v in enumerate([ols_features, ridge_features, lasso_features]):
    axes[0, 1].text(i, v, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

# 图3: 预测效果对比 - 测试集散点图
axes[1, 0].scatter(y_test, y_test_pred_ols, alpha=0.3, s=10, c='coral', label=f'OLS (R²={test_r2_ols:.4f})')
axes[1, 0].scatter(y_test, y_test_pred_ridge, alpha=0.3, s=10, c='steelblue', label=f'Ridge (R²={test_r2_ridge:.4f})')
axes[1, 0].scatter(y_test, y_test_pred, alpha=0.3, s=10, c='forestgreen', label=f'LASSO (R²={test_r2:.4f})')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Return', fontsize=11)
axes[1, 0].set_ylabel('Predicted Return', fontsize=11)
axes[1, 0].set_title('Prediction Comparison (Test Set)', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 图4: LASSO被选中的特征系数条形图
if len(nonzero_coef_df) > 0:
    top_n = min(15, len(nonzero_coef_df))
    top_features = nonzero_coef_df.head(top_n)
    colors = ['forestgreen' if c > 0 else 'darkred' for c in top_features['Coefficient']]
    
    axes[1, 1].barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.8)
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['Feature'], fontsize=9)
    axes[1, 1].set_xlabel('Coefficient', fontsize=11)
    axes[1, 1].set_title(f'LASSO Selected Features (Top {top_n})', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', lw=1)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.suptitle('LASSO vs OLS vs Ridge: Comprehensive Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../lasso_vs_ols_ridge_comparison.png', dpi=300, bbox_inches='tight')
print("LASSO vs OLS vs Ridge 对比图已保存到 lasso_vs_ols_ridge_comparison.png")
plt.close()

# ==================== 7. 保存预测结果 ====================
print("\n" + "=" * 80)
print("7. 保存预测结果")
print("=" * 80)

# 创建预测结果DataFrame
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'LASSO_Predicted': y_test_pred,
    'Ridge_Predicted': y_test_pred_ridge,
    'OLS_Predicted': y_test_pred_ols,
    'LASSO_Residual': y_test - y_test_pred,
    'Ridge_Residual': y_test - y_test_pred_ridge,
    'OLS_Residual': y_test - y_test_pred_ols
})

predictions_df.to_csv('../data/lasso_predictions.csv', index=False)
print(f"测试集预测结果已保存到 data/lasso_predictions.csv")

# 保存模型参数
model_info = pd.DataFrame({
    'Parameter': ['Best Alpha', 'Train R2', 'Test R2', 'Train RMSE', 'Test RMSE', 
                  'Selected Features', 'Total Features', 'Selection Ratio (%)'],
    'Value': [best_alpha, train_r2, test_r2, train_rmse, test_rmse, 
              selected_features, len(feature_vars), selected_features/len(feature_vars)*100]
})
model_info.to_csv('../data/lasso_model_info.csv', index=False)
print(f"模型信息已保存到 data/lasso_model_info.csv")

# 保存三模型对比
comparison_df.to_csv('../data/model_comparison_ols_ridge_lasso.csv', index=False)
print(f"三模型对比已保存到 data/model_comparison_ols_ridge_lasso.csv")

print("\n" + "=" * 80)
print("LASSO回归分析完成!")
print("=" * 80)
print("\n输出文件:")
print("  1. lasso_vs_ols_ridge_comparison.png - LASSO vs OLS vs Ridge对比图")
print("  2. data/lasso_coefficients.csv - 所有特征系数")
print("  3. data/lasso_selected_features.csv - 被选中的特征")
print("  4. data/lasso_predictions.csv - 测试集预测结果")
print("  5. data/lasso_model_info.csv - 模型参数信息")
print("  6. data/model_comparison_ols_ridge_lasso.csv - 三模型对比")
