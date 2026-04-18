import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取处理后的数据
df = pd.read_csv('../data/CHN_sample_data_processed.csv')

print("=" * 80)
print("岭回归模型分析 - 与OLS回归对比")
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

# 数据标准化（手动实现）
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# 划分训练集和测试集
np.random.seed(42)
n_samples = len(X_scaled)
train_size = int(0.7 * n_samples)
indices = np.random.permutation(n_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\n训练集样本量: {len(X_train)}")
print(f"测试集样本量: {len(X_test)}")

# ==================== 1. 手动实现岭回归和交叉验证选择alpha ====================
print("\n" + "=" * 80)
print("1. 交叉验证选择最优正则化参数alpha")
print("=" * 80)

def ridge_regression(X, y, alpha):
    """手动实现岭回归"""
    n_features = X.shape[1]
    # 岭回归闭式解: (X'X + alpha*I)^(-1) X'y
    I = np.eye(n_features)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

def predict_ridge(X, beta):
    """岭回归预测"""
    return X @ beta

def ridge_cv(X, y, alphas, cv=5):
    """岭回归交叉验证"""
    n_samples = len(y)
    fold_size = n_samples // cv
    cv_scores = []
    
    for alpha in alphas:
        fold_mses = []
        for fold in range(cv):
            # 划分验证集
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < cv - 1 else n_samples
            
            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # 训练模型
            beta = ridge_regression(X_train_cv, y_train_cv, alpha)
            y_pred = predict_ridge(X_val_cv, beta)
            
            # 计算MSE
            mse = np.mean((y_val_cv - y_pred) ** 2)
            fold_mses.append(mse)
        
        cv_scores.append(np.mean(fold_mses))
    
    return cv_scores

# 定义alpha候选值
alphas = np.logspace(-4, 4, 50)

# 进行交叉验证
cv_scores = ridge_cv(X_train, y_train, alphas, cv=5)

# 找到最优alpha
best_alpha_idx = np.argmin(cv_scores)
best_alpha = alphas[best_alpha_idx]

print(f"\n最优alpha值: {best_alpha:.6f}")
print(f"对应交叉验证MSE: {cv_scores[best_alpha_idx]:.6f}")

# 绘制交叉验证得分随alpha变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(alphas, cv_scores, 'b-', linewidth=2)
plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'Best alpha = {best_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Mean Squared Error (CV)', fontsize=12)
plt.title('Ridge Regression: Cross-Validation for Alpha Selection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../ridge_alpha_selection.png', dpi=300, bbox_inches='tight')
print("交叉验证曲线已保存到 ridge_alpha_selection.png")
plt.close()

# ==================== 2. 使用最优alpha训练岭回归模型 ====================
print("\n" + "=" * 80)
print("2. 岭回归模型训练与评估")
print("=" * 80)

# 训练岭回归模型
beta_ridge = ridge_regression(X_train, y_train, best_alpha)

# 预测
y_train_pred = predict_ridge(X_train, beta_ridge)
y_test_pred = predict_ridge(X_test, beta_ridge)

# 计算评估指标
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"\n模型评估指标:")
print(f"{'指标':<20} {'训练集':>15} {'测试集':>15}")
print("-" * 50)
print(f"{'MSE':<20} {train_mse:>15.6f} {test_mse:>15.6f}")
print(f"{'RMSE':<20} {train_rmse:>15.6f} {test_rmse:>15.6f}")
print(f"{'R-squared':<20} {train_r2:>15.4f} {test_r2:>15.4f}")

# ==================== 3. 特征重要性分析（系数）====================
print("\n" + "=" * 80)
print("3. 特征重要性分析（岭回归系数）")
print("=" * 80)

# 获取系数
coef_df = pd.DataFrame({
    'Feature': feature_vars,
    'Coefficient': beta_ridge,
    'Abs_Coefficient': np.abs(beta_ridge)
})

# 按绝对值排序
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

print(f"\nTop 20 重要特征（按|系数|排序）:")
print("-" * 60)
print(f"{'排名':<6} {'特征':<15} {'系数':>12} {'|系数|':>12}")
print("-" * 60)

for i, (_, row) in enumerate(coef_df.head(20).iterrows(), 1):
    print(f"{i:<6} {row['Feature']:<15} {row['Coefficient']:>12.4f} {row['Abs_Coefficient']:>12.4f}")

# 保存所有系数到CSV
coef_df.to_csv('../data/ridge_coefficients.csv', index=False)
print(f"\n所有特征系数已保存到 data/ridge_coefficients.csv")

# ==================== 4. 系数压缩效果可视化 ====================
print("\n" + "=" * 80)
print("4. 系数压缩效果分析")
print("=" * 80)

# 比较不同alpha下的系数变化
alphas_to_compare = [0, 0.001, 0.01, 0.1, 1, 10, 100]
coef_comparison = pd.DataFrame(index=feature_vars)

for alpha in alphas_to_compare:
    beta = ridge_regression(X_scaled, y, alpha)
    coef_comparison[f'alpha={alpha}'] = beta

# 保存系数对比
coef_comparison.to_csv('../data/ridge_coefficient_comparison.csv')
print(f"不同alpha下的系数对比已保存到 data/ridge_coefficient_comparison.csv")

# 绘制系数随alpha变化的曲线（Top 10特征）
top_features = coef_df.head(10)['Feature'].values

plt.figure(figsize=(12, 8))
for feature in top_features:
    coef_values = coef_comparison.loc[feature, [f'alpha={a}' for a in alphas_to_compare]].values
    plt.plot(alphas_to_compare, coef_values, marker='o', label=feature, linewidth=2)

plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Ridge Regression Coefficients vs Alpha (Top 10 Features)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_alpha, color='red', linestyle='--', alpha=0.7, label=f'Best alpha = {best_alpha:.4f}')
plt.tight_layout()
plt.savefig('../ridge_coefficient_paths.png', dpi=300, bbox_inches='tight')
print("系数压缩路径图已保存到 ridge_coefficient_paths.png")
plt.close()

# ==================== 5. 与OLS模型对比 ====================
print("\n" + "=" * 80)
print("5. 岭回归 vs OLS 对比")
print("=" * 80)

# 训练OLS模型（alpha=0）
beta_ols = ridge_regression(X_train, y_train, 0)
y_test_pred_ols = predict_ridge(X_test, beta_ols)

# 计算OLS评估指标
test_mse_ols = np.mean((y_test - y_test_pred_ols) ** 2)
test_rmse_ols = np.sqrt(test_mse_ols)
test_r2_ols = 1 - np.sum((y_test - y_test_pred_ols) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

# 计算系数的标准差（多重共线性指标）
ols_coef_std = np.std(beta_ols)
ridge_coef_std = np.std(beta_ridge)

comparison_data = {
    'Metric': ['Test R-squared', 'Test RMSE', 'Coefficient Std Dev', 'Max |Coefficient|', 'Number of Features'],
    'OLS (alpha=0)': [test_r2_ols, test_rmse_ols, ols_coef_std, np.max(np.abs(beta_ols)), len(feature_vars)],
    f'Ridge (alpha={best_alpha:.4f})': [test_r2, test_rmse, ridge_coef_std, np.max(np.abs(beta_ridge)), len(feature_vars)]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n模型对比:")
print(comparison_df.to_string(index=False))

# ==================== 6. 岭回归 vs OLS 详细对比可视化（仅图2和图3）====================
print("\n" + "=" * 80)
print("6. 岭回归 vs OLS 详细对比可视化")
print("=" * 80)

# 创建对比图（仅保留图2和图3）
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 图2: 系数绝对值对比 - 条形图（Top 15）
top_15_features = coef_df.head(15)['Feature'].values
ols_coef_top15 = [beta_ols[feature_vars.index(f)] for f in top_15_features]
ridge_coef_top15 = [beta_ridge[feature_vars.index(f)] for f in top_15_features]

x_pos = np.arange(len(top_15_features))
width = 0.35
axes[0].barh(x_pos - width/2, np.abs(ols_coef_top15), width, label='OLS', color='coral', alpha=0.8)
axes[0].barh(x_pos + width/2, np.abs(ridge_coef_top15), width, label='Ridge', color='steelblue', alpha=0.8)
axes[0].set_yticks(x_pos)
axes[0].set_yticklabels(top_15_features, fontsize=10)
axes[0].set_xlabel('|Coefficient|', fontsize=12)
axes[0].set_title('Top 15 Features: |Coefficients| Comparison', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='x')

# 图3: 预测效果对比 - 测试集
axes[1].scatter(y_test, y_test_pred_ols, alpha=0.3, s=10, c='coral', label=f'OLS (R²={test_r2_ols:.4f})')
axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10, c='steelblue', label=f'Ridge (R²={test_r2:.4f})')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Return', fontsize=12)
axes[1].set_ylabel('Predicted Return', fontsize=12)
axes[1].set_title('Prediction Comparison (Test Set)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Ridge Regression vs OLS: Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../ridge_vs_ols_comparison.png', dpi=300, bbox_inches='tight')
print("岭回归 vs OLS 对比图已保存到 ridge_vs_ols_comparison.png（仅图2和图3）")
plt.close()

# ==================== 7. 预测效果详细可视化 ====================
print("\n" + "=" * 80)
print("7. 预测效果详细可视化")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 训练集 - OLS
axes[0, 0].scatter(y_train, predict_ridge(X_train, beta_ols), alpha=0.3, s=10, c='coral')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Return', fontsize=11)
axes[0, 0].set_ylabel('Predicted Return', fontsize=11)
train_r2_ols = 1 - np.sum((y_train - predict_ridge(X_train, beta_ols)) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
axes[0, 0].set_title(f'OLS - Training Set: R² = {train_r2_ols:.4f}', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 训练集 - Ridge
axes[0, 1].scatter(y_train, y_train_pred, alpha=0.3, s=10, c='steelblue')
axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Return', fontsize=11)
axes[0, 1].set_ylabel('Predicted Return', fontsize=11)
axes[0, 1].set_title(f'Ridge - Training Set: R² = {train_r2:.4f}', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 测试集 - OLS
axes[1, 0].scatter(y_test, y_test_pred_ols, alpha=0.3, s=10, c='coral')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Return', fontsize=11)
axes[1, 0].set_ylabel('Predicted Return', fontsize=11)
axes[1, 0].set_title(f'OLS - Test Set: R² = {test_r2_ols:.4f}', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 测试集 - Ridge
axes[1, 1].scatter(y_test, y_test_pred, alpha=0.3, s=10, c='steelblue')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Return', fontsize=11)
axes[1, 1].set_ylabel('Predicted Return', fontsize=11)
axes[1, 1].set_title(f'Ridge - Test Set: R² = {test_r2:.4f}', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Prediction Performance: OLS vs Ridge Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../ridge_prediction_performance.png', dpi=300, bbox_inches='tight')
print("预测效果对比图已保存到 ridge_prediction_performance.png")
plt.close()

# ==================== 8. 保存预测结果 ====================
print("\n" + "=" * 80)
print("8. 保存预测结果")
print("=" * 80)

# 创建预测结果DataFrame
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Ridge_Predicted': y_test_pred,
    'OLS_Predicted': y_test_pred_ols,
    'Ridge_Residual': y_test - y_test_pred,
    'OLS_Residual': y_test - y_test_pred_ols
})

predictions_df.to_csv('../data/ridge_predictions.csv', index=False)
print(f"测试集预测结果已保存到 data/ridge_predictions.csv")

# 保存模型参数
model_info = pd.DataFrame({
    'Parameter': ['Best Alpha', 'Train R2 (Ridge)', 'Test R2 (Ridge)', 'Train R2 (OLS)', 'Test R2 (OLS)', 
                  'Train RMSE (Ridge)', 'Test RMSE (Ridge)', 'Train RMSE (OLS)', 'Test RMSE (OLS)',
                  'Number of Features', 'OLS Coef Std', 'Ridge Coef Std'],
    'Value': [best_alpha, train_r2, test_r2, train_r2_ols, test_r2_ols,
              train_rmse, test_rmse, 
              np.sqrt(np.mean((y_train - predict_ridge(X_train, beta_ols)) ** 2)), test_rmse_ols,
              len(feature_vars), ols_coef_std, ridge_coef_std]
})
model_info.to_csv('../data/ridge_model_info.csv', index=False)
print(f"模型信息已保存到 data/ridge_model_info.csv")

# 保存系数对比
coef_comparison_final = pd.DataFrame({
    'Feature': feature_vars,
    'OLS_Coefficient': beta_ols,
    'Ridge_Coefficient': beta_ridge,
    'Difference': beta_ols - beta_ridge,
    'Shrinkage_Ratio': np.abs(beta_ridge) / (np.abs(beta_ols) + 1e-10)
})
coef_comparison_final.to_csv('../data/ridge_ols_coefficient_comparison.csv', index=False)
print(f"系数对比已保存到 data/ridge_ols_coefficient_comparison.csv")

print("\n" + "=" * 80)
print("岭回归分析完成!")
print("=" * 80)
print("\n输出文件:")
print("  1. ridge_alpha_selection.png - 交叉验证选择alpha曲线")
print("  2. ridge_coefficient_paths.png - 系数压缩路径图")
print("  3. ridge_prediction_performance.png - 预测效果对比图（4个子图）")
print("  4. ridge_vs_ols_comparison.png - 岭回归vsOLS综合对比图（6个子图）")
print("  5. data/ridge_coefficients.csv - 所有特征系数")
print("  6. data/ridge_coefficient_comparison.csv - 不同alpha系数对比")
print("  7. data/ridge_predictions.csv - 测试集预测结果")
print("  8. data/ridge_model_info.csv - 模型参数信息")
print("  9. data/ridge_ols_coefficient_comparison.csv - OLS与Ridge系数对比")
