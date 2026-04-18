import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取筛选后的数据
df = pd.read_csv('../data/CHN_sample_data_filtered.csv')
print("=" * 80)
print("数据描述性统计分析报告")
print("=" * 80)
print(f"\n数据样本量: {len(df)} 条观测值")
print(f"变量数量: {len(df.columns)} 个")

# 查看变量列表
print("\n" + "=" * 80)
print("变量列表:")
print("=" * 80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# 1. 进行1%缩尾处理 (Winsorization)
print("\n" + "=" * 80)
print("1%缩尾处理")
print("=" * 80)

# 复制数据框用于缩尾处理
df_winsorized = df.copy()

# 对数值型变量进行1%缩尾（排除Dates和stkcd）
numeric_cols = df_winsorized.select_dtypes(include=[np.number]).columns.tolist()
# 排除日期和股票代码
exclude_cols = ['Dates', 'stkcd']
process_cols = [col for col in numeric_cols if col not in exclude_cols]

winsorized_count = 0
for col in process_cols:
    lower = df_winsorized[col].quantile(0.01)
    upper = df_winsorized[col].quantile(0.99)
    original_min = df_winsorized[col].min()
    original_max = df_winsorized[col].max()
    
    df_winsorized[col] = df_winsorized[col].clip(lower=lower, upper=upper)
    
    if original_min != lower or original_max != upper:
        winsorized_count += 1
        print(f"{col}: [{original_min:.4f}, {original_max:.4f}] -> [{lower:.4f}, {upper:.4f}]")

print(f"\n共对 {winsorized_count} 个变量进行了1%缩尾处理")

# 2. 将size变量取对数，变为Lsize
print("\n" + "=" * 80)
print("Size变量对数转换")
print("=" * 80)

# 检查size变量是否存在
if 'size' in df_winsorized.columns:
    # 检查是否有非正值
    size_min = df_winsorized['size'].min()
    size_max = df_winsorized['size'].max()
    print(f"原始size变量范围: [{size_min:.4f}, {size_max:.4f}]")
    
    # 创建Lsize变量（自然对数）
    df_winsorized['Lsize'] = np.log(df_winsorized['size'])
    
    lsize_min = df_winsorized['Lsize'].min()
    lsize_max = df_winsorized['Lsize'].max()
    print(f"转换后Lsize变量范围: [{lsize_min:.4f}, {lsize_max:.4f}]")
    print(f"Lsize = ln(size)")
else:
    print("警告: 未找到size变量")

# 3. 描述性统计分析
print("\n" + "=" * 80)
print("描述性统计量")
print("=" * 80)

# 选择要分析的变量（排除Dates和stkcd）
analysis_cols = [col for col in df_winsorized.columns if col not in ['Dates', 'stkcd']]

# 计算描述性统计量
desc_stats = df_winsorized[analysis_cols].describe()

# 添加额外的统计量
additional_stats = pd.DataFrame({
    col: {
        'skewness': df_winsorized[col].skew(),
        'kurtosis': df_winsorized[col].kurtosis()
    }
    for col in analysis_cols
}).T

# 合并统计量
full_stats = pd.concat([desc_stats.T, additional_stats], axis=1)

# 重新排列列顺序
full_stats = full_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']]

# 保存描述性统计结果
full_stats.to_csv('../data/descriptive_statistics.csv')
print("\n描述性统计结果已保存到 data/descriptive_statistics.csv")

# 打印主要变量的描述性统计
print("\n主要变量描述性统计:")
print("-" * 120)
print(f"{'Variable':<15} {'Obs':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Median':>12} {'Max':>12} {'Skew':>10} {'Kurt':>10}")
print("-" * 120)

for col in analysis_cols[:15]:  # 显示前15个变量
    stats = full_stats.loc[col]
    print(f"{col:<15} {stats['count']:>8.0f} {stats['mean']:>12.4f} {stats['std']:>12.4f} "
          f"{stats['min']:>12.4f} {stats['50%']:>12.4f} {stats['max']:>12.4f} "
          f"{stats['skewness']:>10.4f} {stats['kurtosis']:>10.4f}")

if len(analysis_cols) > 15:
    print(f"\n... 还有 {len(analysis_cols) - 15} 个变量，详见 data/descriptive_statistics.csv")

# 4. 相关系数分析
print("\n" + "=" * 80)
print("相关系数分析")
print("=" * 80)

# 计算相关系数矩阵
correlation_matrix = df_winsorized[analysis_cols].corr()

# 保存相关系数矩阵
correlation_matrix.to_csv('../data/correlation_matrix.csv')
print("相关系数矩阵已保存到 data/correlation_matrix.csv")

# 找出高度相关的变量对（相关系数绝对值 > 0.8）
print("\n高度相关的变量对 (|Correlation| > 0.8):")
print("-" * 60)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            high_corr_pairs.append((var1, var2, corr_val))
            print(f"{var1:<15} - {var2:<15}: {corr_val:>8.4f}")

if not high_corr_pairs:
    print("未发现高度相关的变量对 (|Correlation| > 0.8)")

# 找出与y（收益率）相关性最强的变量
print("\n与收益率(y)相关性最强的15个变量:")
print("-" * 60)
y_corr = correlation_matrix['y'].drop('y').abs().sort_values(ascending=False)
for i, (var, corr) in enumerate(y_corr.head(15).items(), 1):
    actual_corr = correlation_matrix.loc['y', var]
    print(f"{i:2d}. {var:<15}: {actual_corr:>8.4f}")

# 5. 绘制相关系数热力图（选取主要变量）
print("\n" + "=" * 80)
print("生成相关系数热力图")
print("=" * 80)

# 选择主要变量进行可视化（包括y和一些重要特征）
key_vars = ['y', 'ACC', 'size', 'Lsize', 'ROA', 'ROE', 'BM', 'EP', 'mom1m', 'mom6m', 'mom12m', 
            'beta', 'betasq', 'ATO', 'GM', 'GP', 'CFOA', 'SP', 'SG', 'EY']
# 只保留实际存在的变量
key_vars = [v for v in key_vars if v in analysis_cols]

if len(key_vars) > 5:
    # 使用matplotlib绘制热力图
    corr_subset = df_winsorized[key_vars].corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_subset.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(key_vars)))
    ax.set_yticks(np.arange(len(key_vars)))
    ax.set_xticklabels(key_vars, rotation=45, ha='right')
    ax.set_yticklabels(key_vars)
    
    # 添加数值标注
    for i in range(len(key_vars)):
        for j in range(len(key_vars)):
            if i != j:  # 不显示对角线
                text = ax.text(j, i, f'{corr_subset.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Correlation Heatmap of Key Variables', fontsize=14, fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig('../correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("相关系数热力图已保存到 correlation_heatmap.png")
    plt.close()

# 6. 数据质量检查
print("\n" + "=" * 80)
print("数据质量检查")
print("=" * 80)

# 检查缺失值
missing_values = df_winsorized[analysis_cols].isnull().sum()
missing_vars = missing_values[missing_values > 0]
if len(missing_vars) > 0:
    print("\n存在缺失值的变量:")
    for var, count in missing_vars.items():
        pct = count / len(df_winsorized) * 100
        print(f"  {var}: {count} ({pct:.2f}%)")
else:
    print("\n所有变量均无缺失值")

# 检查极端值（缩尾后）
print("\n缩尾后极端值检查:")
outlier_count = 0
for col in analysis_cols[:10]:  # 检查前10个变量
    q1 = df_winsorized[col].quantile(0.25)
    q3 = df_winsorized[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((df_winsorized[col] < lower_bound) | (df_winsorized[col] > upper_bound)).sum()
    if outliers > 0:
        outlier_count += 1
        print(f"  {col}: {outliers} 个极端值 ({outliers/len(df_winsorized)*100:.2f}%)")

if outlier_count == 0:
    print("  主要变量缩尾后无明显极端值")

# 7. 保存处理后的数据
print("\n" + "=" * 80)
print("保存处理后的数据")
print("=" * 80)
df_winsorized.to_csv('../data/CHN_sample_data_processed.csv', index=False)
print("处理后的数据已保存到 data/CHN_sample_data_processed.csv")
print("包含: 1%缩尾处理 + Lsize变量")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
print("\n输出文件:")
print("  1. data/descriptive_statistics.csv - 描述性统计量")
print("  2. data/correlation_matrix.csv - 相关系数矩阵")
print("  3. correlation_heatmap.png - 相关系数热力图")
print("  4. data/CHN_sample_data_processed.csv - 处理后的数据")
