import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_csv('../data/CHN_sample_data.csv')

print("=" * 80)
print("数据清洗与筛选")
print("=" * 80)
print("原始数据行数:", len(df))
print("\n日期列前10个值:")
print(df['Dates'].head(10))
print("\n日期列唯一值数量:", df['Dates'].nunique())
print("\n日期范围:")
print("最小日期:", df['Dates'].min())
print("最大日期:", df['Dates'].max())

# 筛选2016年1月-2018年12月的数据
# 日期格式是YYYYMM，所以2016年1月=201601，2018年12月=201812
df_filtered = df[(df['Dates'] >= 201601) & (df['Dates'] <= 201812)]

print("\n筛选后数据行数:", len(df_filtered))
print("\n筛选后的日期范围:")
print("最小日期:", df_filtered['Dates'].min())
print("最大日期:", df_filtered['Dates'].max())

# 保存筛选后的数据
df_filtered.to_csv('../data/CHN_sample_data_filtered.csv', index=False)
print("\n筛选后的数据已保存到 data/CHN_sample_data_filtered.csv")

# 进行1%缩尾处理
print("\n" + "=" * 80)
print("1%缩尾处理")
print("=" * 80)

df_winsorized = df_filtered.copy()
numeric_cols = df_winsorized.select_dtypes(include=[np.number]).columns.tolist()
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

print(f"共对 {winsorized_count} 个变量进行了1%缩尾处理")

# 将size变量取对数，变为Lsize
print("\n" + "=" * 80)
print("Size变量对数转换")
print("=" * 80)

if 'size' in df_winsorized.columns:
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

# 保存处理后的数据
df_winsorized.to_csv('../data/CHN_sample_data_processed.csv', index=False)
print("\n处理后的数据已保存到 data/CHN_sample_data_processed.csv")
print("包含: 1%缩尾处理 + Lsize变量")

print("\n" + "=" * 80)
print("数据清洗完成!")
print("=" * 80)
