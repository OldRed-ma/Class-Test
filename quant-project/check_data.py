import pandas as pd

df = pd.read_csv('data/processed/monthly_clean_data.csv')
df['Trdmnt'] = pd.to_datetime(df['Trdmnt'])

# 检查NetProfit的缺失情况
print('=== NetProfit缺失情况 ===')
print(df.groupby(df['Trdmnt'].dt.year)['NetProfit'].apply(lambda x: x.notna().sum()))

print('\n=== 有NetProfit数据的最早月份 ===')
df_with_profit = df[df['NetProfit'].notna()]
print('最早月份:', df_with_profit['Trdmnt'].min())
print('有净利润数据的记录数:', len(df_with_profit))

print('\n=== 各年月度有净利润数据的股票数 ===')
monthly_counts = df[df['NetProfit'].notna()].groupby('Trdmnt').size()
print(monthly_counts.head(20))
