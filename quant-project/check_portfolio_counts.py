import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('data/processed/monthly_clean_data.csv')
df['Trdmnt'] = pd.to_datetime(df['Trdmnt'])

# 计算EP
df['EP'] = df['NetProfit'] / df['MarketCap']

# 获取上月市值
df = df.sort_values(['Stkcd', 'Trdmnt'])
df['MarketCap_Lag'] = df.groupby('Stkcd')['MarketCap'].shift(1)
df = df.dropna(subset=['MarketCap_Lag'])

# 剔除最小30%市值股票（基于上月市值）
def exclude_by_month_lag(group):
    group = group.sort_values('MarketCap_Lag')
    n_keep = int(len(group) * 0.7)
    return group.iloc[-n_keep:]

df_filtered = df.groupby('Trdmnt', group_keys=False).apply(exclude_by_month_lag)
df_filtered = df_filtered.reset_index(drop=True)

# 统计每个月6个组合的数据量
print('=== 6个组合的数据量统计 ===\n')

results = []
months = sorted(df_filtered['Trdmnt'].unique())

for month in months[:5]:  # 只看前5个月作为示例
    month_data = df_filtered[df_filtered['Trdmnt'] == month].copy()
    
    # 按上月市值中位数分为Small/Big两组
    median_cap_lag = month_data['MarketCap_Lag'].median()
    small_stocks = month_data[month_data['MarketCap_Lag'] <= median_cap_lag]
    big_stocks = month_data[month_data['MarketCap_Lag'] > median_cap_lag]
    
    # EP分组函数
    def ep_grouping(stocks):
        if len(stocks) == 0:
            return 0, 0, 0
        
        negative_ep = stocks[stocks['EP'] < 0]
        positive_ep = stocks[stocks['EP'] >= 0]
        
        if len(positive_ep) > 0:
            positive_ep = positive_ep.sort_values('EP', ascending=False)
            n = len(positive_ep)
            n_v = int(n * 0.3)
            n_m = int(n * 0.4)
            
            v_count = n_v
            m_count = n_m
            g_count = len(positive_ep) - n_v - n_m + len(negative_ep)
        else:
            v_count = 0
            m_count = 0
            g_count = len(negative_ep)
        
        return v_count, m_count, g_count
    
    sv_n, sm_n, sg_n = ep_grouping(small_stocks)
    bv_n, bm_n, bg_n = ep_grouping(big_stocks)
    
    total_small = len(small_stocks)
    total_big = len(big_stocks)
    
    month_str = month.strftime('%Y-%m')
    print(f'月份: {month_str}')
    print(f'  Small组总数: {total_small} (V:{sv_n}, M:{sm_n}, G:{sg_n})')
    print(f'  Big组总数:   {total_big} (V:{bv_n}, M:{bm_n}, G:{bg_n})')
    print(f'  总计: {total_small + total_big}')
    print()

# 计算所有月份的平均数据量
print('=== 所有月份的平均数据量 ===')
sv_list, sm_list, sg_list = [], [], []
bv_list, bm_list, bg_list = [], [], []

for month in months:
    month_data = df_filtered[df_filtered['Trdmnt'] == month].copy()
    
    median_cap_lag = month_data['MarketCap_Lag'].median()
    small_stocks = month_data[month_data['MarketCap_Lag'] <= median_cap_lag]
    big_stocks = month_data[month_data['MarketCap_Lag'] > median_cap_lag]
    
    def ep_grouping(stocks):
        if len(stocks) == 0:
            return 0, 0, 0
        
        negative_ep = stocks[stocks['EP'] < 0]
        positive_ep = stocks[stocks['EP'] >= 0]
        
        if len(positive_ep) > 0:
            positive_ep = positive_ep.sort_values('EP', ascending=False)
            n = len(positive_ep)
            n_v = int(n * 0.3)
            n_m = int(n * 0.4)
            
            v_count = n_v
            m_count = n_m
            g_count = len(positive_ep) - n_v - n_m + len(negative_ep)
        else:
            v_count = 0
            m_count = 0
            g_count = len(negative_ep)
        
        return v_count, m_count, g_count
    
    sv_n, sm_n, sg_n = ep_grouping(small_stocks)
    bv_n, bm_n, bg_n = ep_grouping(big_stocks)
    
    sv_list.append(sv_n)
    sm_list.append(sm_n)
    sg_list.append(sg_n)
    bv_list.append(bv_n)
    bm_list.append(bm_n)
    bg_list.append(bg_n)

print(f'SV: 平均 {np.mean(sv_list):.1f} 只, 中位数 {np.median(sv_list):.0f} 只')
print(f'SM: 平均 {np.mean(sm_list):.1f} 只, 中位数 {np.median(sm_list):.0f} 只')
print(f'SG: 平均 {np.mean(sg_list):.1f} 只, 中位数 {np.median(sg_list):.0f} 只')
print(f'BV: 平均 {np.mean(bv_list):.1f} 只, 中位数 {np.median(bv_list):.0f} 只')
print(f'BM: 平均 {np.mean(bm_list):.1f} 只, 中位数 {np.median(bm_list):.0f} 只')
print(f'BG: 平均 {np.mean(bg_list):.1f} 只, 中位数 {np.median(bg_list):.0f} 只')
