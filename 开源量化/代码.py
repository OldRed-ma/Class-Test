import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df_close = pd.read_csv("D:\pythonProject\研1\收盘价.csv",index_col=0)
df_market_cap = pd.read_csv("D:\pythonProject\研1\流通市值.csv",index_col=0)
df_volume = pd.read_csv("D:\pythonProject\研1\成交量.csv",index_col=0)
df_amount = pd.read_csv("D:\pythonProject\研1\成交额.csv",index_col=0)
df_turnover_rate = pd.read_csv("D:\pythonProject\研1\换手率.csv",index_col=0)

# 剔除空白列
df_close = df_close.dropna(axis=1, how='any')
# 查看前几行
# print(df.info())
# print(df.head())


# 计算收敛因子
def calculate_cf(df_close, windows=[1, 5, 10, 20, 60, 120]):
    # 创建副本避免修改原数据
    result_df = pd.DataFrame(index=df_close.index, columns=df_close.columns)

    # 计算各周期均线
    ma_dict = {}
    # 当日收盘价即为1
    ma_dict[1] = df_close

    # 计算其他周期的均线
    for window in windows[1:]:
        print(f"正在计算{window}日均线...")
        ma_dict[window] = df_close.rolling(window=window, min_periods=int(window * 0.8)).mean()

    # 计算每条均线的标准差
    for date in df_close.index:
        for stock in df_close.columns:
            # 获取该股票在所有周期的均线值
            ma_values = []
            for window in windows:
                if date in ma_dict[window].index:
                    ma_val = ma_dict[window].loc[date, stock]
                    if not pd.isna(ma_val):
                        ma_values.append(ma_val)

            # 如果有足够的数据计算标准差
            if len(ma_values) >= 2:
                std_value = np.std(ma_values)
                # PCF公式:PCF=−log(1+std(ma1,ma5,ma10,ma30,ma60,ma120))
                pcf_value = -np.log(1 + std_value) if std_value > 0 else 0
                result_df.loc[date, stock] = pcf_value
            else:
                result_df.loc[date, stock] = np.nan
    print("PCF因子计算完成！")
    return result_df

# 因子中性化处理
def neutralize_factor(df_factor, df_market_cap, industry_data=None):
    # 对齐日期格式
    df_factor.index = pd.to_datetime(df_factor.index)
    df_market_cap.index = pd.to_datetime(df_market_cap.index)
    # 创建结果DataFrame
    neutralized_df = pd.DataFrame(index=df_factor.index, columns=df_factor.columns)
    # 按日期逐日处理
    for date in df_factor.index:
        # 获取当天的因子值
        daily_factors = df_factor.loc[date]
        # 获取当天的市值
        daily_mcap = df_market_cap.loc[date]
        # 如果有市值数据，进行市值中性化
        if daily_mcap is not None:
            # 对齐数据
            common_stocks = daily_factors.index.intersection(daily_mcap.index)
            factors_aligned = daily_factors[common_stocks].dropna()
            mcap_aligned = daily_mcap[common_stocks][factors_aligned.index]

            if len(factors_aligned) > 10:  # 至少有10个有效数据点
                # 市值取对数（通常用对数市值）
                log_mcap = np.log(mcap_aligned)

                # 线性回归去除市值影响
                X = np.array(log_mcap).reshape(-1, 1)
                y = np.array(factors_aligned)

                # 使用最小二乘法
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                # 残差即为中性化后的因子值
                residuals = y - y_pred

                # 标准化
                if residuals.std() > 0:
                    residuals = (residuals - residuals.mean()) / residuals.std()

                # 保存结果
                for i, stock in enumerate(factors_aligned.index):
                    neutralized_df.loc[date, stock] = residuals[i]

    print("因子中性化完成！")
    return neutralized_df

# 计算RankIC
def calculate_rankic(df_factor, df_close, period='ME'):
    # 修改日期格式
    df_close.index = pd.to_datetime(df_close.index)
    # 计算下期收益率（月频）
    if period == 'ME':
        # 月末数据
        monthly_factor = df_factor.resample('ME').last()  # 每月最后一个交易日的因子值
        monthly_close = df_close.resample('ME').last()  # 每月最后一个交易日的收盘价
        # 计算下月收益率
        monthly_returns = monthly_close.pct_change().shift(-1)  # 下月收益率

    # 对齐日期
    common_dates = monthly_factor.index.intersection(monthly_returns.index)
    common_dates = [d for d in common_dates if pd.notna(d)]

    # 计算每期RankIC
    rankic_values = []
    rankic_dates = []

    for i in range(len(common_dates) - 1):
        date = common_dates[i]
        next_date = common_dates[i + 1]

        # 当期因子值
        factors_today = monthly_factor.loc[date]

        # 下期收益率
        returns_next = monthly_returns.loc[date]

        # 对齐股票代码
        common_stocks = factors_today.index.intersection(returns_next.index)
        factors_aligned = factors_today[common_stocks].dropna()
        returns_aligned = returns_next[common_stocks][factors_aligned.index].dropna()

        # 计算Spearman秩相关系数（RankIC）
        if len(factors_aligned) > 10 and len(returns_aligned) > 10:
            # 计算秩次
            factor_ranks = factors_aligned.rank()
            return_ranks = returns_aligned.rank()
            # 计算相关系数
            rank_corr = factor_ranks.corr(return_ranks, method='pearson')
            rankic_values.append(rank_corr)
            rankic_dates.append(date)

    # 转换为Series
    rankic_series = pd.Series(rankic_values, index=rankic_dates)
    cum_rankic = rankic_series.cumsum()

    print(f"RankIC均值：{rankic_series.mean():.4%}")
    return rankic_series, cum_rankic

# 绘制图1
def plot_figure1(rankic_series, cum_rankic, title="价格收敛因子RankIC时序图"):

    fig, ax1 = plt.subplots(figsize=(14, 7))
    # 左轴：RankIC值(柱状图)
    color = 'tab:blue'
    ax1.set_xlabel('日期')
    ax1.set_ylabel('RankIC', color=color)
    ax1.bar(rankic_series.index, rankic_series.values,color=color, alpha=0.7, width=20,label='RankIC')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 右轴：累计RankIC(折线图)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('累计RankIC', color=color)
    ax2.plot(cum_rankic.index, cum_rankic.values, color=color, linewidth=2, label='累计RankIC(右轴)')
    ax2.tick_params(axis='y', labelcolor=color)


    # 添加标题和图例
    plt.title(title, fontsize=16, fontweight='bold')
    # 组合图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # 添加统计信息文本框
    stats_text = f'RankIC均值: {rankic_series.mean():.4%}\nRankIC标准差: {rankic_series.std():.4%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.20, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    # print("\nRankIC统计信息：")
    # print(f"均值：{rankic_series.mean():.4%}")
    # print(f"标准差：{rankic_series.std():.4%}")
    # print(f"IR（信息比率）：{rankic_series.mean() / rankic_series.std():.4f}")
    # print(f"正RankIC比例：{(rankic_series > 0).mean():.2%}")

# 计算因子分组收益
def calculate_group_returns(df_factor, df_close, n_groups=5, period='ME'):
    print(f"计算{n_groups}个分组收益...")
    # 1. 获取月末数据
    if period == 'ME':
        monthly_factor = df_factor.resample('ME').last()
        monthly_close = df_close.resample('ME').last()
        monthly_returns = monthly_close.pct_change().shift(-1)

    # 2. 对齐日期
    common_dates = monthly_factor.index.intersection(monthly_returns.index)
    common_dates = [d for d in common_dates if pd.notna(d)]

    # 3. 初始化结果存储
    group_returns_dict = {i: [] for i in range(n_groups)}
    group_dates = []
    long_short_returns = []

    # 4. 逐期计算分组收益
    for i in range(len(common_dates) - 1):
        date = common_dates[i]
        next_date = common_dates[i + 1]
        # 获取数据
        factors_today = monthly_factor.loc[date]
        returns_next = monthly_returns.loc[next_date]
        # 关键修复：更严格的对齐
        # 只保留两个数据集中都存在的股票
        common_stocks = factors_today.index.intersection(returns_next.index)
        factors_aligned = factors_today[common_stocks].dropna()
        returns_aligned = returns_next[common_stocks][factors_aligned.index].dropna()
        # 进一步对齐：确保两个序列索引完全一致
        aligned_index = factors_aligned.index.intersection(returns_aligned.index)
        factors_aligned = factors_aligned[aligned_index]
        returns_aligned = returns_aligned[aligned_index]
        # 确保有足够数据
        if len(factors_aligned) >= n_groups * 5:  # 每组至少5只股票
            try:
                # 使用rank方法确保每只股票都有分组
                ranks = factors_aligned.rank(method='first')
                # 等分成n_groups组
                quantiles = pd.qcut(ranks, q=n_groups, labels=False, duplicates='drop')
                # 计算各分组收益率
                group_returns = []
                for group in range(n_groups):
                    group_stocks = quantiles[quantiles == group].index
                    if len(group_stocks) > 0:
                        # 获取这些股票的下月收益率
                        group_return_series = returns_aligned[group_stocks]
                        # 计算等权平均收益率
                        group_return = group_return_series.mean()
                    else:
                        group_return = np.nan

                    group_returns.append(group_return)
                    group_returns_dict[group].append(group_return)

                # 只有当所有分组都有有效收益时才记录
                if all(not pd.isna(ret) for ret in group_returns):
                    # 计算多空对冲收益（第4组-第0组）
                    ls_return = group_returns[-1] - group_returns[0]
                    long_short_returns.append(ls_return)
                    group_dates.append(date)

            except Exception as e:
                print(f"日期{date}分组出错: {e}")
                continue
    # 5. 转换为DataFrame
    group_returns_df = pd.DataFrame(index=group_dates)
    for group in range(n_groups):
        if group in group_returns_dict and len(group_returns_dict[group]) == len(group_dates):
            group_returns_df[f'group_{group}'] = group_returns_dict[group][:len(group_dates)]

    long_short_series = pd.Series(long_short_returns, index=group_dates)

    print(f"分组收益计算完成！有效期数：{len(group_dates)}")
    return group_returns_df, long_short_series

# 绘制图2
def plot_figure2(group_returns, long_short, title="价格收敛因子分组收益非严格单调"):
    # 计算累计收益
    cum_group_returns = (1 + group_returns).cumprod()
    cum_long_short = (1 + long_short).cumprod()
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 左轴：分组累计收益（折线图）
    # 为每个分组绘制折线
    colors = ['blue', 'red', 'yellow', 'purple', 'green']  # 不同分组的颜色
    for i in range(len(group_returns.columns)):
        group_col = f'group_{i}'
        if group_col in cum_group_returns.columns:
            # 获取该分组的累计收益
            group_cum_returns = cum_group_returns[group_col]
            # 绘制折线图
            ax1.plot(group_cum_returns.index, group_cum_returns.values,
                     color=colors[i % len(colors)], linewidth=2,
                     marker='o', markersize=3, label=f'分组{i}')

    ax1.set_xlabel('日期')
    ax1.set_ylabel('分组累计收益', color='black')
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # 设置x轴日期格式
    plt.xticks(rotation=45)

    # 右轴：多空对冲累计收益（折线图）
    ax2 = ax1.twinx()
    if len(cum_long_short) > 0:
        ax2.plot(cum_long_short.index, cum_long_short.values,
                 color='grey', linewidth=1, linestyle='--',
                 label='多空对冲（右轴）')
        baseline_value = cum_long_short.values.min()
        ax2.fill_between(cum_long_short.index,
                         cum_long_short.values,
                         baseline_value,  # 填充的下边界（0线）
                         color='grey',
                         alpha=0.3,
                         label='多空对冲区域（右轴）')
        ax2.set_ylabel('多空对冲累计收益', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 添加统计信息
    stats_text = f'分组数: {len(group_returns.columns)}\n'
    for i in range(len(group_returns.columns)):
        group_col = f'group_{i}'
        if group_col in cum_group_returns.columns:
            final_return = cum_group_returns[group_col].iloc[-1] if len(cum_group_returns) > 0 else 0
            stats_text += f'分组{i}: {final_return:.2%}\n'

    if len(long_short) > 0:
        stats_text += f'\n多空年化收益: {long_short.mean() * 12:.2%}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.20, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    # 打印详细统计
    print("\n分组收益统计：")
    for i in range(len(group_returns.columns)):
        group_col = f'group_{i}'
        if group_col in cum_group_returns.columns:
            final_return = cum_group_returns[group_col].iloc[-1] if len(cum_group_returns) > 0 else 0
            print(f"分组{i}累计收益：{final_return:.2%}")

# 计算PCF与VCF相关性
def calculate_pcf_vcf(pcf_neutralized, vcf_neutralized, title="价格收敛因子与成交量收敛因子保持弱负相关性"):
    # 确保日期对齐
    common_dates = pcf_neutralized.index.intersection(vcf_neutralized.index)
    pcf_aligned = pcf_neutralized.loc[common_dates]
    vcf_aligned = vcf_neutralized.loc[common_dates]
    # 计算每日截面相关性
    correlation_series = []
    correlation_dates = []
    for date in common_dates:
        pcf_daily = pcf_aligned.loc[date].dropna()
        vcf_daily = vcf_aligned.loc[date].dropna()

        # 只计算共同股票
        common_stocks = pcf_daily.index.intersection(vcf_daily.index)
        if len(common_stocks) > 10:  # 至少有10只股票
            pcf_values = pcf_daily[common_stocks]
            vcf_values = vcf_daily[common_stocks]

            # 计算相关系数
            correlation = pcf_values.corr(vcf_values, method='pearson')
            if not pd.isna(correlation):
                correlation_series.append(correlation)
                correlation_dates.append(date)

    # 转换为Series
    correlation_series = pd.Series(correlation_series, index=correlation_dates)

    # 绘制图形
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制相关系数时序图
    ax.plot(correlation_series.index, correlation_series.values,
            color='blue', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=correlation_series.mean(), color='red', linestyle='-',
               alpha=0.7, label=f'均值: {correlation_series.mean():.2%}')

    # 设置图表属性
    ax.set_xlabel('日期')
    ax.set_ylabel('相关系数')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 添加统计信息
    stats_text = f'相关系数均值: {correlation_series.mean():.4%}\n'
    stats_text += f'相关系数标准差: {correlation_series.std():.4%}\n'
    stats_text += f'负相关比例: {(correlation_series < 0).mean():.2%}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    print("PCF与VCF相关性统计：")
    print(f"均值: {correlation_series.mean():.4%}")
    print(f"标准差: {correlation_series.std():.4%}")
    print(f"最小值: {correlation_series.min():.4%}")
    print(f"最大值: {correlation_series.max():.4%}")
    print(f"负相关比例: {(correlation_series < 0).mean():.2%}")

    return correlation_series

# 计算PVCF
def calculate_pvcf(pcf_neutralized, vcf_neutralized):
    print("构建价量双收敛因子(PVCF)...")
    # 确保日期对齐
    common_dates = pcf_neutralized.index.intersection(vcf_neutralized.index)
    pcf_aligned = pcf_neutralized.loc[common_dates]
    vcf_aligned = vcf_neutralized.loc[common_dates]
    # 创建结果DataFrame
    pvcf_df = pd.DataFrame(index=common_dates, columns=pcf_aligned.columns)

    # 逐日构建PVCF
    for date in common_dates:
        pcf_daily = pcf_aligned.loc[date]
        vcf_daily = vcf_aligned.loc[date]

        # 只处理共同股票
        common_stocks = pcf_daily.dropna().index.intersection(vcf_daily.dropna().index)

        if len(common_stocks) > 0:
            # 在截面上标准化后进行加总
            pcf_values = pcf_daily[common_stocks]
            vcf_values = vcf_daily[common_stocks]

            # 截面标准化（z-score）
            pcf_zscore = (pcf_values - pcf_values.mean()) / pcf_values.std()
            vcf_zscore = (vcf_values - vcf_values.mean()) / vcf_values.std()

            # 加总得到PVCF
            pvcf_values = pcf_zscore + vcf_zscore

            # 对PVCF再进行标准化（使其截面均值为0，标准差为1）
            pvcf_values = (pvcf_values - pvcf_values.mean()) / pvcf_values.std()

            # 保存结果
            pvcf_df.loc[date, common_stocks] = pvcf_values

    print("价量双收敛因子构建完成！")
    return pvcf_df

# 绘制图17
def plot_single_index_rankic(rankic_mean, rankicir, index_name="沪深300"):
    fig, ax1 = plt.subplots(figsize=(6, 8))
    # 左轴：RankIC均值柱状图
    bars = ax1.bar([index_name], [rankic_mean * 100],  # 转换为百分比
                   color='tab:blue', alpha=0.7, width=0.6)
    ax1.set_xlim([-1, 1])
    ax1.set_ylabel('RankIC均值 (%)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, max(rankic_mean * 100 * 1.5, 5)])  # 设置合理的y轴范围

    # 在柱子上方添加数值标签
    ax1.text(0, rankic_mean * 100, f'{rankic_mean:.3%}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 在图表上方添加RankICIR信息
    ax1.text(0.5, 0.95, f'RankICIR = {rankicir:.3f}',
             ha='center', va='top', transform=ax1.transAxes,
             fontsize=12, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # 添加标题
    plt.title(f'TRCF在{index_name}中的预测能力', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# TRCF因子的增强回测
def run_enhancement_backtest(df_trcf, df_close, top_n=50):
    # 将日度TRCF转换为月度（取月末值）
    df_trcf_monthly = df_trcf.resample('ME').last()  # 取每月最后一个交易日
    #  使用月度数据进行回测
    monthly_dates = df_trcf_monthly.index
    #  初始化记录
    portfolio_returns = []
    benchmark_returns = []
    trade_record_dates = []
    #  月度调仓循环
    for i in range(len(monthly_dates) - 1):
        current_month = monthly_dates[i]
        next_month = monthly_dates[i + 1]
        # 获取当期因子值（月度）
        factor_today = df_trcf_monthly.loc[current_month].dropna()
        # 找到实际交易日（与收盘价对齐）
        # 当期：找到小于等于当前月末的最后一个交易日
        current_trade_dates = df_close[df_close.index <= current_month].index
        if len(current_trade_dates) == 0:
            continue
        trade_date = current_trade_dates[-1]
        # 下期：找到小于等于下个月末的最后一个交易日
        next_trade_dates = df_close[df_close.index <= next_month].index
        if len(next_trade_dates) == 0:
            continue
        next_trade_date = next_trade_dates[-1]

        # 获取价格数据
        price_start = df_close.loc[trade_date]
        price_end = df_close.loc[next_trade_date]
        # 对齐股票池
        valid_stocks = (factor_today.index
                        .intersection(price_start.dropna().index)
                        .intersection(price_end.dropna().index))

        if len(valid_stocks) < max(top_n, 100):
            continue
        # 选股策略
        selected_stocks = factor_today[valid_stocks].nlargest(top_n).index.tolist()

        # 计算收益率
        stock_returns = (price_end[selected_stocks] / price_start[selected_stocks]) - 1
        portfolio_return = stock_returns.mean()

        # 基准收益率
        benchmark_return = (price_end[valid_stocks] / price_start[valid_stocks] - 1).mean()

        # 记录结果
        portfolio_returns.append(portfolio_return)
        benchmark_returns.append(benchmark_return)
        trade_record_dates.append(current_month)

    # 转换为时间序列
    portfolio_returns_series = pd.Series(portfolio_returns, index=trade_record_dates)
    benchmark_returns_series = pd.Series(benchmark_returns, index=trade_record_dates)

    # 计算净值曲线
    portfolio_nav = (1 + portfolio_returns_series).cumprod()
    benchmark_nav = (1 + benchmark_returns_series).cumprod()
    excess_nav = portfolio_nav / benchmark_nav

    print(f"\n✓ 回测完成！共 {len(portfolio_returns_series)} 个月度调仓点")
    print(f"  时间范围: {trade_record_dates[0].date()} 至 {trade_record_dates[-1].date()}")

    return {
        'portfolio_nav': portfolio_nav,
        'benchmark_nav': benchmark_nav,
        'excess_nav': excess_nav,
        'portfolio_returns': portfolio_returns_series,
        'benchmark_returns': benchmark_returns_series,
        'trade_dates': trade_record_dates
    }
# 绘制图18
def plot_figure18(results_dict):
    # 提取数据
    portfolio_nav = results_dict['portfolio_nav']
    benchmark_nav = results_dict['benchmark_nav']
    excess_nav = results_dict['excess_nav']
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 左轴：净值曲线
    ax1.fill_between(benchmark_nav.index, benchmark_nav.values, 0, color='gray', alpha=0.3, label='沪深300等权基准')

    # TRCF增强组合净值 - 蓝色实线
    ax1.plot(portfolio_nav.index, portfolio_nav.values, color='#1E90FF', linewidth=2.5, alpha=0.9, label='TRCF增强组合')

    # 左轴设置
    ax1.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax1.set_ylabel('组合净值', fontsize=12, fontweight='bold', color='#2F4F4F')
    ax1.tick_params(axis='y', labelcolor='#2F4F4F')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # 右轴：超额收益净值
    ax2 = ax1.twinx()
    ax2.plot(excess_nav.index, excess_nav.values, color='#DC143C', linewidth=2, alpha=0.9, label='超额收益（右轴）')
    # 水平参考线 (y=1)
    ax2.axhline(y=1.0, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
    # 右轴设置
    ax2.set_ylabel('超额收益净值', fontsize=12, fontweight='bold', color='#DC143C')
    ax2.tick_params(axis='y', labelcolor='#DC143C')
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # 图表标题
    plt.title('TRCF在沪深300指数中的增强表现', fontsize=16, fontweight='bold', pad=20)

    # ===== 添加关键统计信息 =====
    # 计算统计指标
    total_months = len(portfolio_nav)
    total_years = total_months / 12

    # 累计收益
    total_return_port = portfolio_nav.iloc[-1] - 1
    total_return_bench = benchmark_nav.iloc[-1] - 1
    total_excess = excess_nav.iloc[-1] - 1

    # 年化收益
    annual_port = (1 + total_return_port) ** (1 / total_years) - 1 if total_years > 0 else 0
    annual_bench = (1 + total_return_bench) ** (1 / total_years) - 1 if total_years > 0 else 0
    annual_excess = annual_port - annual_bench

    # 最大回撤（超额收益）
    excess_returns = results_dict['portfolio_returns'] - results_dict['benchmark_returns']
    excess_cum = (1 + excess_returns).cumprod()
    rolling_max = excess_cum.expanding().max()
    drawdown = (excess_cum - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 胜率（月度超额收益为正的月份比例）
    win_rate = (excess_returns > 0).mean()

    # 信息比率
    if len(excess_returns) > 1 and excess_returns.std() > 0:
        ir = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    else:
        ir = 0

    # 统计信息文本框
    stats_text = (f'统计期间: {total_months:.0f} 个月 ({total_years:.1f} 年)\n'
                  f'累计超额: {total_excess:.2%}\n'
                  f'年化超额: {annual_excess:.2%}\n'
                  f'信息比率: {ir:.2f}\n'
                  f'胜率: {win_rate:.2%}\n'
                  f'最大回撤: {max_drawdown:.2%}')

    props = dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.95,
                 edgecolor='#DAA520', linewidth=1.5)
    ax1.text(0.2, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # 优化布局
    plt.tight_layout()

    # 保存图表
    print("✓ 图表绘制完成!")
    plt.show()

    # ===== 打印详细统计报告 =====
    print(f"{'指标':<20} {'数值':<20} {'说明':<30}")
    print("-" * 70)
    print(f"{'累计超额收益':<20} {total_excess:<20.2%} {'增强组合相对基准总收益'}")
    print(f"{'年化超额收益':<20} {annual_excess:<20.2%} {'年化后的超额收益'}")
    print(f"{'信息比率 (IR)':<20} {ir:<20.2f} {'单位风险带来的超额收益'}")
    print(f"{'月度胜率':<20} {win_rate:<20.2%} {'月度超额收益为正的比例'}")
    print(f"{'最大回撤':<20} {max_drawdown:<20.2%} {'超额收益最大下跌幅度'}")
    print(f"{'基准年化收益':<20} {annual_bench:<20.2%} {'沪深300等权基准年化收益'}")
    print(f"{'组合年化收益':<20} {annual_port:<20.2%} {'TRCF增强组合年化收益'}")
    print("=" * 60)

    return fig
# 尝试复现图22等
def calculate_rankic_week(df_factor, df_close, period='W-FRI'):
    # 修改日期格式
    df_close.index = pd.to_datetime(df_close.index)
    # 计算下期收益率（周频）
    if period == 'W-FRI':
        monthly_factor = df_factor.resample('W-FRI').last()
        monthly_close = df_close.resample('W-FRI').last()
        monthly_returns = monthly_close.pct_change(fill_method=None).shift(-1)

    # 对齐日期
    common_dates = monthly_factor.index.intersection(monthly_returns.index)
    common_dates = [d for d in common_dates if pd.notna(d)]

    # 计算每期RankIC
    rankic_values = []
    rankic_dates = []

    for i in range(len(common_dates) - 1):
        date = common_dates[i]
        next_date = common_dates[i + 1]
        # 当期因子值
        factors_today = monthly_factor.loc[date]
        # 下期收益率
        returns_next = monthly_returns.loc[date]
        # 对齐股票代码
        common_stocks = factors_today.index.intersection(returns_next.index)
        factors_aligned = factors_today[common_stocks].dropna()
        returns_aligned = returns_next[common_stocks][factors_aligned.index].dropna()
        # 计算RankIC
        if len(factors_aligned) > 10 and len(returns_aligned) > 10:
            # 计算秩次
            factor_ranks = factors_aligned.rank()
            return_ranks = returns_aligned.rank()
            # 计算相关系数
            rank_corr = factor_ranks.corr(return_ranks, method='pearson')
            rankic_values.append(rank_corr)
            rankic_dates.append(date)

    # 转换为Series
    rankic_series = pd.Series(rankic_values, index=rankic_dates)
    cum_rankic = rankic_series.cumsum()

    print(f"RankIC均值：{rankic_series.mean():.4%}")
    return rankic_series, cum_rankic
def calculate_group_returns_week(df_factor, df_close, n_groups=5, period='W-FRI'):
    print(f"计算{n_groups}个分组收益...")
    if period == 'W-FRI':
        monthly_factor = df_factor.resample('W-FRI').last()
        monthly_close = df_close.resample('W-FRI').last()
        monthly_returns = monthly_close.pct_change(fill_method=None).shift(-1)
    common_dates = monthly_factor.index.intersection(monthly_returns.index)
    common_dates = [d for d in common_dates if pd.notna(d)]
    group_returns_dict = {i: [] for i in range(n_groups)}
    group_dates = []
    long_short_returns = []


    for i in range(len(common_dates) - 1):
        date = common_dates[i]
        next_date = common_dates[i + 1]
        # 获取数据
        factors_today = monthly_factor.loc[date]
        returns_next = monthly_returns.loc[next_date]
        # 关键修复：更严格的对齐
        # 只保留两个数据集中都存在的股票
        common_stocks = factors_today.index.intersection(returns_next.index)
        factors_aligned = factors_today[common_stocks].dropna()
        returns_aligned = returns_next[common_stocks][factors_aligned.index].dropna()
        # 进一步对齐：确保两个序列索引完全一致
        aligned_index = factors_aligned.index.intersection(returns_aligned.index)
        factors_aligned = factors_aligned[aligned_index]
        returns_aligned = returns_aligned[aligned_index]
        # 确保有足够数据
        if len(factors_aligned) >= n_groups * 5:  # 每组至少5只股票
            try:
                # 使用rank方法确保每只股票都有分组
                ranks = factors_aligned.rank(method='first')
                # 等分成n_groups组
                quantiles = pd.qcut(ranks, q=n_groups, labels=False, duplicates='drop')
                # 计算各分组收益率
                group_returns = []
                for group in range(n_groups):
                    group_stocks = quantiles[quantiles == group].index
                    if len(group_stocks) > 0:
                        group_return_series = returns_aligned[group_stocks]

                        group_return = group_return_series.mean()
                    else:
                        group_return = np.nan

                    group_returns.append(group_return)
                    group_returns_dict[group].append(group_return)

                if all(not pd.isna(ret) for ret in group_returns):
                    # 计算多空对冲收益（第4组-第0组）
                    ls_return = group_returns[-1] - group_returns[0]
                    long_short_returns.append(ls_return)
                    group_dates.append(date)

            except Exception as e:
                print(f"日期{date}分组出错: {e}")
                continue

    group_returns_df = pd.DataFrame(index=group_dates)
    for group in range(n_groups):
        if group in group_returns_dict and len(group_returns_dict[group]) == len(group_dates):
            group_returns_df[f'group_{group}'] = group_returns_dict[group][:len(group_dates)]

    long_short_series = pd.Series(long_short_returns, index=group_dates)

    print(f"分组收益计算完成！有效期数：{len(group_dates)}")
    return group_returns_df, long_short_series
# 仅更改了柱状图粗细
def plot_figure22(rankic_series, cum_rankic, title="收敛因子RankIC时序图"):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    # 左轴：RankIC值(柱状图)
    color = 'tab:blue'
    ax1.set_xlabel('日期')
    ax1.set_ylabel('RankIC', color=color)
    ax1.bar(rankic_series.index, rankic_series.values,color=color, alpha=0.7, width=5,label='RankIC')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 右轴：累计RankIC(折线图)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('累计RankIC', color=color)
    ax2.plot(cum_rankic.index, cum_rankic.values, color=color, linewidth=2, label='累计RankIC(右轴)')
    ax2.tick_params(axis='y', labelcolor=color)


    # 添加标题和图例
    plt.title(title, fontsize=16, fontweight='bold')
    # 组合图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # 添加统计信息文本框
    stats_text = f'RankIC均值: {rankic_series.mean():.4%}\nRankIC标准差: {rankic_series.std():.4%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.20, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

# 绘制图1，2
# 计算PCF因子，并保存，以便后续调试
# pcf_df = calculate_cf(df_close)
# # print(pcf_df)
# pcf_df.to_csv('PCF因子.csv')
# # 因子中性化处理，并更新保存的因子
# pcf_cleaned = pcf_df.dropna()
# pcf_neutralized = neutralize_factor(pcf_cleaned, df_market_cap)
# pcf_neutralized.to_csv('PCF因子.csv')
# pcf_df = pd.read_csv("D:\pythonProject\研1\PCF因子.csv",index_col=0)
# # print(pcf_neutralized)修改日期格式
# pcf_df.index = pd.to_datetime(pcf_df.index)
# # 计算RankIC
# rankic_series, cum_rankic = calculate_rankic(pcf_df, df_close)
# # 绘制图1
# plot_figure1(rankic_series, cum_rankic, title="价格收敛因子RankIC均值为0.2151%")
# # 计算因子分组收益
# group_returns, long_short = calculate_group_returns(pcf_df, df_close, n_groups=5)
# # 绘制图2
# plot_figure2(group_returns, long_short, title="价格收敛因子分组收益非严格单调")
#
# # 相同方法绘制图3，4
# vcf_df = calculate_cf(df_volume)
# # print(vcf_df)
# vcf_df.to_csv('VCF因子.csv')
# vcf_cleaned = vcf_df.dropna(how='all')
# vcf_neutralized = neutralize_factor(vcf_cleaned, df_market_cap)
# vcf_neutralized.to_csv('VCF因子.csv')
# vcf_df = pd.read_csv("D:\pythonProject\研1\VCF因子.csv",index_col=0)
# vcf_df.index = pd.to_datetime(vcf_df.index)
# rankic_series_volume, cum_rankic_volume = calculate_rankic(vcf_df, df_close)
# plot_figure1(rankic_series_volume, cum_rankic_volume, title="成交量收敛因子RankIC均值为1.9147%")
# group_returns_volume, long_short_volume = calculate_group_returns(vcf_df, df_close, n_groups=5)
# plot_figure2(group_returns_volume, long_short_volume, title="成交量收敛因子多空表现不佳")
#
# #相关性计算并绘制图5
# pcf_df = pd.read_csv("D:\pythonProject\研1\PCF因子.csv", index_col=0)
# vcf_df = pd.read_csv("D:\pythonProject\研1\VCF因子.csv", index_col=0)
# pcf_vcf = calculate_pcf_vcf(pcf_df, vcf_df)
#
# # 绘制图6，7
# pvcf_df = calculate_pvcf(pcf_df,vcf_df)
# # print(pvcf_df)
# pvcf_df.to_csv('PVCF因子.csv')
# pvcf_cleaned = pvcf_df.dropna(how='all')
# pvcf_neutralized = neutralize_factor(pvcf_cleaned, df_market_cap)
# pvcf_neutralized.to_csv('PVCF因子.csv')
# pvcf_df = pd.read_csv("D:\pythonProject\研1\PVCF因子.csv",index_col=0)
# pvcf_df.index = pd.to_datetime(pvcf_df.index)
# rankic_series_pvcf, cum_rankic_pvcf = calculate_rankic(pvcf_df, df_close)
# plot_figure1(rankic_series_pvcf, cum_rankic_pvcf, title="价量双收敛因子RankIC均值为1.6358%")
# group_returns_pvcf, long_short_pvcf = calculate_group_returns(pvcf_df, df_close, n_groups=5)
# plot_figure2(group_returns_pvcf, long_short_pvcf, title="价量双收敛因子分组收益单调性弱")
#
# # 相同方法绘制图8，9
# acf_df = calculate_cf(df_amount)
# # print(acf_df)
# # acf_df.to_csv('ACF因子.csv')
# acf_cleaned = acf_df.dropna(how='all')
# acf_neutralized = neutralize_factor(acf_cleaned, df_market_cap)
# acf_neutralized.to_csv('ACF因子.csv')
# acf_df = pd.read_csv("D:\pythonProject\研1\ACF因子.csv",index_col=0)
# acf_df.index = pd.to_datetime(acf_df.index)
# rankic_series_amount, cum_rankic_amount = calculate_rankic(acf_df, df_close)
# plot_figure1(rankic_series_amount, cum_rankic_amount, title="成交额收敛因子RankIC均值为2.2087%")
# group_returns_amount, long_short_amount = calculate_group_returns(acf_df, df_close, n_groups=5)
# plot_figure2(group_returns_amount, long_short_amount, title="成交额收敛因子分组收益")
#
# #相同方法绘制图11,12
# trcf_df = calculate_cf(df_turnover_rate)
# print(trcf_df)
# trcf_cleaned = trcf_df.dropna(how='all')
# trcf_neutralized = neutralize_factor(trcf_cleaned, df_market_cap)
# trcf_neutralized.to_csv('TRCF因子.csv')
# trcf_df = pd.read_csv("D:\pythonProject\研1\TRCF因子.csv",index_col=0)
# trcf_df.index = pd.to_datetime(trcf_df.index)
# rankic_series_turnover_rate, cum_rankic_turnover_rate = calculate_rankic(trcf_df, df_close)
# plot_figure1(rankic_series_turnover_rate, cum_rankic_turnover_rate, title="换手率收敛因子RankIC均值为2.3924%")
# group_returns_turnover_rate, long_short_turnover_rate = calculate_group_returns(trcf_df, df_close, n_groups=5)
# plot_figure2(group_returns_turnover_rate, long_short_turnover_rate, title="换手率收敛因子分组收益")
# 计算量化RankICIR
# rankic_mean = rankic_series_turnover_rate.mean()
# rankic_std = rankic_series_turnover_rate.std()
# rankicir_annualized = (rankic_mean / rankic_std) * np.sqrt(12)
# print(f'年化RankICIR: {rankicir_annualized:.4f}')

# # 绘制图17
# rankicir_annualized = (rankic_series_turnover_rate.mean() / rankic_series_turnover_rate.std()) * np.sqrt(12)
# plot_single_index_rankic(rankic_series_turnover_rate.mean(),rankicir_annualized,"沪深300")
#
# #运行回测,绘制图18
# df_close.index = pd.to_datetime(df_close.index)
# result_18 = run_enhancement_backtest(trcf_df, df_close, top_n=50)
# plot_figure18(result_18)
#
# # 尝试复现图22等
# trcf_df = pd.read_csv("D:\pythonProject\研1\TRCF因子.csv",index_col=0)
# trcf_df.index = pd.to_datetime(trcf_df.index)
# rankic_series_turnover_rate, cum_rankic_turnover_rate = calculate_rankic_week(trcf_df, df_close,period='W-FRI')
# plot_figure22(rankic_series_turnover_rate, cum_rankic_turnover_rate, title="周频调仓下换手率收敛因子RankIC均值为1.2372%")
# group_returns_turnover_rate, long_short_turnover_rate = calculate_group_returns_week(trcf_df, df_close, n_groups=5)
# plot_figure2(group_returns_turnover_rate, long_short_turnover_rate, title="换手率收敛因子分组收益")
# rankic_mean = rankic_series_turnover_rate.mean()
# rankic_std = rankic_series_turnover_rate.std()
# rankicir_annualized = (rankic_mean / rankic_std) * np.sqrt(12)
# print(f'年化RankICIR: {rankicir_annualized:.4f}')