# -*- coding: utf-8 -*-
"""
Created on 2025/8/7 20:36
@author: Wang bo
"""

import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr, rankdata
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False



def get_data():
    datas = []
    for file in os.listdir('./data'):
        df = pd.read_csv('./data/' + file)
        df['last'] = pd.to_numeric(df['last'], errors='coerce').astype('float32').ffill()
        datas.append(df)
    return datas


def statistic_analysis(data):
    data.sort_values(by='date', inplace=True)
    highest = data['high'].max()
    lowest = data['low'].min()
    cum_ret = data['last'].pct_change().sum()
    cum_amount = data.groupby('date')['amount'].tail(1).sum()

    bp1 = data['bp1'].values
    ap1 = data['ap1'].values
    bp1_diff = bp1[1:] != bp1[:-1]
    ap1_diff = ap1[1:] != ap1[:-1]
    bbo_change = np.sum(bp1_diff | ap1_diff)
    return highest, lowest, cum_ret, cum_amount, bbo_change


def resample(data):
    data[['date', 'time']] = data[['date', 'time']].astype(str)
    data['time'] = data['time'].astype(str).str.zfill(9).str[:6]

    data['time'] = data['time'].str.replace(r'(\d{2})(\d{2})(\d{2})', r'\1:\2:\3', regex=True)

    date = data['date'].iloc[0]
    times = pd.date_range(date+' 09:30:00', date+' 11:30:00', freq='3s').append(
        pd.date_range(date+' 13:00:00', date+' 14:56:57', freq='3s')
    )
    data['DateTime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data.set_index('DateTime', inplace=True)
    data = data.reindex(times)
    data = data.ffill()
    return data


def calculate_spread(data):
    data['mid_price'] = (data['bp1'] + data['ap1']) / 2
    spread = (data['ap1'] - data['bp1']) / data['mid_price']
    return spread.values


def calculate_distribution(spread, percentiles, dates, codes):
    spread_pct = np.percentile(spread, percentiles)
    spread_pct = np.char.mod('%.6f', spread_pct)
    print('\nThe Spread Distribution of All Sample:')
    column0 = np.array([f'{p}%' for p in percentiles])
    print(np.column_stack([column0, spread_pct]))

    columns = []
    for d in np.unique(dates):
        mask = dates == d
        spread_split = spread[mask]
        spread_pct = np.percentile(spread_split, percentiles)
        columns.append(spread_pct)
    matrix = np.column_stack(columns)
    matrix = np.char.mod('%.6f', matrix)
    matrix = np.column_stack([column0, matrix])
    table = np.vstack([np.array(['Percentile'] + list(map(str, np.unique(dates)))), matrix])
    print('\nThe Spread Distribution cross date:')
    print(table)

    columns = []
    for c in np.unique(codes):
        mask = codes == c
        spread_split = spread[mask]
        spread_pct = np.percentile(spread_split, percentiles)
        columns.append(spread_pct)
    matrix = np.column_stack(columns)
    matrix = np.char.mod('%.6f', matrix)
    matrix = np.column_stack([column0, matrix])
    table = np.vstack([np.array(['Percentile'] + list(map(str, np.unique(codes)))), matrix])
    print('\nThe Spread Distribution cross section:')
    print(table)


def calculate_volatility(spread, codes, window=20):
    volatility = []
    for c in np.unique(codes):
        mask = codes == c
        spread_split = spread[mask]
        shape = (len(spread_split) - window + 1, window)
        strides = (spread_split.strides[0], spread_split.strides[0])
        windows = np.lib.stride_tricks.as_strided(spread_split, shape, strides)
        stds = np.std(windows, axis=1)
        x = np.arange(window)
        x_mean = np.mean(x)
        x_centered = x - x_mean
        denom = np.sum(x_centered ** 2)

        y_centered = windows - np.mean(windows, axis=1, keepdims=True)
        numer = np.sum(x_centered * y_centered, axis=1)
        slopes = numer / denom
        volatility.append(stds * (1 + np.abs(slopes)))
    return volatility


def calculate_weighted_price(data):
    return (data['bp1'] * data['bq1'] + data['ap1'] * data['aq1']) / (data['bq1'] + data['aq1'])
    # return (data['bp1'] + data['ap1']) / 2


def calculate_corr(data):
    data['imbalance'] = (data['bq1'] - data['aq1']) / (data['bq1'] + data['aq1'])
    data['mid_price'] = calculate_weighted_price(data)
    data['ret'] = data.groupby(['skey', 'date'])['mid_price'].pct_change()
    data['ret'] = data.groupby('skey')['ret'].shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data['DateTime'] = pd.to_datetime(data['date'] + ' ' + data['time'])

    def _corr(x):
        x = x[['imbalance', 'ret']].dropna()
        if len(x) <= 2:
            return np.nan
        x1 = x['imbalance'].values
        x2 = x['ret'].values
        if np.std(x1) == 0 or np.std(x2) == 0:
            return np.nan
        return np.corrcoef(x1, x2)[0, 1]

    ic = data.groupby('DateTime').apply(_corr).values

    ic_mean = np.nanmean(ic)
    ic_std = np.nanstd(ic)
    icir = ic_mean / ic_std
    print(f'Pearson Correlation: IC_mean {ic_mean:.4f}, ICIR: {icir:.4f}')

    def rank_group(x):
        ranks = rankdata(x, method='average')
        groups = np.floor((ranks - 1) / len(x) * 3).astype(int)
        return groups

    data['factor_rank'] = data.groupby('DateTime')['imbalance'].transform(rank_group)


    plt_df = []
    for rank, group in data.groupby('factor_rank'):
        group = group.groupby('DateTime')['ret'].mean().sort_index()
        cum_ret = group.fillna(0).cumsum().rename(rank)
        plt_df.append(cum_ret)
    plt_df = pd.concat(plt_df, axis=1).ffill()
    plt_df['x_str'] = plt_df.index.strftime('%m-%d %H:%M:%S')
    print(plt_df)

    plt.figure(figsize=(10, 5))
    colors = ['darkblue', 'blue', 'red']
    for i, col in enumerate(plt_df.columns[:-1]):
        plt.plot(plt_df['x_str'], plt_df[col].values, label=f'第{i+1}组', color=colors[i])
    xticks = plt_df['x_str'][::500]
    plt.xticks(ticks=xticks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


    



if __name__ == '__main__':
    datas = get_data()

    # 计算数据基本统计指标
    for data in datas:
        matrices = statistic_analysis(data)
        print(matrices)

    # 将样本转化为标准样本
    datas_new = []
    for data in datas:
        code = data['skey'].iloc[0]
        standard = []
        for date, group in data.groupby('date'):
            group_new = resample(group)
            print(f'股票{code}-{date}共有: {len(group_new)}行')
            standard.append(group_new)
        data_new = pd.concat(standard, axis=0)
        datas_new.append(data_new)

    # 计算Spread相关统计指标
    spread = np.concatenate([calculate_spread(data) for data in datas_new])
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dates = np.concatenate([data['date'].values for data in datas_new])
    codes = np.concatenate([data['skey'].values for data in datas_new])
    calculate_distribution(spread, percentiles, dates, codes)

    # 计算波动性指标
    volatility = calculate_volatility(spread, codes)
    print(len(volatility))

    # 计算不平衡因子以及和未来收益的相关性
    calculate_corr(pd.concat(datas_new, axis=0))



