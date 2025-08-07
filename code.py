# -*- coding: utf-8 -*-
"""
Created on 2025/8/7 20:36
@author: Wang bo
"""

import pandas as pd
import numpy as np
import os




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

def calculate_distribution(factor):
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    spread_percentile = np.percentile(factor, percentiles)
    for p, val in zip(percentiles, spread_percentile):
        print(f'{p}%percentile: {val:.6f}')
    return spread_percentile


if __name__ == '__main__':
    datas = get_data()

    # for data in datas:
    #     matrices = statistic_analysis(data)
    #     print(matrices)

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


    spread = np.concatenate([calculate_spread(data) for data in datas_new])
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print('\nThe Spread Distribution of All Sample:')
    calculate_distribution(spread)

    dates = np.concatenate([data['date'].values for data in datas_new])
    codes = np.concatenate([data['skey'].values for data in datas_new])

    columns = []
    for d in np.unique(dates):
        mask = dates == d
        spread_split = spread[mask]
        spread_pct = calculate_distribution(spread_split)
        columns.append(spread_pct)
    matrix = np.column_stack(columns)
    matrix = np.char.mod('%.6f', matrix)
    matrix = np.column_stack([np.array([f'{p}%' for p in percentiles]), matrix])
    table = np.vstack([np.array(['Percentile'] + list(map(str, np.unique(dates)))), matrix])
    print(table)




    for c in np.unique(codes):
        mask = codes == c
        spread_split = spread[mask]
        print('\nThe Spread Distribution cross section:')
        calculate_distribution(spread_split)



