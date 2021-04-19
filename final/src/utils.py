# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:51:51 2021

@author: snetkova
"""
import pandas as pd
import numpy as np

def prefilter_items(data, item_features):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data.loc[data['item_id'].isin(top_popular), 'item_id'] = 999999
    # data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    # data = data[~data['item_id'].isin(top_notpopular)]
    data.loc[data['item_id'].isin(top_notpopular), 'item_id'] = 999999
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    older_52_week = set(item_features['item_id']) - set(data.loc[data['week_no'] >= max(data['week_no']) - 52, 'item_id'])
    # data = data[~data['item_id'].isin(older_52_week)]
    data.loc[data['item_id'].isin(older_52_week), 'item_id'] = 999999
    
    # Уберем не интересные для рекомендаций категории (department)
    data = data.merge(item_features[['department','item_id']], how = 'left', on = 'item_id')
#     data = pd.merge(data, item_features[['department','item_id']], on = 'item_id')
    total =  data['sales_value'].sum()
    departments = data.groupby('department')['sales_value'].sum().sort_values(ascending = False).cumsum().reset_index()
    departments_list = departments[departments['sales_value'] > 0.99 * total].department.to_list()
    # data = data[~data['department'].isin(departments_list)]
    data.loc[data['department'].isin(departments_list), 'item_id'] = 999999
    
    # Уберем слишком дешевые товары (на них не заработаем). 
    data['price'] = data['sales_value'] / (data['quantity'] + 1e-8)
    prices = data.groupby('item_id')['price'].mean().reset_index()
    cheap_items = prices[prices['price'] > 2].item_id.tolist()
    # data = data[~data['item_id'].isin(cheap_items)]
    data.loc[data['item_id'].isin(cheap_items), 'item_id'] = 999999
    
    # Уберем слишком дорогие товарыs
    expensive_items = prices[prices['price'] > prices['price'].quantile(0.99)].item_id.tolist()
    # data = data[~data['item_id'].isin(expensive_items)]
    data.loc[data['item_id'].isin(expensive_items), 'item_id'] = 999999
    # ...
    
    data = data.drop(['price'], axis = 1)
#     data = data.drop(['department', 'price'], axis = 1)
    
    return data
    
def postfilter_items(user_id, recommednations):
    pass

def get_candidates(data, rec_func, score_func, usr_list, n = 50):
    user_data = pd.DataFrame(data['user_id'].unique())
    user_data.columns = ['user_id']

    # Пока только warm start
#     train_users = data_train_lvl_1['user_id'].unique() # data_train_lvl_1!!!
    user_data = user_data[user_data['user_id'].isin(usr_list)]

    user_data['candidates'] = user_data['user_id'].apply(lambda x: rec_func(x, N = n))
    user_data['scores'] = user_data['user_id'].apply(lambda x: score_func(x, N = n))
    
    s1 = user_data.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    s2 = user_data.apply(lambda x: pd.Series(x['scores']), axis=1).stack().reset_index(level=1, drop=True)
    s1.name = 'item_id'
    s2.name = 'item_score'
    
    user_data = user_data.join(pd.concat([s1, s2], axis = 1)).drop(['candidates', 'scores'], axis=1).reset_index(drop=True)
    user_data['flag'] = 1
    
    return user_data

def get_base_dataset(data, user_data):
    target_data = data[['user_id', 'item_id']].copy()
    target_data['target'] = 1  # тут только покупки 
    target_data = user_data.merge(target_data, on=['user_id', 'item_id'], how='left')
    target_data['target'].fillna(0, inplace= True)
    target_data.drop('flag', axis=1, inplace=True)
    target_data = target_data.drop_duplicates()
    target_data = target_data.reset_index(drop=True)

    return target_data

def add_features(data, stats, item_feats, user_feats):
    dt = data.copy()
    stats = stats.merge(item_feats[['item_id', 'manufacturer', 'department', 'brand']], how = 'left', on = 'item_id')
    
    # средний чек и количество покупок в неделю
    user_stats = stats.groupby('user_id')['quantity', 'sales_value', 'week_no', 'basket_id'].agg(money = ('sales_value', 'sum'), qty = ('quantity', 'sum'),  bask = ('basket_id', 'nunique'), wk = ('week_no', 'nunique')).reset_index()
    user_stats['avg_bill'] = user_stats['money'] / user_stats['bask']
    user_stats['weekly_purch'] = user_stats['bask'] / user_stats['wk']
    user_feats = user_feats.merge(user_stats[['user_id', 'avg_bill', 'weekly_purch']], how = 'right', on = 'user_id')

    # цена товара и количество покупок в неделю
    item_stats = stats.groupby('item_id')['quantity', 'sales_value', 'week_no'].agg(money = ('sales_value', 'sum'), qty = ('quantity', 'sum'), wk = ('week_no', 'nunique')).reset_index()
    item_stats['price'] = item_stats['money'] / (item_stats['qty']+1e5)
    item_stats['weekly'] = item_stats['qty'] / (item_stats['wk']+1e5)
    item_feats = item_feats.merge(item_stats[['item_id', 'price', 'weekly']], how = 'right', on = 'item_id')
    item_feats = item_feats.fillna(0) #если не продавались

    # количество покупок в каждой категории
    dc = stats.groupby(['user_id', 'department'])['item_id'].count().reset_index()
    dc.columns = ['user_id', 'department', 'd_qty']
    
    # количество покупок по производителю
    sc = stats.groupby(['user_id', 'manufacturer'])['item_id'].count().reset_index()
    sc.columns = ['user_id', 'manufacturer', 'm_qty']
    
    dt = dt.merge(item_feats, on='item_id', how='left')
    dt = dt.merge(user_feats, on='user_id', how='left')
    dt = pd.merge(dt, dc, on=['user_id', 'department'], how = 'left')
    dt = pd.merge(dt, sc, on=['user_id', 'manufacturer'], how = 'left')
    
    return dt

def get_model_recs(data, preds, n = 10):
    probs = pd.Series(preds, name = 'probs', index = data.index)
    item_probs = pd.concat([data[['user_id', 'item_id']], probs], axis = 1).drop_duplicates()
    item_probs = item_probs.sort_values(by = ['user_id', 'probs'], ascending = False)
    item_probs = item_probs.groupby('user_id')['item_id'].agg(model_recs = 'unique').reset_index()
    item_probs['model_recs'] = item_probs['model_recs'].apply(lambda x: x[:n])
    
    return item_probs