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
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    data['price'] = data['sales_value'] / (data['quantity'] + 1e-8)
    prices = data.groupby('item_id')['price'].mean().reset_index()
    cheap_items = prices[prices['price'] > 0.60].item_id.tolist()
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