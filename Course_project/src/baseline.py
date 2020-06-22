# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from src.metrics import precision_at_k, recall_at_k


def random_recommendation(items, n=5):
    """Случайные рекоммендации"""
    
    items = np.array(items)
    recs = np.random.choice(items, size=n, replace=False)
    
    return recs.tolist()

def weighted_random_recommendation(items_weights, n=5):
    """Взвешенные cлучайные рекоммендации"""

    proba = items_weights['weight'].to_list()

    items = np.array(items_weights['item_id'])
    recs = np.random.choice(items, size=n, replace=False, p=proba)
    
    
    return recs.tolist()

def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""
    
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    
    recs = popular.head(n).item_id
    
    return recs.tolist()

def baseline(data_train,data_test, n=5):
  """Получаем значение precision@5 для базовых алгоритмов"""

  result = data_test.groupby('user_id')['item_id'].unique().reset_index()
  result.columns=['user_id', 'actual']

  # random_recommendation
  items = data_train.item_id.unique()
  result['random_recommendation'] = result['user_id'].apply(lambda x: random_recommendation(items, n=n))

  # weighted_random_recommendation
  tmp_df = data_train.groupby('item_id')['sales_value'].sum().reset_index()
  tmp_df = tmp_df[tmp_df['sales_value'] >= 1]
  tmp_df['log_sales_value'] = np.log(tmp_df['sales_value'])
  tmp_df['weight'] = tmp_df['log_sales_value'] / tmp_df['log_sales_value'].sum()
  tmp_df = tmp_df[['item_id', 'weight']]
  result['weighted_random_recommendation'] = result['user_id'].apply(lambda x: weighted_random_recommendation(tmp_df, n=n))

  # popular_recommendation
  popular_recs = popularity_recommendation(data_train, n=n)
  result['popular_recommendation'] = result['user_id'].apply(lambda x: popular_recs)

  # precision@5
  res = {}
  res['random_recommendation'] = result.apply(lambda row: precision_at_k(row['random_recommendation'], row['actual']), axis=1).mean()
  res['weighted_random_recommendation'] = result.apply(lambda row: precision_at_k(row['weighted_random_recommendation'], row['actual']), axis=1).mean()
  res['popular_recommendation'] = result.apply(lambda row: precision_at_k(row['popular_recommendation'], row['actual']), axis=1).mean()

  return res