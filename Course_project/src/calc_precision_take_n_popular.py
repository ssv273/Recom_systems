import pandas as pd
import numpy as np
from src.metrics import precision_at_k, recall_at_k
from src.utils import prefilter_items
from src.recommenders import MainRecommender

def calc_precision_take_n_popular(data_train, data_test, item_features, take_n_popular):
    data_train = prefilter_items(data_train, item_features=item_features, take_n_popular=take_n_popular)
    als_model = MainRecommender(data_train)

    result = data_test.groupby('user_id')['item_id'].unique().reset_index()
    result.columns=['user_id', 'actual']

    # result['als_recommendations'] = result['user_id'].apply(lambda x: als_model.get_als_recommendations(x, N=5))
    result['own_recommendations'] = result['user_id'].apply(lambda x: als_model.get_own_recommendations(x, N=5))
    # result['similar_items_recommendations'] = result['user_id'].apply(lambda x: als_model.get_similar_items_recommendation(x, N=5))

    res = {}
    # res['als_recommendations'] = result.apply(lambda row: precision_at_k(row['als_recommendations'], row['actual']), axis=1).mean()
    res['own_recommendations'] = result.apply(lambda row: precision_at_k(row['own_recommendations'], row['actual']), axis=1).mean()
    # res['similar_items_recommendations'] = result.apply(lambda row: precision_at_k(row['similar_items_recommendations'], row['actual']), axis=1).mean()

    return res, result