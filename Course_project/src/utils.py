import pandas as pd
import numpy as np

def prefilter_items(data, item_features, take_n_popular=5000):

    # # убираем самые популярные товары, их и так купят
    # popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    # popularity.rename(columns={'user_id':'share_unique_users'}, inplace=True)

    # top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    # data = data[~data['item_id'].isin(top_popular)]

    # # уберем самые НЕпопулярные, их и так не купят
    # top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    # data = data[~data['item_id'].isin(top_notpopular)]

    # # уберем товары, которые не продавались за последние 12 месяцев

    # # уберем не интересные для рекомендаций категории (department)
    # department_size = pd.DataFrame(item_features.\
    #                                 groupby('department')['item_id'].nunique().\
    #                                 sort_values(ascending=False)).reset_index()

    # department_size.columns = ['department', 'n_items']

    # rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    # items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    # data = data[~data['item_id'].isin(rare_departments)]

    # # уберем слишком дешевые товары, на них не заработаем
    # data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    # data = data[data['price'] > 2]

    # # уберем слишком дорогие товары
    # data = data[data['price'] < 50]

    # возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity':'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # заведем фиктивный товар (которым заменим все остальные товары не из топ-N)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data