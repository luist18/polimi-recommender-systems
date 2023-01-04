import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# this cell contains code that may not be explained in this specific notebook as it was used in other notebooks
# im using this to save this functions in case we want to use them in future implementations


def load_interactions(path):
    # load
    res = pd.read_csv(path,
                      usecols=['UserID', 'ItemID', 'Data']) \
        .rename(columns={'UserID': 'user_id', 'ItemID': 'item_id', 'Data': 'data'})
    return res


def sum_all_interactions(res, scale=True):
    # summing all interactions
    res['data'] = 1
    res['data'] = res.groupby(['user_id', 'item_id', 'data'])[
        'data'].transform('count')
    res = res.drop_duplicates(
        ['user_id', 'item_id', 'data']).reset_index(drop=True)

    if scale:
        # create a MinMaxScaler object
        scaler = RobustScaler()

        # fit the scaler to the "data" column
        scaler.fit(res[['data']])

        # the dataframe does not have any data = 0, we set this minimum value manually
        scaler.min_ = 0

        # transform the "data" column and store the result in a new column called "data_scaled"
        res['data'] = scaler.transform(res[['data']])

    return res


def one_interacted(res):
    # set all interactions to 1
    res['data'] = 1
    res = res.drop_duplicates(
        ['user_id', 'item_id', 'data']).reset_index(drop=True)

    return res


def negative_checking_positive_viewing(res):
    # we start by replacing 1 with -1 and 0 with 1
    res['data'] = res['data'].replace(1, -1)
    res['data'] = res['data'].replace(0, 1)
    # we then group by user_id, item_id, and data
    # we then sum the data column
    res['count'] = res.groupby(['user_id', 'item_id', 'data'])[
        'data'].transform('count')
    # we drop duplicates
    res = res.drop_duplicates(['user_id', 'item_id', 'data'])

    # we make count negative for data -1 and doule for data +1
    res['count'] = np.where(res['data'] < 0, -res['count'], 2*res['count'])
    # and then we sum it
    res = res.groupby(['user_id', 'item_id']).sum().drop(
        columns=['data'], axis=1).reset_index()

    # we will sclae with a quantile transfomer
    scaler = MinMaxScaler()
    # fit the scaler to the "count" column
    scaler.fit(res[['count']])
    # transform the "count" column and store the result
    res['data'] = scaler.transform(res[['count']])

    return res


def negative_checking_zero_to_one_viewing(length_archive, cap=True):
    # cap is True if we want max percentage to be 1
    res = pd.read_csv('../data/interactions_and_impressions.csv', usecols=['UserID', 'ItemID', 'Data']).rename(
        columns={'UserID': 'user_id', 'ItemID': 'item_id', 'Data': 'data'})

    # we start by replacing 1 with -1 and 0 with 1
    res['data'] = res['data'].replace(1, -1)
    res['data'] = res['data'].replace(0, 1)
    # we then group by user_id, item_id, and data
    # we then sum the data column
    res['count'] = res.groupby(['user_id', 'item_id', 'data'])[
        'data'].transform('count')
    # we drop duplicates
    res = res.drop_duplicates(['user_id', 'item_id', 'data'])

    # we merge with the length datafraem
    res = pd.merge(length_archive, res, on='item_id')
    # we calculate the seeing percentage
    res['percentage'] = res['count'] / res['length']
    if cap:
        # we cap it at 1
        res['percentage'] = res['percentage'].apply(
            lambda x: 1 if x > 1 else x)

    # we put it all together
    res = res.groupby(['item_id', 'length', 'user_id'], as_index=False) \
        .agg({'data': 'max', 'percentage': 'min'})

    # we set only checking for negative and the rest for ]0, 1]
    res['data'] = res.apply(
        lambda x: x['percentage'] if x['data'] == 1 else -1, axis=1)
    res = res.drop(['length', 'data', 'percentage'],
                   axis=1).reset_index().set_index('index')

    return res


def one_checking_one_to_two_viewing(res, length_archive, cap=True):
    # we start by replacing 1 with -1 and 0 with 1
    res['data'] = res['data'].replace(1, -1)
    res['data'] = res['data'].replace(0, 1)
    # we then group by user_id, item_id, and data
    # we then sum the data column
    res['count'] = res.groupby(['user_id', 'item_id', 'data'])[
        'data'].transform('count')
    # we drop duplicates
    res = res.drop_duplicates(['user_id', 'item_id', 'data'])

    # we merge with the length datafraem
    res = pd.merge(length_archive, res, on='item_id')
    # we calculate the seeing percentage
    res['percentage'] = res['count'] / res['length']
    if cap:
        # we cap it at 1
        res['percentage'] = res['percentage'].apply(
            lambda x: 1 if x > 1 else x)

    # we put it all together
    res = res.groupby(['item_id', 'length', 'user_id'], as_index=False) \
        .agg({'data': 'max', 'percentage': 'min'})

    # we want value = 1 if they only checked, and value between 1 and 2 for how much they saw it
    res['data'] = res.apply(
        lambda x: x['percentage'] + 1 if x['data'] == 1 else 1, axis=1)
    res = res.drop(['length', 'data', 'percentage'], axis=1).reset_index()

    return res


def load_length_feature(path):
    # load length features
    length_features = pd.read_csv(path)

    length_features = length_features.drop(columns=['feature_id'])
    length_features.rename(columns={'data': 'length'}, inplace=True)

    # save this for the future
    length_archive = length_features.copy()

    length_features = length_features.copy()
    length_features['length'] = length_features['length'].apply(
        lambda x: 0 if x == 1.0 else 1 if x <= 3.0 else 2 if x <= 10.0 else 3 if x <= 27.0 else 4 if x <= 55.0 else 5)

    length_features = pd.get_dummies(length_features, columns=['length'])

    return length_features, length_archive


def load_type_feature(path):
    # load types
    type_features = pd.read_csv(path)

    type_features = type_features.drop(
        columns=['data']).rename(columns={'feature_id': 'type'})

    type_features = pd.get_dummies(type_features, columns=['type'])

    return type_features


def load_target_ids(path):
    # load target_ids
    target_ids = pd.read_csv(path)

    user_ids = target_ids['user_id'].values.tolist()

    return user_ids
