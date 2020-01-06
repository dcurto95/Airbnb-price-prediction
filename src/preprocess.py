from datetime import datetime

import numpy as np
import pandas as pd


def z_score_all_num_features(df, exclude_norm_cols=()):
    num_atts = df.select_dtypes(include=np.number).columns.to_list()
    for num_att in num_atts:
        if num_att not in exclude_norm_cols:
            df.loc[:, num_att] = z_score_normalization(df.loc[:, num_att])


def z_score_normalization(array):
    return ((array - array.mean()) / array.std()).astype(np.float)


def minmax_all_num_features(df, exclude_norm_cols=()):
    num_atts = df.select_dtypes(include=np.number).columns.to_list()
    for num_att in num_atts:
        if num_att not in exclude_norm_cols:
            df.loc[:, num_att] = minmax(df.loc[:, num_att])


def minmax(array):
    return ((array - array.min()) / (array.max() - array.min())).astype(np.float)


def one_hot_all_cat_features(df):
    cat_atts = df.select_dtypes(exclude=np.number).columns.to_list()

    new_encoded_attributes = []
    for cat_att in cat_atts:
        new_encoded_attributes.append(one_hot(df, cat_att))
    return new_encoded_attributes


def one_hot(df, att):
    values = df[att]
    new_encoded_columns = []
    uniques = np.unique(values)

    for value in uniques:
        if type(value) == str:
            df.insert(df.shape[1] - 1, att + '_' + value, (values.to_numpy().astype(str) == value).astype(np.int))
            new_encoded_columns.append(att + '_' + value)
        else:
            df.insert(df.shape[1] - 1, att + '_' + str(value.decode("utf-8")), (values == value).astype(np.int))
            new_encoded_columns.append(att + '_' + str(value.decode("utf-8")))

    df.drop(columns=[att], inplace=True)
    return new_encoded_columns


def label_encoding_all_cat_features(df):
    cat_atts = df.select_dtypes(exclude=[np.number, np.datetime64]).columns.to_list()
    map_ = {}
    for cat_att in cat_atts:
        map_ = {**map_, **label_encoding(df, cat_att)}
    return map_


def label_encoding(df, att):
    values = df[att].to_numpy()
    uniques = np.unique(values)
    sub_map = {}
    for i in range(len(uniques)):
        value = uniques[i]
        if type(value) == str:
            values[values.astype(str) == value] = i
        else:
            values[values == value] = i
        sub_map[i] = value
    df[att + '_enc'] = values.astype(np.int)
    df.drop(columns=[att], inplace=True)
    return {att + '_enc': sub_map}


def preprocess_dataset(dataset, norm_technique="minmax", exclude_norm_cols=()):
    if norm_technique == "z-score":
        z_score_all_num_features(dataset, exclude_norm_cols=exclude_norm_cols)
    else:
        minmax_all_num_features(dataset, exclude_norm_cols=exclude_norm_cols)

    one_hot_all_cat_features(dataset)
    # label_encoding_all_cat_features(dataset)

    return dataset


def clean_dataframe(df):
    df.drop(columns=df.columns[:4], inplace=True)

    df.drop(df[df['price'] == 0].index, inplace=True)

    df.loc[(df['number_of_reviews'] == 0), 'reviews_per_month'] = 0

    df['last_review'] = pd.to_datetime(df['last_review'])
    df.loc[(df['number_of_reviews'] == 0), 'last_review'] = np.min(df['last_review'])

    return df


def date_to_timestamp(df):
    date_atts = df.select_dtypes(include=np.datetime64).columns.to_list()

    for date_att in date_atts:
        last_review = datetime.timestamp(np.max(df[date_att]))

        df[date_att] = df[date_att].transform(lambda x: (last_review - datetime.timestamp(x)) / (24 * 3600))

    return df


def removal_of_outliers(df, filter_by, filter_value, target, distance):
    filtered_data = df[df[filter_by] == filter_value][target]
    # defining quartiles and interquartile range
    q1 = filtered_data.quantile(0.25)
    q3 = filtered_data.quantile(0.75)
    IQR = q3 - q1

    cleaned = df[(df[filter_by] == filter_value) & (df[target] > (q1 - distance * IQR)) & (df[target] < (q3 + distance * IQR))]
    return cleaned
