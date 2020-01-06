import math
import time
from datetime import datetime

import numpy as np
import pandas as pd


def z_score_all_num_features(df, exclude_norm_cols=[]):
    num_atts = df.select_dtypes(include=np.number).columns.to_list()
    for num_att in num_atts:
        if num_att not in exclude_norm_cols:
            df.loc[:, num_att] = z_score_normalization(df.loc[:, num_att])


def z_score_normalization(array):
    return ((array - array.mean()) / array.std()).astype(np.float)


def minmax_all_num_features(df, exclude_norm_cols=[]):
    num_atts = df.select_dtypes(include=np.number).columns.to_list()
    for num_att in num_atts:
        if num_att not in exclude_norm_cols:
            df.loc[:, num_att] = minmax(df.loc[:, num_att])


def minmax(array):
    return ((array - array.min()) / (array.max() - array.min())).astype(np.float)


def one_hot_all_cat_features(df, avoid_class=False):
    cat_atts = df.select_dtypes(exclude=np.number).columns.to_list()
    cat_atts = cat_atts if not avoid_class or np.issubdtype(df.dtypes[-1], np.number) else cat_atts[:-1]

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


def fix_missing_values_from_dataset(dataset):
    column_types = dataset.dtypes

    for column_name, col_type in zip(dataset, column_types):
        if np.issubdtype(col_type, np.number):
            # Check if column is full of Missing values
            if np.isnan(dataset[column_name]).all():
                dataset.drop(columns=[column_name], inplace=True)
            else:
                mean = dataset[column_name].mean()
                for value, index in zip(dataset[column_name], dataset.index):
                    # Fix missing values
                    if math.isnan(value):
                        dataset.at[index, column_name] = mean
        else:
            # dataset[column_name] = dataset[column_name].astype(str)
            values, count = np.unique(dataset[column_name], return_counts=True)
            most_common_value = values[np.argmax(count)]

            # If most common value is a missing value, get the next most common
            if most_common_value == b'?':
                max_sorted_indexes = np.argsort(count)[::-1]
                for index in max_sorted_indexes:
                    if values[index] != b'?':
                        most_common_value = values[index]
                        break

            for value, index in zip(dataset[column_name], dataset.index):
                # Fix missing values
                if value == b'?':
                    dataset.at[index, column_name] = most_common_value


def preprocess_dataset(dataset, class_col, norm_technique="minmax", exclude_norm_cols=[]):
    cols = dataset.columns.tolist()
    cols.insert(len(cols), cols.pop(cols.index(class_col)))
    dataset = dataset.reindex(columns=cols)

    exclude_norm_cols.append(class_col)
    if norm_technique == "z-score":
        z_score_all_num_features(dataset, exclude_norm_cols=exclude_norm_cols)
    else:
        minmax_all_num_features(dataset, exclude_norm_cols=exclude_norm_cols)

    new_features = one_hot_all_cat_features(dataset, avoid_class=True)

    return dataset, new_features


def clean_dataframe(dataset):
    zero_reviews_index = np.where(dataset['number_of_reviews'] == 0)[0]
    dataset.loc[(dataset['number_of_reviews'] == 0), 'reviews_per_month'] = 0
    dataset.loc[(dataset['number_of_reviews'] == 0), 'last_review'] = np.min(dataset['last_review'])

    date_to_timestamp(dataset)
    return dataset


def preprocess_for_plot(dataset):
    fix_missing_values_from_dataset(dataset)
    z_score_all_num_features(dataset)
    label_encoding_all_cat_features(dataset)
    return dataset


def date_to_timestamp(df):
    date_atts = df.select_dtypes(include=np.datetime64).columns.to_list()

    for date_att in date_atts:
        last_review = datetime.timestamp(np.max(df[date_att]))

        mean = np.mean(df[date_att])
        ind = np.where(pd.isnull(df[date_att]))
        df.loc[ind[0], date_att] = mean
        df[date_att] = df[date_att].transform(lambda x: (last_review - datetime.timestamp(x)) / (24 * 3600))
    return df


def removal_of_outliers(df, nhood, distance):
    new_piece = df[df["neighbourhood_group"] == nhood]["log_price"]
    # defining quartiles and interquartile range
    q1 = new_piece.quantile(0.25)
    q3 = new_piece.quantile(0.75)
    IQR = q3 - q1

    trimmed = df[(df["neighbourhood_group"] == nhood) & (df.log_price > (q1 - distance * IQR)) & (
                df.log_price < (q3 + distance * IQR))]
    return trimmed
