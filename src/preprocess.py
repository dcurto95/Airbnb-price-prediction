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


def one_hot_all_cat_features(df, avoid_class=False, metadata=None):
    cat_atts = df.select_dtypes(exclude=np.number).columns.to_list()
    cat_atts = cat_atts if not avoid_class or np.issubdtype(df.dtypes[-1], np.number) else cat_atts[:-1]

    new_encoded_attributes = []
    for cat_att in cat_atts:
        new_encoded_attributes.append(one_hot(df, cat_att, metadata))
    return new_encoded_attributes


def one_hot(df, att, metadata=None):
    values = df[att]
    new_encoded_columns = []
    if metadata and att in metadata:
        uniques = metadata[att]
    else:
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


def label_encoding_all_cat_features(df, metadata=None):
    cat_atts = df.select_dtypes(exclude=[np.number, np.datetime64]).columns.to_list()
    map_ = {}
    for cat_att in cat_atts:
        map_ = {**map_, **label_encoding(df, cat_att, metadata)}
    return map_


def label_encoding(df, att, metadata=None):
    values = df[att].to_numpy()
    if metadata and att in metadata:
        uniques = metadata[att]
    else:
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


def preprocess_dataset(dataset, norm_technique="minmax", metadata=None, exclude_norm_cols=[]):
    cols = dataset.columns.tolist()
    cols.insert(len(cols), cols.pop(cols.index('price')))
    dataset = dataset.reindex(columns=cols)

    exclude_norm_cols.append('price')
    if norm_technique == "z-score":
        z_score_all_num_features(dataset, exclude_norm_cols=exclude_norm_cols)
    else:
        minmax_all_num_features(dataset, exclude_norm_cols=exclude_norm_cols)

    new_features = one_hot_all_cat_features(dataset, avoid_class=True, metadata=metadata)

    return dataset, new_features


def clean_dataframe(dataset):
    zero_reviews_index = np.where(dataset['number_of_reviews'] == 0)[0]
    dataset.at[zero_reviews_index, 'reviews_per_month'] = 0
    dataset.at[zero_reviews_index, 'last_review'] = np.min(dataset['last_review'])

    date_to_timestamp(dataset)
    return dataset


def preprocess_for_plot(dataset):
    fix_missing_values_from_dataset(dataset)
    z_score_all_num_features(dataset)
    label_encoding_all_cat_features(dataset)
    return dataset


def array_to_dataframe(data, index, columns, classes):
    dataframe = pd.DataFrame(data=data, index=index, columns=columns)
    dataframe["classes"] = classes
    return dataframe


def parse_metadata(metadata):
    d = {}
    lines = str(metadata).split("\n")
    for line in lines:
        if 'nominal' in line:
            values = []
            spl = line.split('\'')
            att = spl[0][1:]
            for i in range(2, len(spl) - 1):
                if i % 2 == 0:
                    values.append(spl[i])
            d[att] = values
    return d


def date_to_timestamp(df):
    cat_atts = df.select_dtypes(include=np.datetime64).columns.to_list()

    for cat_att in cat_atts:
        last_review = datetime.timestamp(np.max(df[cat_att]))

        mean = np.mean(df[cat_att])
        ind = np.where(pd.isnull(df[cat_att]))
        df.loc[ind[0], cat_att] = mean
        df[cat_att] = df[cat_att].transform(lambda x: (last_review - datetime.timestamp(x)) / (24 * 3600))
    return df


def removal_of_price_outliers(features, values, feature_base):
    new_values = values[feature_base.astype(np.bool)]
    # defining quartiles and interquartile range
    first_quantile = np.quantile(new_values, 0.25)
    third_quantile = np.quantile(new_values, 0.75)
    IQR = third_quantile - first_quantile

    indexs = np.where((new_values < (first_quantile - 1.5 * IQR)) | (new_values > (third_quantile + 1.5 * IQR)))[0]
    print("Outliers:", len(indexs))
    trimmed_features = np.delete(features, indexs, axis=0)
    trimmed_values = np.delete(values, indexs, axis=0)
    return trimmed_features, trimmed_values
