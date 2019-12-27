import math

import numpy as np
import pandas as pd


def z_score_all_num_features(df):
    num_atts = df.select_dtypes(include=np.number).columns.to_list()
    for num_att in num_atts:
        df.loc[:, num_att] = z_score_normalization(df.loc[:, num_att])


def z_score_normalization(array):
    return ((array - array.mean()) / array.std()).astype(np.float)


def minmax_all_num_features(df):
    num_atts = df.select_dtypes(include=np.number).columns.to_list()
    for num_att in num_atts:
        df.loc[:, num_att] = minmax(df.loc[:, num_att])


def minmax(array):
    return ((array - array.min()) / (array.max() - array.min())).astype(np.float)


def one_hot_all_cat_features(df, avoid_class=False, metadata=None):
    cat_atts = df.select_dtypes(exclude=np.number).columns.to_list()
    cat_atts = cat_atts if not avoid_class or np.issubdtype(df.dtypes[-1], np.number) else cat_atts[:-1]
    for cat_att in cat_atts:
        one_hot(df, cat_att, metadata)


def one_hot(df, att, metadata=None):
    values = df[att]

    if metadata and att in metadata:
        uniques = metadata[att]
    else:
        uniques = np.unique(values)

    for value in uniques:
        if type(value) == str:
            df[att + '_' + value] = (values.to_numpy().astype(str) == value).astype(np.int)
        else:
            df[att + '_' + str(value.decode("utf-8"))] = (values == value).astype(np.int)

    df.drop(columns=[att], inplace=True)


def label_encoding_all_cat_features(df, metadata=None):
    cat_atts = df.select_dtypes(exclude=np.number).columns.to_list()
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
            dataset[column_name] = dataset[column_name].astype(str)
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


def preprocess_dataset(dataset, to_numerical='oh', norm_technique="minmax", metadata=None):
    fix_missing_values_from_dataset(dataset)
    class_column_name = dataset.columns.to_list()[-1]

    if norm_technique == "z-score":
        z_score_all_num_features(dataset)
    else:
        minmax_all_num_features(dataset)

    if to_numerical == 'oh':
        one_hot_all_cat_features(dataset, avoid_class=True, metadata=metadata)
        d = label_encoding(dataset, class_column_name, metadata=metadata)
    elif to_numerical == 'le':
        d = label_encoding_all_cat_features(dataset, metadata=metadata)
    elif to_numerical is None:
        d = label_encoding(dataset, class_column_name, metadata=metadata)
    else:
        raise ValueError("Expected None, \'oh\' or \'le\' for to_numerical parameter, obtained " + to_numerical)
    return d


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
