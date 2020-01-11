import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import preprocess


def create_preprocessed_csv(data_df, title, with_neighbourhood=True):
    if title != 'fuzzy':
        logged_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'last_review',
                       'calculated_host_listings_count', 'availability_365']
    else:
        logged_cols = ['minimum_nights', 'calculated_host_listings_count', 'availability_365']
    for col in logged_cols:
        data_df[col] = np.log(data_df[col] + 1)
    data_df['price'] = np.log(data_df.price)
    print("Shape after cleaning and transformations", data_df.shape)
    # Cleaning outliers
    neighbourhoods = data_df.neighbourhood_group.unique().tolist()
    df_without_outliers = pd.DataFrame()
    for neighborhood in neighbourhoods:
        cleaned = preprocess.removal_of_outliers(data_df, 'neighbourhood_group', neighborhood, 'price', 1.5)
        df_without_outliers = df_without_outliers.append(cleaned)
    data_df = df_without_outliers

    # Change neighbourhood_group and neighbourhood
    if not with_neighbourhood:
        data_df.loc[(data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood_group'] = data_df.loc[
            (data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood']
        data_df = data_df.drop(columns=['neighbourhood'])

    # Normalize numerical and One-Hot categorical
    data_df = preprocess.preprocess_dataset(data_df, norm_technique='z-score', exclude_norm_cols=['price'])
    # Separate into X, y
    train, test = train_test_split(data_df, test_size=0.3, random_state=64)

    validation, test = train_test_split(test, test_size=0.5, random_state=64)

    if with_neighbourhood:
        train.to_csv(r'../data/train_' + title + '.csv', index=False)
        validation.to_csv(r'../data/validation_' + title + '.csv', index=False)
        test.to_csv(r'../data/test_' + title + '.csv', index=False)
    else:
        train.to_csv(r'../data/train_' + title + '_wo_Neigh.csv', index=False)
        validation.to_csv(r'../data/validation_' + title + '_wo_Neigh.csv', index=False)
        test.to_csv(r'../data/test_' + title + '_wo_Neigh.csv', index=False)


if __name__ == '__main__':
    # If cleaned dataset hasn't been created yet, set "export_clean_data = True"
    export_clean_data = False

    if export_clean_data:
        data_df = pd.read_csv('../data/AB_NYC_2019.csv')
        print("Original shape", data_df.shape)

        # Cleaning
        data_df = preprocess.clean_dataframe(data_df)

        # Transformations
        data_df = preprocess.date_to_timestamp(data_df)

        data_df.to_csv(r'../data/AB_NYC_2019_cleaned.csv', index=False)
    else:
        data_df_cleaned = pd.read_csv('../data/AB_NYC_2019_cleaned.csv')
        print("Original cleaned shape", data_df_cleaned.shape)
        data_df_fuzzy = pd.read_csv('../data/AB_NYC_2019_fuzzy.csv')
        print("Original fuzzy shape", data_df_fuzzy.shape)

        create_preprocessed_csv(data_df_cleaned, 'cleaned')
        create_preprocessed_csv(data_df_fuzzy, 'fuzzy')
        create_preprocessed_csv(data_df_cleaned, 'cleaned', with_neighbourhood=False)
        create_preprocessed_csv(data_df_fuzzy, 'fuzzy', with_neighbourhood=False)
