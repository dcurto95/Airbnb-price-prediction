import pandas as pd
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

import preprocess

if __name__ == '__main__':
    data_df = pd.read_csv("../data/AB_NYC_2019.csv")

    data_df = data_df.drop(columns=data_df.columns[:4])
    data_df['last_review'] = pd.to_datetime(data_df['last_review'])

    preprocess.preprocess_dataset(data_df, to_numerical='le', norm_technique='z-score',
                                  exclude_norm_cols=['number_of_reviews', 'last_review', 'reviews_per_month'])
    print(data_df.isnull().values.any())
    data = data_df.to_numpy()

    # train_test_split()
    # print(data_df.head())
