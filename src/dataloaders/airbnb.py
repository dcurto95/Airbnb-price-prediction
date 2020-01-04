from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pandas as pd

import preprocess


class AIRBNB(Dataset):

    """
    """

    def __init__(self, path='../data/', data_set='AB_NYC_2019_cleaned.csv', transform=None):

        data_df = pd.read_csv(os.path.join(path, data_set))

        data_df = data_df.drop(columns=['neighbourhood'])
        wrong_price_ids = np.where(data_df['price'] == 0)[0]
        data_df = data_df.drop(wrong_price_ids)

        # Logs
        data_df['minimum_nights'] = np.log((data_df['minimum_nights']))
        data_df['number_of_reviews'] = np.log(data_df['number_of_reviews'] + 1)
        data_df['last_review'] = np.log(data_df['last_review'] + 1)
        data_df['reviews_per_month'] = np.log(data_df['reviews_per_month'] + 1)
        data_df['calculated_host_listings_count'] = np.log(data_df['calculated_host_listings_count'] + 1)
        data_df['availability_365'] = np.log(data_df['availability_365'] + 1)

        data_df = preprocess.preprocess_dataset(data_df, norm_technique='z-score')

        data_df_y = data_df['price']
        data_df_x = data_df.drop(columns=['price'])
        data_x = data_df_x.to_numpy()
        data_y = data_df_y.to_numpy()

        data_y = np.log(data_y)

        neighbourhood_cols = ['neighbourhood_group_Bronx', 'neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
                              'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island']
        neighbourhood_indexs = [list(data_df_x.columns).index(name) for name in neighbourhood_cols]
        for neighbourhood_ind in neighbourhood_indexs:
            x_train, y_train = preprocess.removal_of_price_outliers(data_x, data_y, data_x[:, neighbourhood_ind])


        # data_df = preprocess.preprocess_dataset(data_df, norm_technique='z-score')
        # self.data = data_df.to_numpy()
        self.features = x_train
        self.gt = y_train
        # self.n_features = self.data.shape[1] - 1
        self.n_features = self.features.shape[1]

    def __len__(self):
        # return self.data.shape[0]
        return self.gt.shape[0]

    def __getitem__(self, index):
        # return torch.FloatTensor(self.data[index, :-1].astype(np.float32)), torch.FloatTensor([self.data[index, -1].astype(np.float32)])
        return torch.FloatTensor(self.features[index].astype(np.float32)), torch.FloatTensor([self.gt[index].astype(np.float32)])

    def get_n_features(self):
        return self.n_features
