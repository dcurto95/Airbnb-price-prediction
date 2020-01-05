from torch.utils.data import Dataset
import torch
import os
import pandas as pd

import preprocess


class AIRBNB(Dataset):

    """
    """

    def __init__(self, path='../data/', data_set='AB_NYC_2019_cleaned.csv', transform=None):

        data_df = pd.read_csv(os.path.join(path, data_set))

        data_df, new_feature_cols = preprocess.preprocess_dataset(data_df, norm_technique='z-score')
        self.data = data_df.to_numpy()
        self.n_features = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index, :-1]), torch.tensor(self.data[index, -1])

    def get_n_features(self):
        return self.n_features
