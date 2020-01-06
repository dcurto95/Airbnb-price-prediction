from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pandas as pd


class AIRBNB(Dataset):

    """
    """

    def __init__(self, path='../data/', data_set='AB_NYC_2019_cleaned.csv', transform=None):

        data_df = pd.read_csv(os.path.join(path, data_set))

        self.targets = data_df['price'].to_numpy()
        self.features = data_df.drop(columns=['price']).to_numpy()
        self.n_features = self.features.shape[1]

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return torch.FloatTensor(self.features[index].astype(np.float32)), torch.FloatTensor([self.targets[index].astype(np.float32)])

    def get_n_features(self):
        return self.n_features
