from torch.utils.data import Dataset

import os
import pandas as pd

import preprocess


class AIRBNB(Dataset):

    """
    """

    def __init__(self, path='../data/', data_set='AB_NYC_2019_cleaned.csv', transform=None):

        data_df = pd.read_csv(os.path.join(path, data_set))

        preprocess.preprocess_dataset(data_df, to_numerical='oh', norm_technique='z-score')
        self.data = data_df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, -1]
