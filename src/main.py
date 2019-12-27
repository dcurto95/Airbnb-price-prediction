import pandas as pd
from sklearn.model_selection import train_test_split

import preprocess

if __name__ == '__main__':
    data_df = pd.read_csv("../data/AB_NYC_2019.csv")

    preprocess.preprocess_dataset(data_df, to_numerical='oh')
    print(data_df.isnull().values.any())
    data = data_df.to_numpy()

    #train_test_split()
    #print(data_df.head())
