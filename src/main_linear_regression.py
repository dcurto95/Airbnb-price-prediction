import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import data_exploration
import preprocess
import regression

if __name__ == '__main__':
    data_df = pd.read_csv("../data/AB_NYC_2019_cleaned.csv")

    data_exploration.show_missing_data(data_df)
    # data_df = data_df.drop(columns=data_df.columns[:4])
    data_df = data_df.drop(columns=['neighbourhood'])
    wrong_price_ids = np.where(data_df['price'] == 0)[0]
    data_df = data_df.drop(wrong_price_ids)
    # data_exploration.show_data_exploration(data_df)
    # data_df['last_review'] = pd.to_datetime(data_df['last_review'])

    # data_df = preprocess.clean_dataframe(data_df)
    # data_df.to_csv(r'../data/AB_NYC_2019_cleaned.csv', index=False)

    data_df = preprocess.preprocess_dataset(data_df, norm_technique='z-score')  # ,
    # exclude_norm_cols=['number_of_reviews', 'last_review', 'reviews_per_month'])

    # data_exploration.show_missing_data(data_df)
    # data_exploration.plot_correlation(data_df)
    # plt.show()
    # print(data_df.isnull().values.any())
    # print(data_df.columns)
    # data = data_df.to_numpy()
    data_df_y = data_df['price']
    data_df_x = data_df.drop(columns=['price'])
    data_x = data_df_x.to_numpy()
    data_y = data_df_y.to_numpy()

    # data_x = PCA().fit_transform(data_x)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)

    print("\nLinear regression:")
    since = time.time()
    mse_lr, r2_lr = regression.linear_regression(x_train, y_train, x_test, y_test)
    print('\tMean squared error linear regression: %.2f' % mse_lr)
    print('\tCoefficient of determination linear regression: %.2f' % r2_lr)
    print("\tExecution time:", time.time() - since, "s")

    print("\nSVR:")
    since = time.time()
    mse_svr, r2_svr = regression.svr(x_train, y_train, x_test, y_test)
    print('\tMean squared error SVR: %.2f' % mse_svr)
    print('\tCoefficient of determination SVR: %.2f' % r2_svr)
    print("\tExecution time:", time.time() - since, "s")

    # print(data_df.head())
