import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import data_exploration
import plot
import preprocess
import regression
import mlp_regressor

if __name__ == '__main__':
    data_df = pd.read_csv("../data/AB_NYC_2019.csv")

    data_df["reviews_per_month"] = data_df["reviews_per_month"].fillna(0)

    # data_exploration.show_missing_data(data_df)
    data_df = data_df.drop(columns=['last_review', 'id', 'host_id', 'name', 'host_name'])
    # data_df.loc[(data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood_group'] = data_df.loc[(data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood']
    # data_df = data_df.drop(columns=['neighbourhood'])
    wrong_price_ids = np.where(data_df['price'] == 0)[0]
    data_df = data_df.drop(wrong_price_ids)
    # data_exploration.show_data_exploration(data_df)
    # data_df['last_review'] = pd.to_datetime(data_df['last_review'])

    # min_nights = np.where(data_df['minimum_nights'] > 30)[0]
    # data_df = data_df.drop(min_nights)

    # data_df = preprocess.clean_dataframe(data_df)
    # data_df.to_csv(r'../data/AB_NYC_2019_cleaned.csv', index=False)

    # Logs
    data_df['minimum_nights'] = np.log(data_df['minimum_nights'] + 1)
    data_df['number_of_reviews'] = np.log(data_df['number_of_reviews'] + 1)
    # data_df['last_review'] = np.log(data_df['last_review'] + 1)
    data_df['reviews_per_month'] = np.log(data_df['reviews_per_month'] + 1)
    data_df['calculated_host_listings_count'] = np.log(data_df['calculated_host_listings_count'] + 1)
    data_df['availability_365'] = np.log(data_df['availability_365'] + 1)

    # histograms = [data_df['minimum_nights'],
    #               data_df['number_of_reviews'],
    #               data_df['last_review'],
    #               data_df['reviews_per_month'],
    #               data_df['calculated_host_listings_count'],
    #               data_df['availability_365']]
    # titles = ['minimum_nights',
    #           'number_of_reviews',
    #           'last_review',
    #           'reviews_per_month',
    #           'calculated_host_listings_count',
    #           'availability_365']

    # plot.plot_histograms(histograms, titles)

    # plot.show_all()

    def removal_of_outliers2(df, nhood, distance):
        '''Function removes outliers that are above 3rd quartile and below 1st quartile'''
        '''The exact cutoff distance above and below can be adjusted'''

        new_piece = df[df["neighbourhood_group"] == nhood]["price"]
        # defining quartiles and interquartile range
        q1 = new_piece.quantile(0.25)
        q3 = new_piece.quantile(0.75)
        IQR = q3 - q1

        trimmed = df[(df["neighbourhood_group"] == nhood) & (df.price > (q1 - distance * IQR)) & (df.price < (q3 + distance * IQR))]
        return trimmed


    data_df['price'] = np.log(data_df['price'])
    neighbourhoods = data_df['neighbourhood_group'].unique().tolist()
    print("Before outliers", data_df.shape[0])
    df_nei = pd.DataFrame()
    for neighborhood in neighbourhoods:
        a = removal_of_outliers2(data_df, neighborhood, 1.5)
        df_nei = df_nei.append(a)

    data_df = df_nei.copy()
    print("After outliers", data_df.shape[0])

    # print("Before outliers", data_df_x.shape[0])
    # for neighbourhood in neighbourhoods:
    #     data_df_x, data_df_y = preprocess.removal_of_price_outliers(data_df_x, data_df_y, data_df_x[data_df_x['neighbourhood_group'] == neighbourhood])
    # print("After outliers", data_df_x.shape[0])

    data_df = preprocess.preprocess_dataset(data_df, norm_technique='z-score')  # ,
    # exclude_norm_cols=['number_of_reviews', 'last_review', 'reviews_per_month'])



    # data_df = pd.concat([data_df, pd.get_dummies(data_df['neighbourhood'], drop_first=False)], axis=1)
    # data_df.drop(columns=['neighbourhood'], inplace=True)
    #
    # data_df = pd.concat([data_df, pd.get_dummies(data_df["room_type"], drop_first=False)], axis=1)
    # data_df.drop(['room_type'], axis=1, inplace=True)
    #
    # data_df = pd.concat([data_df, pd.get_dummies(data_df['neighbourhood_group'], drop_first=False)], axis=1)
    # data_df.drop(['neighbourhood_group'], axis=1, inplace=True)
    #
    # num_cols = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    # for c in num_cols:
    #     data_df[c + '_norm'] = (data_df[c] - data_df[c].mean()) / data_df[c].std()
    #     data_df.drop(columns=c, inplace=True)




    # data_exploration.show_missing_data(data_df)
    # data_exploration.plot_correlation(data_df)
    # plt.show()
    # print(data_df.isnull().values.any())
    # print(data_df.columns)
    # data = data_df.to_numpy()

    # data_df['price'] = np.log((data_df['price']))


    # from rafel_preproc import preprocess_dataset
    # df2 = preprocess_dataset()
    # df2.drop(columns='Ford Wadsworth')

    # data_y = df2['log_price'].copy().to_numpy()
    #
    # data_x = df2.drop(['log_price'], axis=1).copy().to_numpy()
    #
    data_df_y = data_df['price']
    data_df_x = data_df.drop(columns=['price'])  # , 'minimum_nights', 'number_of_reviews', 'last_review', 'calculated_host_listings_count', 'availability_365'])

    # neighbourhood_cols = new_feature_cols[0]
    # neighbourhood_indexs = [list(data_df_x.columns).index(name) for name in neighbourhood_cols]

    data_x = data_df_x.to_numpy()
    data_y = data_df_y.to_numpy()
    print(data_x.shape)

    # data_x = PCA().fit_transform(data_x)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    # data_y = np.log(data_y)






    mlp_regressor.do_mlp_regressor(x_train, y_train, x_test, y_test)
    # print("\nLinear regression:")
    # since = time.time()
    # mse_lr, r2_lr = regression.linear_regression(x_train, y_train, x_test, y_test)
    # print('\tMean squared error linear regression: %.2f' % mse_lr)
    # print('\tCoefficient of determination linear regression: %.2f' % r2_lr)
    # print("\tExecution time:", time.time() - since, "s")
    #
    # print("\nSVR:")
    # since = time.time()
    # mse_svr, r2_svr = regression.svr(x_train, y_train, x_test, y_test)
    # print('\tMean squared error SVR: %.2f' % mse_svr)
    # print('\tCoefficient of determination SVR: %.2f' % r2_svr)
    # print("\tExecution time:", time.time() - since, "s")

    # print(data_df.head())
