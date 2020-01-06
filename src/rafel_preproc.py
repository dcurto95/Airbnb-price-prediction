import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import preprocess

data_df = pd.read_csv('../data/AB_NYC_2019.csv')
print(data_df.shape)

data_df = data_df.drop(columns=data_df.columns[:4])
data_df.loc[(data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood_group'] = data_df.loc[
    (data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood']
data_df = data_df.drop(columns=['neighbourhood'])
wrong_price_ids = np.where(data_df['price'] == 0)[0]
data_df = data_df.drop(wrong_price_ids)
data_df['last_review'] = pd.to_datetime(data_df['last_review'])
data_df = preprocess.clean_dataframe(data_df)
# data_df.to_csv(r'../data/AB_NYC_2019_cleaned.csv', index=False)

logged_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
                'availability_365']
for col in logged_cols:
    data_df["log_(" + col + " +1)"] = np.log(data_df[col] + 1)
data_df['log_price'] = np.log(data_df.price)
data_df.drop(columns=logged_cols + ['price'], inplace=True)

print(data_df.shape)

neighbourhoods = data_df.neighbourhood_group.unique().tolist()

df_without_outliers = pd.DataFrame()
for neighborhood in neighbourhoods:
    a = preprocess.removal_of_outliers(data_df, neighborhood, 1.5)
    df_without_outliers = df_without_outliers.append(a)

data_df = df_without_outliers

data_df, new_features = preprocess.preprocess_dataset(data_df, 'log_price', norm_technique='z-score')

target = data_df['log_price'].to_numpy()

features = data_df.drop(['log_price'], axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=1)

print("\nLinear regression:")

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.scatter(y_pred, y_test)

y_pred[np.where(y_pred < 0)[0]] = 0
mse_reg = mean_squared_error(np.exp(y_test), np.exp(y_pred))
r2_reg = r2_score(y_test, y_pred)

print('\tMean squared error linear regression: %.2f' % mse_reg)
print('\tCoefficient of determination linear regression: %.2f' % r2_reg)

print("\nMulti Layer Perceptron:")
mlp = MLPRegressor(activation='relu', max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
predictions[np.where(predictions < 0)[0]] = 0

mse_mlp = mean_squared_error(np.exp(y_test), np.exp(predictions))
r2_mlp = r2_score(y_test, predictions)

print('\tMean squared error linear regression: %.2f' % mse_mlp)
print('\tCoefficient of determination linear regression: %.2f' % r2_mlp)

plt.show()
