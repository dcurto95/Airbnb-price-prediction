import time
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('../data/AB_NYC_2019.csv')
print(df.shape)

# Fill Missing Values
df[["last_review", "reviews_per_month"]] = df[["last_review", "reviews_per_month"]].fillna(0)

# if there is no host name or listing name fill in None
df[["name", "host_name"]] = df[["name", "host_name"]].fillna("None")

# Only Turistic
df = df[df.price != 0].copy()
# df = df[df["minimum_nights"] <= 31].copy()

# LOGS
num_cols_log = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
for num in num_cols_log:
    df["log_("+num+" +1)"] = np.log(df[num]+1)
df['log_price'] = np.log(df.price)
df.drop(columns=num_cols_log + ['price'], inplace=True)

print(df.shape)

boroughs = df.neighbourhood_group.unique().tolist()

df = pd.concat([df, pd.get_dummies(df['neighbourhood'], drop_first=False)], axis=1)
df.drop(columns=['neighbourhood'], inplace=True)


def removal_of_outliers(df, room_t, nhood, distance):
    '''Function removes outliers that are above 3rd quartile and below 1st quartile'''
    '''The exact cutoff distance above and below can be adjusted'''

    new_piece = df[(df["room_type"] == room_t) & (df["neighbourhood_group"] == nhood)]["log_price"]
    # defining quartiles and interquartile range
    q1 = new_piece.quantile(0.25)
    q3 = new_piece.quantile(0.75)
    IQR = q3 - q1

    trimmed = df[(df.room_type == room_t) & (df["neighbourhood_group"] == nhood) & (df.log_price > (q1 - distance * IQR)) & (df.log_price < (q3 + distance * IQR))]
    return trimmed


def removal_of_outliers2(df, nhood, distance):
    '''Function removes outliers that are above 3rd quartile and below 1st quartile'''
    '''The exact cutoff distance above and below can be adjusted'''

    new_piece = df[df["neighbourhood_group"] == nhood]["log_price"]
    # defining quartiles and interquartile range
    q1 = new_piece.quantile(0.25)
    q3 = new_piece.quantile(0.75)
    IQR = q3 - q1

    trimmed = df[(df["neighbourhood_group"] == nhood) & (df.log_price > (q1 - distance * IQR)) & (df.log_price < (q3 + distance * IQR))]
    return trimmed


# apply the function
df_private = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Private room", neighborhood, 3)
    df_private = df_private.append(a)

df_shared = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Shared room", neighborhood, 3)
    df_shared = df_shared.append(a)

df_apt = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers(df, "Entire home/apt", neighborhood, 3)
    df_apt = df_apt.append(a)

df_nei = pd.DataFrame()
for neighborhood in boroughs:
    a = removal_of_outliers2(df, neighborhood, 1.5)
    df_nei = df_nei.append(a)

# Create new dataframe to absorb newly produced data
# df = pd.DataFrame()
# df = df.append([df_private, df_shared, df_apt])
df = df_nei.copy()

df = pd.concat([df, pd.get_dummies(df["room_type"], drop_first=False)], axis=1)
df.drop(['room_type'], axis=1, inplace=True)


# import datetime as dt
# df["last_review"] = pd.to_datetime(df["last_review"])
# df["last_review"] = df["last_review"].apply(lambda x: dt.datetime(2019, 7, 8)-x)
# df["last_review"] = df["last_review"].dt.days.astype("int").replace(18085, 1900)
#

# def date_replacement(date):
#     if date <= 3:
#         return "Last_review_last_three_day"
#     elif date <= 7:
#         return "Last_review_last_week"
#     elif date <= 30:
#         return "Last_review_last_month"
#     elif date <= 183:
#         return "Last_review_last_half_year"
#     elif date <= 365:
#         return "Last_review_last year"
#     elif date <= 1825:
#         return "Last_review_last_5_years"
#     else:
#         return "Last_review_never"


# df["last_review"] = df["last_review"].apply(lambda x: date_replacement(x))

# df = pd.concat([df, pd.get_dummies(df["last_review"], drop_first=False)], axis=1)
df.drop(["last_review"], axis=1, inplace=True)

df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'], drop_first=False)], axis=1)
df.drop(['neighbourhood_group'], axis=1, inplace=True)

df = df.drop(['id', 'name', 'host_id', 'host_name'], axis=1).copy()

print(df.shape)

num_cols = ['latitude', 'longitude', 'log_(minimum_nights +1)', 'log_(number_of_reviews +1)', 'log_(reviews_per_month +1)', 'log_(calculated_host_listings_count +1)', 'log_(availability_365 +1)']
for c in num_cols:
    df[c + '_norm'] = (df[c] - df[c].mean()) / df[c].std()
    df.drop(columns=c, inplace=True)


target = df['log_price'].copy().to_numpy()

features = df.drop(['log_price'], axis=1).copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=1)


print("\nLinear regression:")

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse_reg = mean_squared_error(np.exp(y_test), np.exp(y_pred))
r2_reg = r2_score(y_test, np.maximum(y_pred, 3))

plt.scatter(np.maximum(y_pred, 3), y_test)
plt.show()

print('\tMean squared error linear regression: %.2f' % mse_reg)
print('\tCoefficient of determination linear regression: %.2f' % r2_reg)

print("\nMulti Layer Perceptron:")
mlp = MLPRegressor(activation='relu', max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

mse_mlp = mean_squared_error(np.exp(y_test), np.exp(predictions))
r2_mlp = r2_score(y_test, predictions)

print('\tMean squared error linear regression: %.2f' % mse_mlp)
print('\tCoefficient of determination linear regression: %.2f' % r2_mlp)

# for c in features.columns:
#     features_red = features.drop(columns=[c]).copy()
#
#     X_train, X_test, y_train, y_test = train_test_split(features_red, target, test_size=0.20, random_state=1)
#
#     mlp.fit(X_train, y_train)
#     predictions = mlp.predict(X_test)
#
#     mse = np.mean((np.exp(predictions) - np.exp(y_test))**2)
#
#     print("Mean square error dropping column " + str(c) + ": " + str(mse))
