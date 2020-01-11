import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import preprocess

data_df = pd.read_csv('../data/AB_NYC_2019.csv')
print("Original shape", data_df.shape)

# Cleaning
data_df = preprocess.clean_dataframe(data_df)

# Transformations
data_df = preprocess.date_to_timestamp(data_df)

logged_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'last_review',
               'calculated_host_listings_count', 'availability_365']

for col in logged_cols:
    data_df[col] = np.log(data_df[col] + 1)
data_df['price'] = np.log(data_df.price)

print("Shape after cleaning and transformations", data_df.shape)

# Cleaning outliers
neighbourhoods = data_df.neighbourhood_group.unique().tolist()

df_without_outliers = pd.DataFrame()
for neighborhood in neighbourhoods:
    cleaned = preprocess.removal_of_outliers(data_df, 'neighbourhood_group', neighborhood, 'price', 1.5)
    df_without_outliers = df_without_outliers.append(cleaned)

data_df = df_without_outliers

# Change neighbourhood_group and neighbourhood
# data_df.loc[(data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood_group'] = data_df.loc[
#     (data_df['neighbourhood_group'] == 'Manhattan'), 'neighbourhood']
# data_df = data_df.drop(columns=['neighbourhood'])

# Normalize numerical and One-Hot categorical
data_df = preprocess.preprocess_dataset(data_df, norm_technique='z-score', exclude_norm_cols='price')

# Separate into X, y
target = data_df['price'].to_numpy()
features = data_df.drop(columns=['price']).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

# Prediction methods
print("\nLinear regression:")

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

y_pred[(y_pred < 0)] = 0
y_pred[(y_pred > 10)] = np.max(y_pred[(y_pred < 10)])

mse_reg = mean_squared_error(np.exp(y_test), np.exp(y_pred))
r2_reg = r2_score(y_test, y_pred)

print('\tMean squared error linear regression: %.2f' % mse_reg)
print('\tCoefficient of determination linear regression: %.2f' % r2_reg)
