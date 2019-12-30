import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px


def show_missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()) / data.isnull().count()
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False)
    print(missing_data)


def plot_correlation(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(15, 6))
    sns.heatmap(corr_matrix, annot=True, linewidths=0.1, cmap='Blues', cbar=False)
    plt.xticks(rotation=45)
    plt.title("Correlation matrix")
    plt.tight_layout()

    # sns.set(font_scale=0.8)
    # f, ax = plt.subplots(figsize=(15, 12))
    # sns.heatmap(corrmatrix, vmax=0.8, square=True)
    # sns.set(font_scale=0.8)


def plot_location_distribution(data):
    plt.figure(figsize=(15, 6))
    img = plt.imread('../data/New_York_City_.png', 0)
    coordenates_to_extent = [-74.258, -73.69, 40.49, 40.92]
    plt.imshow(img, zorder=0, extent=coordenates_to_extent)

    # Plotting
    scatter_map = sns.scatterplot(data.longitude, data.latitude, hue=data.neighbourhood_group, s=10)
    plt.grid(True)
    plt.legend(title='Neighbourhood Groups')
    plt.xlim(-74.258, -73.7)
    plt.ylim(40.49, 40.92)

def plot_location_price_distribution(data):
    plt.figure(figsize=(15, 6))
    img = plt.imread('../data/New_York_City_.png', 0)
    coordenates_to_extent = [-74.258, -73.69, 40.49, 40.92]
    plt.imshow(img, zorder=0, extent=coordenates_to_extent)

    lat_long_subset_data = data[['latitude', 'longitude', 'price']].drop_duplicates()
    # lat_long_subset_data = lat_long_subset_data.pivot_table('price',['latitude','longitude'],aggfunc='average').reset_index()
    fig = px.scatter(lat_long_subset_data, x="latitude", y="longitude", color='price' )
    fig.update_layout(
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        title='Manhattan LatLong vs Price Plot'
    )
    fig.show()

def plot_count_neigbourhood_type(data):
    plt.figure(figsize=(15, 6))
    sns.countplot(data=data, x='neighbourhood_group', hue='room_type', palette='GnBu_d')
    plt.title('Counts of airbnb in neighbourhoods with room type category', fontsize=15)
    plt.xlabel('Neighbourhood group')
    plt.ylabel("Count")
    plt.legend(frameon=False, fontsize=12)


def plot_price_distribution(data):
    plt.figure(figsize=(15, 6))
    sns.violinplot(data=data[data.price < 500], x='neighbourhood_group', y='price', palette='GnBu_d')
    plt.title('Density and distribution of prices for each neighbourhood group', fontsize=15)
    plt.xlabel('Neighbourhood group')
    plt.ylabel("Price")


def plot_most_popular_neighbourhood(data):
    neighbourhood = data.neighbourhood.value_counts()[:10]
    plt.figure(figsize=(12, 8))
    x = list(neighbourhood.index)
    y = list(neighbourhood.values)
    x.reverse()
    y.reverse()

    plt.title("Most Popular Neighbourhood")
    plt.ylabel("Neighbourhood Area")
    plt.xlabel("Number of guest Who host in this Area")
    plt.barh(x, y)


def show_data_exploration(data):
    # TODO: Utilizar mateixa paleta de color + titols a totes les figures
    plot_correlation(data)
    plot_count_neigbourhood_type(data)
    plot_location_distribution(data)
    plot_location_price_distribution(data)
    plot_price_distribution(data)
    plot_most_popular_neighbourhood(data)
    plt.show()
