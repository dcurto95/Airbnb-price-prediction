import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def show_missing_data(data):
    total = data.isnull().sum()#.sort_values(ascending=False)
    percent = (100 * data.isnull().sum()) / data.isnull().count()
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False)
    print(missing_data)


def plot_correlation(data):
    corr_matrix = data.corr()
    plt.figure(figsize=(15, 6))
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    sns.heatmap(corr_matrix, annot=True, fmt='.1g', square=True, linewidths=0.1, cmap='Blues', cbar_kws= {'orientation': 'vertical'}, cbar=False)
    #plt.xticks(rotation=45)
    #plt.title("Correlation matrix")
    plt.tight_layout()


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
    # initializing the figure size
    f  = plt.figure(figsize=(10, 8))
    ax = f.gca()
    nyc_img = plt.imread('../data/New_York_City_.png', 0)
    lat_long_subset_data = data[['latitude', 'longitude', 'price']].drop_duplicates()
    # scaling the image based on the latitude and longitude max and mins for proper output
    data = data[data.price <500]
    ax.imshow(nyc_img, zorder=0, extent=[-74.258, -73.69, 40.49, 40.92])
    s = ax.scatter(x= data['longitude'], y =data['latitude'], c=data['price'], alpha=0.5, zorder=5, s=10)
    cb = plt.colorbar(s)
    cb.set_label('Price')
    plt.xlabel('Longitude')
    plt.ylabel('Latidude')
    # using scatterplot again
    # plt.plot(kind='scatter', x='longitude', y='latitude', c='price', ax=ax,
    #            cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)
    plt.legend()
    plt.show()

def plot_count_neigbourhood_type(data):
    plt.figure(figsize=(15, 6))
    #sns.countplot(data=data, x='neighbourhood_group', hue='room_type', palette='GnBu_d')
    sns.barplot(x='neighbourhood_group', hue='room_type', y='price', data=data, palette='GnBu_d')
    plt.title('Counts of airbnb listings by neighbourhood group and room type', fontsize=15)
    plt.xlabel('Neighbourhood group')
    plt.ylabel("Price")
    plt.legend(frameon=False, fontsize=12)


def plot_price_distribution(data):
    plt.figure(figsize=(15, 6))
    sns.violinplot(data=data[data.price < 500], x='neighbourhood_group', y='price',  palette='GnBu_d')
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

def plot_data_distribution(data, columns):
    data.hist(grid=False,  figsize=(12, 8))

def show_statistical_information(data, columns):

    for i in columns:
        curr_col = data[i]
        print("---------", i ,"----------")
        print("Mean: ", np.exp(data[i]).mean())
        print("Std: ", np.exp(data[i]).std())





def show_data_exploration(data):
    # TODO: Utilizar mateixa paleta de color + titols a totes les figures
    #plot_correlation(data)
    plot_count_neigbourhood_type(data)
    plot_location_distribution(data)
    plot_location_price_distribution(data)
    plot_price_distribution(data)
    plot_most_popular_neighbourhood(data)
    plt.show()
