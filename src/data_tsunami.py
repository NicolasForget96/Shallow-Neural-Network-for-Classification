import sys
sys.path.append('src')

import kagglehub
from os import listdir
from os.path import join
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import geodatasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from splitter import transform_to_numpy

def load_data_tsunami(as_numpy=True):
    # Download latest dataset version
    folder = kagglehub.dataset_download("ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset")
    file = listdir(folder)[0]
    path = join(folder, file)
    print("\nPath to dataset files:", path, "\n")

    # Read data
    data = pd.read_csv(path)

    # Get features and target columns names
    features = ["longitude", "latitude", "depth", "magnitude"] 
    target = 'tsunami'

    # ---------------------------
    # Features/observations modifications

    # All Year < 2013 are non-tsunami => restrict to 2013 and after to focus on other factors
    data = data[data['Year'] >= 2013]
    
    # A gap in the middle values of longitude was visible => added square of longitude to cut around
    #data['longitude_sqr'] = data['longitude']**2
    #features.append('longitude_sqr')

    #data = data.iloc[0:100]

    if as_numpy:
        return transform_to_numpy(data, features, target)
    else:
        return data, features, target

def plot_world_map(data):
    geometry = [Point(z) for z in zip(data['longitude'], data['latitude'])]
    gdf = GeoDataFrame(data, geometry=geometry)
    world = gpd.read_file(geodatasets.data.naturalearth.land['url'])

    fig, ax = plt.subplots(figsize=(10, 6))
    world.plot(ax=ax, color='lightgray')
    gdf[gdf['tsunami'] == 0].plot(ax=ax, marker='o', color='black', markersize=15, label='No Tsunami')
    gdf[gdf['tsunami'] == 1].plot(ax=ax, marker='o', color='red', markersize=15, label='Tsunami')

    plt.axis('off')
    plt.title('Recorded occurences of earthquakes in the world between 2001 and 2022')
    plt.legend()
    plt.show()

def plot_features_vs_target(data, features, target):
    col = np.append(features, target)
    sns.pairplot(data[col], hue=target)
    plt.show()

if __name__ == "__main__":
    data, features, target = load_data_tsunami(as_numpy=False)
    data.info()
    print(data.head())

    nb_1 = sum(data[target])
    pct_1 = 100 * nb_1 / data.shape[0]
    print(f'There are {pct_1:.2f}% of 1s ({nb_1} occurences)')

    plot_world_map(data)
    plot_features_vs_target(data, features, target)