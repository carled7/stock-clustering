import requests
import csv
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
              
def build_data_set(tickers_df):
    with open("out/output.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # write header
        writer.writerow(['ticker', 'av_close', 
                     'av_volume', 'av_dividend_amount',
                     'highest', 'lowest',
                     'difference'
                     ])
        
        for index, row in tickers_df.iterrows():
            
            path = (f'data/{row["ticker"]}.SAO.csv')
            
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                writer.writerow([
                                    row['ticker'], 
                                    df['close'].mean(), 
                                    df['volume'].mean(), 
                                    df['dividend_amount'].mean(),
                                    df['close'].max(),
                                    df['close'].min(),
                                    ]) 
            
def download_single_stock(ticker_name):
    ticker = (f'{ticker_name}.SAO')
    path = (f'data/{ticker}.csv')

    key = 'XLSA5LZF5KUUXJX3'

    if not os.path.exists(path):
        url = (f'https://www.alphavantage.co/query?apikey={key}&function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&datatype=csv&outputsize=full')

        response = requests.get(url)

        if response.status_code == 200:
            content = response.content.decode()
            
            reader = csv.reader(content.splitlines(), delimiter=',')
            
            dataset = list(reader)
            
            with open(path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(dataset)
                
            print(f"New .csv downloaded for {ticker}: {response.status_code}")
        else:
            print(f"Request failed for {ticker}: {response.status_code}")

def train_kmeans_model(final_df):
    
    df = pd.read_csv('out/output.csv')

    del df['ticker']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    X = scaled_df[['av_close', 'av_volume', 'av_dividend_amount', 'highest', 'lowest']]
    kmeans = KMeans(n_clusters=5)  
    kmeans.fit(X)

    return kmeans
    
def main(ticker_name):
    
    tickers_df = pd.read_csv('in/TICKERS.csv')
    build_data_set(tickers_df)
    
    final_df = pd.read_csv('out/output.csv')
    kmeans = train_kmeans_model(final_df)
    
    download_single_stock(ticker_name)
    
    cluster_new_stock = pd.read_csv(f'data/{ticker_name}.SAO.csv')
    
    pred_cluster = kmeans.predict(
                    [[
                        cluster_new_stock['close'].mean(),
                        cluster_new_stock['volume'].mean(),
                        cluster_new_stock['dividend_amount'].mean(),
                        cluster_new_stock['close'].max(),
                        cluster_new_stock['close'].max(),
                        ]])
    
    
    custom_colors = ['#8B00FF','#00FFFF','#00FF00', '#FF7F00', '#FF0000']  

    custom_cmap = mcolors.ListedColormap(custom_colors)
    
        
    legend_elements = []
    for label, color in zip(set(kmeans.labels_), ['#8B00FF','#00FFFF','#00FF00', '#FF7F00', '#FF0000']):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                        markerfacecolor=color, markersize=10))
        
    plt.scatter(final_df['av_close'], final_df['av_volume'], c=kmeans.labels_, cmap=custom_cmap)
    plt.scatter(cluster_new_stock['close'].mean(), cluster_new_stock['volume'].mean(), color = 'black', label=(f'{ticker_name}'))
    
    plt.legend(loc='upper right', handles=legend_elements)
    
    plt.title('av_volume x av_close')
    
    plt.xlabel('av_close',)
    plt.ylabel('av_volume')
    
    plt.savefig('static/chart.jpg', format='jpg')
    plt.cla()
    
    plt.scatter(final_df['av_close'], final_df['highest'], c=kmeans.labels_, cmap=custom_cmap)
    plt.scatter(cluster_new_stock['close'].mean(), cluster_new_stock['close'].max(), color = 'black', label=(f'{ticker_name}'))
    
    plt.legend(loc='upper right', handles=legend_elements)
    
    plt.title('highest x av_close')
    
    plt.xlabel('av_close',)
    plt.ylabel('highest')
    
    plt.savefig('static/chart2.jpg', format='jpg')
    plt.cla()
    
    plt.scatter(final_df['av_close'], final_df['lowest'], c=kmeans.labels_, cmap=custom_cmap)
    plt.scatter(cluster_new_stock['close'].mean(), cluster_new_stock['close'].min(), color = 'black', label=(f'{ticker_name}'))
    
    plt.legend(loc='upper right', handles=legend_elements)
    
    plt.title('lowest x av_close')
    
    plt.xlabel('av_close',)
    plt.ylabel('lowest')
    
    plt.savefig('static/chart3.jpg', format='jpg')
    
    
    return pred_cluster
