"""
This script will perform correlation analysis.

Usage: python correlation_analysis.py
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from utils.influxdb_client import get_client

def get_raw_data(client):
    result = client.query(
     '''SELECT
        symbol,
        price,
        time
      FROM "token_prices"
      WHERE time >= now() - INTERVAL '3 day' 
        AND exchange = 'coinbase' 
        AND currency = 'usd' 
        AND symbol NOT IN ('usdc', 'usdt')
        ORDER BY symbol, time DESC
      ''')

    return result

def print_heatmap(pearson_correlation_matrix, spearman_correlation_matrix):
    correlation_diff = pearson_correlation_matrix - spearman_correlation_matrix 
    plt.figure(figsize=(12, 8))

    sns.heatmap(correlation_diff, annot=True, cmap='coolwarm', center=0)

    # highlight most divergent correlation
    max_diff = correlation_diff.abs().max().max()
    ax = plt.gca()
    for i in correlation_diff.index:
        for j in correlation_diff.columns:
            if np.isclose(abs(correlation_diff.loc[i, j]), max_diff):
                ax.add_patch(
                    plt.Rectangle((correlation_diff.columns.get_loc(j),
                                    correlation_diff.index.get_loc(i)),
                                    1, 
                                    1, 
                                    fill=False,
                                    edgecolor='black', 
                                    linewidth=3)
                )
    
    plt.suptitle('Difference in Correlation (Pearson vs Spearman)\nRed = Pearson > Spearman | Blue = Spearman > Pearson', 
                 fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(f'pearson_spearman_heatmap.svg',format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'pearson_spearman_heatmap.png',format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def print_scatterplot(df_pivot, x_asset, y_asset):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df_pivot[x_asset], y=df_pivot[y_asset])
    m, b = np.polyfit(df_pivot[x_asset], df_pivot[y_asset], 1)
    plt.plot(df_pivot[x_asset], m*df_pivot[x_asset] + b, color='red')
    
    x_asset = x_asset.upper()
    y_asset = y_asset.upper()
    plt.xlabel(x_asset, fontsize=12)
    plt.ylabel(y_asset, fontsize=12)
    plt.title(f'{x_asset} vs {y_asset} Price (USD)', fontsize=14)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{x_asset}_{y_asset}_scatterplot.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{x_asset}_{y_asset}_scatterplot.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def print_correlations(pearson_correlation_matrix, spearman_correlation_matrix):
    print('Pearson Correlation Matrix:')
    print(pearson_correlation_matrix)
    print('Spearman Correlation Matrix:')
    print(spearman_correlation_matrix)
    
def get_correlations(df_pivot):
    pearson_correlation_matrix = df_pivot.corr(method='pearson')
    spearman_correlation_matrix = df_pivot.corr(method='spearman') 

    return pearson_correlation_matrix, spearman_correlation_matrix

def get_pivot_table(df):
    # Preprocess data
    df['time'] = pd.to_datetime(df['time'])
    
    df_pivot = df.pivot(index='time', columns='symbol', values='price')

    # Interpolate missing values
    df_pivot = df_pivot.interpolate(method='linear').ffill().bfill()

    return df_pivot

def main():
    print('Finding correlations from pricing data...')
    client = get_client()
    arrow_table = get_raw_data(client)
    df = arrow_table.to_pandas()
    df_pivot = get_pivot_table(df)
    pearson_correlation_matrix, spearman_correlation_matrix = get_correlations(df_pivot)
    print_correlations(pearson_correlation_matrix, spearman_correlation_matrix)
    print_heatmap(pearson_correlation_matrix, spearman_correlation_matrix)
    print_scatterplot(df_pivot, 'link', 'sol')
    print_scatterplot(df_pivot, 'btc', 'xrp')
    print_scatterplot(df_pivot, 'btc', 'eth')
    client.close()

if __name__ == '__main__':
    main()
