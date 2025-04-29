"""
This script will perform anomaly detection amongst digital asset prices using Z-Score.

Usage: python anomaly_detection.py
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.influxdb_client import get_client
from datetime import datetime, timedelta
from influxdb_client_3 import InfluxDBClient3

def get_raw_data(
        client: InfluxDBClient3,
        symbol: str,
        interval: str = '1d',
        exchange: str = 'coinbase',
        currency: str = 'usd',
    ) -> pd.DataFrame:
    """
    Fetch raw data from InfluxDB.
    
    Args:
        client: InfluxDB client instance.
        
    Returns:
        Arrow table containing the raw data.
    """

    result = client.query(
    f'''SELECT
        symbol,
        price,
        time
      FROM "token_prices"
      WHERE time >= now() - INTERVAL '{interval}' 
        AND exchange = '{exchange}'
        AND currency = '{currency}'
        AND symbol = '{symbol}'
        ORDER BY symbol, time DESC
      ''')
    print(f"Fetched {len(result)} rows of data for symbol {symbol} from InfluxDB.")
    return result

def resample_data(df: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    """
    Resample the data to a specified frequency.
    
    Args:
        df: DataFrame containing the raw data.
        
    Returns:
        Resampled DataFrame.
    """
    # Creates a copy for manipulation
    df = df.copy()

    # Convert the time column to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Set the time column as the index for time-based filtering and aggregation
    df.set_index('time', inplace=True)

    # Resample the data to the specified interval
    df = df[['price']].resample(interval).mean().interpolate()
 
    return df

def detect_price_anomalies(df: pd.DataFrame, window: int = 30, threshold: float = 5) -> pd.DataFrame:
    """
    Detect anomalies in the price data using Z-Score.
    
    Args:
        df: DataFrame containing the raw data.
        
    Returns:
        DataFrame with anomalies marked.
    """
    # Creates a copy for manipulation
    df = df.copy()
    df['median'] = df['price'].rolling(window=window, min_periods=1).median()
    df['mad'] = df['price'].rolling(window=window, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    df['mad'] = df['mad'].replace(0, 1e-6)
    df['z_score'] = 0.6745 * (df['price'] - df['median']) / df['mad']
    df['anomaly'] = df['z_score'].abs() > threshold

    return df

def plot_price_with_anomalies(df: pd.DataFrame, symbol) -> None:
    """
    Plot the price data with anomalies highlighted.

    Args:
        df: DataFrame containing the raw data.
        for_paper: If True, generates a version optimized for paper (no legend, no stat box).
    """
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    plt.plot(df.index, df['price'], label='Price', color='blue')
    anomalies = df[df['anomaly']]

    plt.scatter(
        anomalies.index,
        anomalies['price'],
        s=250,
        edgecolor='white',
        marker='o',
        alpha=0.9,
        zorder=1,
    )

    # Highlight the anomalies
    cluster_start = None
    highlighted_clusters = []
    for i in range(1, len(df)):
        if df['anomaly'].iloc[i] and df['anomaly'].iloc[i-1]:
            if cluster_start is None:
                cluster_start = df.index[i-1]

        elif cluster_start:
            cluster_end = df.index[i-1]
            if (cluster_end - cluster_start).seconds > 180:
                plt.axvspan(cluster_start, cluster_end, color='orange', alpha=0.20, zorder=0)
                highlighted_clusters.append((cluster_start, cluster_end))
            cluster_start = None

    up = anomalies[anomalies['z_score'] > 0]
    down = anomalies[anomalies['z_score'] < 0]

    # Highlight up anomalies
    plt.scatter(
        up.index,
        up['price'],
        c='green',
        s=150,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.7,
        zorder=2,
        label='Anomalies (Up)',
    )

    # Highlight down anomalies
    plt.scatter(
        down.index,
        down['price'],
        c='red',
        s=150,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.7,
        zorder=2,
        label='Anomalies (Down)',
    )

    # Highlight largest anomaly
    if not anomalies.empty:
        max_anomaly = anomalies.loc[anomalies['z_score'].idxmax()]
        plt.scatter(
            [max_anomaly.name],
            [max_anomaly['price']],
            c='yellow',
            s=300,
            edgecolor='black',
            alpha=0.9,
            marker='*',
            linewidth=1.2,
            zorder=3,
            label='Largest Anomaly',
        )
    
    # Add legend
    plt.title(f'{symbol.upper()} Prices with Anomalies', fontsize=14)
    plt.xlabel('Time (UTC)', fontsize=12)
    plt.ylabel('Price', fontsize=12)

    # Customize display
    plt.grid()
    ymin, ymax = plt.ylim()
    padding = (ymax - ymin) * 0.05
    plt.ylim(ymin - padding, ymax + padding)
    plt.tight_layout()
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plots in different formats
    plt.savefig(f'price_anomalies.svg', format='svg', dpi=300)
    plt.savefig(f'price_anomalies.png', format='png', dpi=300)
    plt.show()
    plt.close()

def output_results(df: pd.DataFrame, symbol, window: int, threshold: int) -> None:
    """
    Output the results of the anomaly detection.
    
    Args:
        df: DataFrame containing the raw data.
    """

    print('Sample anomalies detected:')
    anomalies = df[df['anomaly']]
    print(anomalies[['price', 'z_score']].head())

    num_anomalies = df['anomaly'].sum()
    print(f'Total anomalies detected: {num_anomalies}')

    anomaly_times = df[df['anomaly']].index
    print('Anomaly times:')
    for time in anomaly_times:
        print(time)
    
def main() -> None:
    print('Finding anomalies from pricing data...')
    symbol = 'btc'
    window = 60
    threshold = 3.5

    # Get the InfluxDB client
    client = get_client()

    # Fetch the raw data
    arrow_table = get_raw_data(client, symbol)

    # Convert the Arrow table to a Pandas DataFrame
    df = arrow_table.to_pandas()

    # Resample the data
    df = resample_data(df)

    window = 60
    threshold = 3.5
    # Perform anomaly detection using Z-Score
    df = detect_price_anomalies(df, window, threshold)

    # Output results
    output_results(df, symbol, window, threshold)
    plot_price_with_anomalies(df, symbol) 

    # Close the InfluxDB client
    client.close()

if __name__ == '__main__':
    main()
