import pandas as pd
import numpy as np
from utils.influxdb_client import get_client
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve
)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def compute_features(df):
    """
    Compute features for trend classification.
    Args:
        df (pd.DataFrame): DataFrame containing the raw data.
    Returns:
        pd.DataFrame: DataFrame with computed features.
    """

    df['price_scaled'] = df.groupby('symbol')['price'].transform(
        lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).squeeze()
    )
    df['pct_change'] = df.groupby('symbol')['price'].pct_change()
    df['momentum'] = df['price'] - df['price'].shift(5)
    df['volatility'] = df['price'].rolling(window=5).std()
    df = df.dropna()

    return df

def label_trends(df, up_thresh=0.002, down_thresh=-0.002):
    """
    Label trends based on percentage change.
    Args:
        df (pd.DataFrame): DataFrame containing the raw data.
        up_thresh (float): Threshold for bullish trend.
        down_thresh (float): Threshold for bearish trend.
    Returns:
        pd.DataFrame: DataFrame with trend labels.
    """

    df['trend'] = 0
    df.loc[df['pct_change'] > up_thresh, 'trend'] = 1
    df.loc[df['pct_change'] < down_thresh, 'trend'] = -1
    df = df[df['trend'] != 0]  # remove neutral

    return df

def get_sequences(df, seq_len, feature_cols):
    """
    Generate sequences of features and labels for LSTM model.
    Args:
        df (pd.DataFrame): DataFrame containing the raw data.
        seq_len (int): Length of the sequences.
        feature_cols (list): List of feature columns.
    Returns:
        tuple: Tuple containing sequences, labels, symbols, and timestamps.
    """

    sequences, labels, symbols, timestamps = [], [], [], []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        values = symbol_df[feature_cols].values
        trend_labels = symbol_df['trend'].values
        for i in range(len(values) - seq_len):
            sequences.append(values[i:i + seq_len])
            labels.append(trend_labels[i + seq_len])
            symbols.append(symbol)
            timestamps.append(symbol_df.index[i + seq_len])

    return np.array(sequences), np.array(labels), np.array(symbols), np.array(timestamps)

def get_model(input_shape):
    """
    Create and compile the LSTM model.
    Args:
        input_shape (tuple): Shape of the input data.
    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """

    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def main():
    """
    Main function to fetch data, process it, train the model, and generate reports.
    """

    print("Fetching and processing data...")
    client = get_client()
    result = client.query('''SELECT symbol, price, time FROM "token_prices" 
                             WHERE time >= now() - INTERVAL '14 days' 
                               AND exchange = 'coinbase' 
                               AND currency = 'usd' 
                               AND symbol = 'btc' 
                             ORDER BY time ASC''')
    df = result.to_pandas()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    df = compute_features(df)
    df = label_trends(df)

    print("Class distribution:\n", df['trend'].value_counts(normalize=True))


    SEQ_LENGTH = 5
    FEATURE_COLS = ['price_scaled', 'pct_change', 'momentum', 'volatility']
    X, y, symbols, timestamps = get_sequences(df, SEQ_LENGTH, FEATURE_COLS)

    if X.shape[0] == 0:
        raise ValueError("No sequences generated. Try lowering SEQ_LENGTH or increasing the data window.")

    X = X.reshape(X.shape[0], SEQ_LENGTH, len(FEATURE_COLS))
    y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    timestamps_train, timestamps_test = train_test_split(timestamps, test_size=0.2, shuffle=False)

    class_weights_dict = dict(enumerate(class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train)))

    print("Training model...")
    model = get_model((SEQ_LENGTH, len(FEATURE_COLS)))
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict
    )

    print("Generating report...")
    y_pred = model.predict(X_test).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    report = pd.DataFrame({
        'timestamp': timestamps_test,
        'actual_trend': y_test,
        'predicted_trend': y_pred,
        'predicted_label': y_pred_binary
    })

    report.to_csv("trend_prediction_report.csv", index=False)
    print("Report saved to trend_prediction_report.csv")
    print("Accuracy:", accuracy_score(y_test, y_pred_binary))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bearish", "Bullish"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    client.close()

if __name__ == "__main__":
    main()
