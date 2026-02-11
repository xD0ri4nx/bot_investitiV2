import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import warnings
import os

# ---- CONFIGURATION ----
# 1. GPU Setup: Allow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

st.set_page_config(page_title="Pro-Grade AI Backtest", layout="wide")
st.title("Pro-Grade Multivariate AI (Attention + Bi-LSTM)")

st.markdown("""
### The "Formula 1" Upgrade
This model uses **Attention Mechanisms** to focus on critical market events. 
It analyzes **Price, Volatility (ATR), Momentum (RSI), Volume (OBV), and Market Velocity (Log Returns)**.
""")

# ---- CUSTOM ATTENTION LAYER ----
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# ---- SIDEBAR ----
st.sidebar.header("Configuration")
tickers = ["BTC-USD", "ETH-USD", "AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "NVDA", "META", "SPY"]
symbol = st.sidebar.selectbox("Asset:", tickers)

# Default to a range that guarantees data
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01").date())
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today").date())

st.sidebar.subheader("Hyperparameters")
epochs = st.sidebar.slider("Training Epochs", 20, 200, 100) # Increased default
time_step = st.sidebar.slider("Lookback Window (Days)", 15, 90, 60)

# ---- HELPER FUNCTIONS ----

def add_technical_indicators(df):
    """Adds EMA, RSI, MACD, Bollinger Bands, ATR, OBV, and Log Returns."""
    # 1. EMA (Exponential Moving Average)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # 4. Bollinger Bands (Width)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

    # 5. ATR (Volatility)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 6. OBV (Volume Pressure)
    df['Volume'] = df['Volume'].fillna(0)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # 7. Log Returns (Market Velocity) - NEW
    # Helps the model understand the "Speed" of price changes, not just the price itself.
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # 8. Cyclical Date Encoding (Seasonality) - NEW
    # Helps the model learn patterns like "Fridays are usually red"
    day = df.index.dayofweek
    df['Day_Sin'] = np.sin(2 * np.pi * day / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * day / 7)
    
    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    return df

def create_multivariate_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :]) 
        Y.append(dataset[i + time_step, 0]) # 0 is 'Close' price
    return np.array(X), np.array(Y)

# ---- MAIN APP LOGIC ----

if st.button("Run Advanced Simulation"):
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner(f"Acquiring & Engineering Data for {symbol}..."):
        # 1. Download Data
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(df) == 0:
                st.error(f"No data found for {symbol}.")
                st.stop()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                if "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
                else: st.error("No Close data."); st.stop()
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()
        
        # 2. Feature Engineering
        df = add_technical_indicators(df)
        
        if len(df) < (time_step + 50):
            st.error(f"Not enough data. Need {time_step + 50} points."); st.stop()

        # Select Features (Added new ones)
        feature_cols = ['Close', 'EMA_20', 'RSI', 'MACD', 'BB_Width', 'ATR', 'OBV', 'Log_Returns', 'Day_Sin', 'Day_Cos']
        target_col = 'Close'
        
        dataset = df[feature_cols].values
        target = df[[target_col]].values 

        # 3. Data Splitting
        training_size = int(len(dataset) * 0.80)
        
        scaler_X = RobustScaler()
        scaler_Y = RobustScaler()

        train_data = dataset[:training_size]
        scaler_X.fit(train_data)
        scaler_Y.fit(target[:training_size])

        test_data_raw = dataset[training_size - time_step:]
        train_data_scaled = scaler_X.transform(train_data)
        test_data_scaled = scaler_X.transform(test_data_raw)
        
        X_train, y_train = create_multivariate_dataset(train_data_scaled, time_step)
        X_test, y_test = create_multivariate_dataset(test_data_scaled, time_step)

        n_features = X_train.shape[2]

    # ---- MODEL BUILDING ----
    st.text("Training Neural Network on GPU (with Attention & Scheduler)...")
    
    model = Sequential()
    model.add(Input(shape=(time_step, n_features)))
    
    # Bi-LSTM with Regularization
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    
    # Attention Mechanism
    model.add(Attention())
    
    # Dense Head with GELU (Faster/Better than ReLU for Transformers/Attention)
    model.add(Dense(64, activation='gelu'))
    model.add(Dropout(0.2))
    model.add(Dense(1)) 

    model.compile(optimizer='adam', loss='huber')

    # ---- CALLBACKS (The "Gearbox") ----
    # ReduceLROnPlateau: If error doesn't drop for 5 epochs, slow down learning by 50%
    lr_reduction = ReduceLROnPlateau(monitor='loss', patience=5, verbose=0, factor=0.5, min_lr=0.00001)
    early_stop = EarlyStopping(patience=15, restore_best_weights=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Training Epoch {epoch + 1}/{epochs} | Loss: {logs['loss']:.5f}")

    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[StreamlitCallback(), early_stop, lr_reduction], # Added lr_reduction
        verbose=0
    )
    
    status_text.text("Training Complete!")
    progress_bar.progress(100)

    # ---- VISUALIZATION ----
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    y_test_real = scaler_Y.inverse_transform(y_test.reshape(-1, 1))
    test_predict_real = scaler_Y.inverse_transform(test_predict)

    test_indices = df.index[training_size:]
    min_len = min(len(test_indices), len(y_test_real))
    
    results = pd.DataFrame({
        "Actual": y_test_real[:min_len].flatten(),
        "Predicted": test_predict_real[:min_len].flatten()
    }, index=test_indices[:min_len])

    results['Error'] = np.abs(results['Actual'] - results['Predicted'])
    mae = results['Error'].mean()
    mape = (results['Error'] / results['Actual']).mean() * 100

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "Bi-LSTM + Attention")
    col2.metric("MAE (Avg Error)", f"${mae:.2f}")
    col3.metric("MAPE (Accuracy)", f"{100 - mape:.1f}%")

    st.subheader(f"Prediction vs Reality: {symbol}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[training_size-200:training_size], df['Close'][training_size-200:training_size], label="History (Train)", color='gray', alpha=0.4)
    ax.plot(results.index, results['Actual'], label="Actual Price", color='#00CC96', linewidth=2)
    ax.plot(results.index, results['Predicted'], label="AI Prediction", color='#FF5733', linewidth=2, linestyle='--')
    ax.set_title(f"AI Forecast Performance: {symbol}")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    with st.expander("See Raw Data & Export"):
        st.dataframe(results.tail(20))
        csv = results.to_csv().encode('utf-8')
        st.download_button(label="Download Data as CSV", data=csv, file_name=f'{symbol}_AI_Predictions.csv', mime='text/csv')
        
# Windows
# source .venv/bin/activate
# streamlit run app_interactiv.py