import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Layer, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
import os
import gc

# ---- 1. SYSTEM CONFIGURATION ----
tf.random.set_seed(42)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Institutional Quant AI", layout="wide")
st.title("Institutional Quant AI: Trend-Boosted Edition")

st.markdown("""
### System Architecture
This version includes a Trend Booster to wake up the AI for strong stocks.
* **Signal Boosting:** Amplifies small moves so the AI actually trades them.
* **Trend Filter:** Doubles conviction if Price > 50-Day Moving Average.
* **Low Noise:** Reduced Gaussian interference for clearer signal detection.
""")

# ---- 2. CUSTOM LAYERS ----
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# ---- 3. QUANT PIPELINE ----
class QuantPipeline:
    def __init__(self, symbol, start_date, end_date, time_step):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.time_step = time_step
        self.scaler = RobustScaler()
        
    def fetch_data(self):
        try:
            df = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)
            if len(df) == 0: raise ValueError("No data found.")
            
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
            
            req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in req_cols):
                 if 'Adj Close' in df.columns: 
                     df['Close'] = df['Adj Close']
            
            return df
        except Exception as e:
            st.error(f"Data Feed Error: {e}")
            st.stop()

    def engineer_features(self, df):
        # 1. Log Returns
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Target Generation (Shifted T+1)
        df['Target'] = df['Log_Returns'].shift(-1)

        # 3. Garman-Klass Volatility
        log_hl = np.log(df['High'] / df['Low'])**2
        log_co = np.log(df['Close'] / df['Open'])**2
        df['Garman_Klass'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # 4. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. Chaikin Money Flow (CMF)
        high_low = df['High'] - df['Low']
        high_low = high_low.replace(0, 0.0001)
        ad_val = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low) * df['Volume']
        df['CMF'] = ad_val.rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['CMF'] = df['CMF'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # 6. Seasonality
        t = df.index
        df['Day_Sin'] = np.sin(2 * np.pi * t.dayofweek / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * t.dayofweek / 7)
        
        df.dropna(inplace=True)
        return df

    def prepare_tensors(self, df, split_ratio=0.8):
        feature_cols = ['Log_Returns', 'Garman_Klass', 'RSI', 'CMF', 'Day_Sin', 'Day_Cos']
        
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Exponential Decay Weights
        decay_rate = 0.001 
        dates = np.arange(len(train_df))
        weights = np.exp(decay_rate * dates)
        weights = weights / weights.mean()
        
        X_train_scaled = self.scaler.fit_transform(train_df[feature_cols])
        X_test_scaled = self.scaler.transform(test_df[feature_cols])
        
        y_train = train_df['Target'].values
        y_test = test_df['Target'].values

        def create_window(X, y, time_step, w=None):
            Xs, ys, ws = [], [], []
            for i in range(len(X) - time_step):
                Xs.append(X[i:(i + time_step)])
                ys.append(y[i + time_step])
                if w is not None: ws.append(w[i + time_step])
            
            if w is not None:
                return np.array(Xs), np.array(ys), np.array(ws)
            return np.array(Xs), np.array(ys)

        X_train_3d, y_train_3d, w_train = create_window(X_train_scaled, y_train, self.time_step, weights)
        X_test_3d, y_test_3d = create_window(X_test_scaled, y_test, self.time_step)
        
        return X_train_3d, y_train_3d, w_train, X_test_3d, y_test_3d, test_df.iloc[self.time_step:]

    def build_model(self, input_shape):
        tf.keras.backend.clear_session()
        gc.collect()
        
        model = Sequential([
            Input(shape=input_shape),
            GaussianNoise(0.005), 
            BatchNormalization(),
            Bidirectional(LSTM(128, return_sequences=True)), 
            Dropout(0.4), 
            Attention(), 
            BatchNormalization(),
            Dense(64, activation='gelu'),
            Dropout(0.3),
            Dense(1, activation='linear') 
        ])
        
        optimizer = Adam(learning_rate=0.0005, clipnorm=0.5) 
        model.compile(optimizer=optimizer, loss='huber') 
        return model

# ---- 4. UI & EXECUTION ----
with st.sidebar:
    st.header("Strategy Config")
    ticker = st.selectbox("Asset Class", ["AAPL", "NVDA", "MSFT", "BTC-USD", "ETH-USD", "SPY"])
    start = st.date_input("Start", value=pd.to_datetime("2018-01-01"))
    end = st.date_input("End", value=pd.to_datetime("today"))
    
    st.subheader("Risk Management")
    long_only = st.checkbox("Long Only Mode (No Shorts)", value=True, help="Prevents betting against the market. Essential for Stocks.")
    ensemble_size = 3 
    cost_bps = st.slider("Transaction Cost (bps)", 0, 50, 10)
    target_vol = st.slider("Target Volatility (%)", 10, 100, 30, help="Higher = More Aggressive")

if st.button("Launch Strategy"):
    pipeline = QuantPipeline(ticker, start, end, 60)
    
    # ---- TIMELINE VISUALIZATION ----
    with st.status("Initializing Quantum Engine...", expanded=True) as status:
        
        st.write("Fetching institutional data feed...")
        raw_df = pipeline.fetch_data()
        st.write(f"Data Acquired: {len(raw_df)} candles.")
        
        st.write("Calculating Garman-Klass Volatility & CMF...")
        processed_df = pipeline.engineer_features(raw_df)
        
        st.write("Applying Exponential Decay Weights...")
        X_train, y_train, w_train, X_test, y_test, test_context = pipeline.prepare_tensors(processed_df)
        
        st.write(f"Training Ensemble of {ensemble_size} Models (Bagging)...")
        
        ensemble_preds = []
        progress_bar = st.progress(0)
        
        for i in range(ensemble_size):
            st.write(f"Training Neural Network {i+1}/{ensemble_size} on GPU")
            
            model = pipeline.build_model((X_train.shape[1], X_train.shape[2]))
            
            early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
            
            model.fit(
                X_train, y_train, 
                sample_weight=w_train, 
                validation_data=(X_test, y_test),
                epochs=60, 
                batch_size=64, 
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            preds = model.predict(X_test, verbose=0).flatten()
            ensemble_preds.append(preds)
            progress_bar.progress((i + 1) / ensemble_size)
        
        st.write("Aggregating predictions & Boosting Trend Signal...")
        avg_preds = np.mean(ensemble_preds, axis=0)
        
        status.update(label="Optimization Complete", state="complete", expanded=False)

    # ---- 5. BACKTEST ENGINE ----
    backtest_df = test_context.copy()
    backtest_df['Predicted_Return'] = avg_preds
    backtest_df['Actual_Return'] = y_test 
    
    # 1. Trend Filter
    backtest_df['SMA_50'] = backtest_df['Close'].rolling(window=50).mean()
    backtest_df['Trend_Bullish'] = backtest_df['Close'] > backtest_df['SMA_50']
    
    # 2. Volatility Targeting
    current_vol = np.sqrt(backtest_df['Garman_Klass']) * np.sqrt(252)
    safe_vol = (current_vol * 100).replace(0, 0.01)
    backtest_df['Vol_Scalar'] = target_vol / safe_vol
    backtest_df['Vol_Scalar'] = backtest_df['Vol_Scalar'].clip(upper=2.0) 
    
    # 3. Signal Boosting
    boost_factor = np.where(backtest_df['Trend_Bullish'], 2.0, 1.0) 
    
    # Sigmoid Conviction (Boosted 500x)
    backtest_df['Conviction'] = tf.math.sigmoid(backtest_df['Predicted_Return'] * 500 * boost_factor).numpy() - 0.5
    backtest_df['Conviction'] = backtest_df['Conviction'] * 2 
    
    # 4. Position Logic
    raw_position = backtest_df['Conviction'] * backtest_df['Vol_Scalar']
    
    if long_only:
        backtest_df['Position'] = raw_position.clip(lower=0)
    else:
        backtest_df['Position'] = raw_position

    backtest_df['Position'] = backtest_df['Position'].fillna(0)
    
    # Cost & PnL
    backtest_df['Turnover'] = backtest_df['Position'].diff().abs().fillna(0)
    backtest_df['Cost'] = backtest_df['Turnover'] * (cost_bps / 10000)
    
    backtest_df['Gross_Return'] = backtest_df['Position'] * backtest_df['Actual_Return']
    backtest_df['Net_Return'] = backtest_df['Gross_Return'] - backtest_df['Cost']
    
    backtest_df['Cum_Market'] = (1 + backtest_df['Actual_Return']).cumprod()
    backtest_df['Cum_Strategy'] = (1 + backtest_df['Net_Return']).cumprod()
    
    # Metrics
    total_ret = backtest_df['Cum_Strategy'].iloc[-1] - 1
    sharpe = np.sqrt(252) * (backtest_df['Net_Return'].mean() / backtest_df['Net_Return'].std())
    peak = backtest_df['Cum_Strategy'].cummax()
    max_dd = ((backtest_df['Cum_Strategy'] - peak) / peak).min()
    
    wins = backtest_df[backtest_df['Net_Return'] > 0]['Net_Return']
    win_rate = len(wins) / len(backtest_df) if len(backtest_df) > 0 else 0
    
    # ---- GENERAL ACCURACY METRIC ----
    # Measures if the Prediction Sign matched the REALIZED Net Return Sign.
    # This accounts for fees: If we predicted Up, but lost money due to fees, it counts as a Miss.
    # We use a small epsilon for floating point safety.
    backtest_df['Realized_Correctness'] = np.sign(backtest_df['Predicted_Return']) == np.sign(backtest_df['Net_Return'])
    
    # Filter for days where we actually took a position (ignore cash days)
    active_days = backtest_df[backtest_df['Position'].abs() > 0.01]
    accuracy = active_days['Realized_Correctness'].mean() * 100

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net Profit", f"{total_ret*100:.2f}%")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("Win Rate", f"{win_rate*100:.1f}%")
    c4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    c5.metric("Realized Accuracy", f"{accuracy:.1f}%", help="Did the strategy actually profit when it predicted a win?")
    
    st.subheader("Performance (Net of Fees)")
    st.line_chart(backtest_df[['Cum_Strategy', 'Cum_Market']])
    
    with st.expander("See Trade Log"):
        st.dataframe(backtest_df[['Predicted_Return', 'Trend_Bullish', 'Position', 'Cost', 'Net_Return']].tail(100))
        csv = backtest_df.to_csv().encode('utf-8')
        st.download_button("Download Data", csv, "backtest_data.csv", "text/csv") 