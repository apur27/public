import pandas as pd
import numpy as np
np.NaN = np.nan  # For pandas_ta compatibility
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from prophet import Prophet
import optuna
import logging
import warnings
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.python.client import device_lib
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Configure logging
logging.basicConfig(level=logging.INFO, filename="bitcoin_forecast.log", 
                    format='%(asctime)s %(message)s')

class UltimateBitcoinForecaster:
    def __init__(self):
        """Initialize the Bitcoin forecasting tools."""
        self.data = None
        self.api_url = "https://www.alphavantage.co/query"
        self.xgb_model = xgb.XGBRegressor(random_state=42)
        self.lstm_model = None
        self.prophet_model = None
        self.cat_model = CatBoostRegressor(random_state=42, silent=True)
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.xgb_weight = 0.0
        self.lstm_weight = 0.0
        self.prophet_weight = 0.0
        self.cat_weight = 0.0

    def check_gpu(self):
        """Ensure NVIDIA RTX 3050 GPU is used."""
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError("No GPU detected. Ensure NVIDIA drivers and CUDA/cuDNN are installed.")
        # Get detailed device info
        devices = device_lib.list_local_devices()
        gpu_details = [d for d in devices if d.device_type == 'GPU']
        if gpu_details:
            gpu_name = gpu_details[0].name
            gpu_memory = gpu_details[0].memory_limit / (1024 ** 2)  # Convert to MB
            print(f"Using GPU: {gpu_name} with {gpu_memory:.0f} MB memory")
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.0f} MB memory")
        else:
            print(f"Using GPU: {gpus[0].name}")
            logging.info(f"Using GPU: {gpus[0].name}")

    def fetch_bitcoin_data(self, days=500, vs_currency='usd', api_key=None):
        """Fetch Bitcoin price history from Alpha Vantage API."""
        if not api_key:
            logging.error("API key is required.")
            print("API key is required.")
            return None
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': 'BTC',
            'market': vs_currency.upper(),
            'apikey': api_key
        }
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            time_series = data.get('Time Series (Digital Currency Daily)', {})
            if not time_series:
                logging.error("No data found in API response.")
                print("No data found.")
                return None
            records = [
                {
                    'Timestamp': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['5. volume'])
                }
                for date_str, values in time_series.items()
            ]
            df = pd.DataFrame(records).set_index('Timestamp').sort_index()
            self.data = df[-days:] if len(df) > days else df
            logging.info(f"Fetched {len(self.data)} days of data.")
            print(f"Fetched {len(self.data)} days of data.")
            return self.data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            print(f"Error fetching data: {e}")
            return None

    def calculate_indicators(self):
        """Calculate technical indicators without TA-Lib dependency."""
        if self.data is None:
            logging.error("No data available for indicators.")
            print("No data available. Please run fetch_bitcoin_data() first.")
            return None
        
        df = self.data.copy()
        print("Calculating technical indicators...")
        
        # Moving Averages
        df['SMA7'] = ta.sma(df['Close'], length=7)
        df['SMA25'] = ta.sma(df['Close'], length=25)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['EMA12'] = ta.ema(df['Close'], length=12)
        df['EMA26'] = ta.ema(df['Close'], length=26)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # ROC
        df['ROC'] = ta.roc(df['Close'], length=14)
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['StochK'] = stoch['STOCHk_14_3_3']
        df['StochD'] = stoch['STOCHd_14_3_3']
        
        # ATR
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0']
        
        # Volume Indicators
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['CMF'] = ta.adosc(df['High'], df['Low'], df['Close'], df['Volume'], fast=3, slow=10)
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Force Index
        df['ForceIndex'] = df['Close'].diff(1) * df['Volume']
        df['ForceIndex13'] = ta.ema(df['ForceIndex'], length=13)
        
        # CCI
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
        
        # ADX and DMI
        dmi = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = dmi['ADX_14']
        df['DI+'] = dmi['DMP_14']
        df['DI-'] = dmi['DMN_14']
        
        # Keltner Channels
        keltner = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
        df['KC_Upper'] = keltner['KCUe_20_2.0']
        df['KC_Lower'] = keltner['KCLe_20_2.0']
        
        # Additional Features: Returns and Log-Returns
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close']).diff()
        
        # Market Regime Indicator
        df['BullishRegime'] = (df['Close'] > df['SMA50']).astype(int)
        
        # Historical Volatility
        df['Volatility'] = df['Close'].rolling(window=14).std()
        
        # Donchian Channels
        df['DC_Upper'] = df['High'].rolling(window=20).max()
        df['DC_Lower'] = df['Low'].rolling(window=20).min()
        
        # Simple Engulfing Pattern Detection (without TA-Lib)
        df['BullEngulf'] = 0  # Default to no pattern
        df.loc[(df['Close'].shift(1) < df['Open'].shift(1)) & 
               (df['Close'] > df['Open']) & 
               (df['Close'] > df['Open'].shift(1)) & 
               (df['Open'] < df['Close'].shift(1)), 'BullEngulf'] = 100
        df.loc[(df['Close'].shift(1) > df['Open'].shift(1)) & 
               (df['Close'] < df['Open']) & 
               (df['Close'] < df['Open'].shift(1)) & 
               (df['Open'] > df['Close'].shift(1)), 'BullEngulf'] = -100
        
        self.data = df.dropna()
        logging.info(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        print(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        return df

    def prepare_ml_data(self, lookback=2):
        """Prepare features and target with lagged features."""
        if self.data is None:
            logging.error("No indicators available for ML data.")
            print("No indicators available. Run calculate_indicators() first.")
            return None
        
        df = self.data.copy()
        print("Preparing ML data with lookback...")
        
        for lag in range(1, lookback + 1):
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        print(f"Data shape after dropping NaNs: {df.shape}")
        
        if df.empty:
            logging.error("No data left after dropping NaNs.")
            print("Error: No data left after dropping NaNs.")
            return None
        
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA25', 'SMA50', 'EMA12', 'EMA26', 
            'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ROC', 'StochK', 'StochD', 'ATR', 
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CMF', 'VWAP', 'ForceIndex13', 'CCI', 
            'ADX', 'DI+', 'DI-', 'KC_Upper', 'KC_Lower', 'Returns', 'LogReturns', 'BullishRegime', 
            'Volatility', 'DC_Upper', 'DC_Lower', 'BullEngulf'
        ] + [f'Close_lag_{lag}' for lag in range(1, lookback + 1)] + \
            [f'Volume_lag_{lag}' for lag in range(1, lookback + 1)]
        
        X = df[features]
        y = df['Target']
        
        X_scaled = self.scaler.fit_transform(X)
        logging.info(f"Prepared {len(X)} samples with {len(features)} features.")
        print(f"Prepared {len(X)} samples with {len(features)} features.")
        return X_scaled, y, df.index

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost with Optuna hyperparameter optimization."""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'random_state': 42
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            mse = mean_squared_error(y_train, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        self.xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
        self.xgb_model.fit(X_train, y_train)
        logging.info(f"XGBoost trained with best parameters: {best_params}")
        print(f"Best XGBoost parameters: {best_params}")

    def prepare_lstm_data(self, X, y, lookback):
        """Prepare data for LSTM with correct alignment."""
        X_lstm, y_lstm = [], []
        for i in range(lookback, len(X)):
            X_lstm.append(X[i - lookback:i])
            y_lstm.append(y.iloc[i] if isinstance(y, pd.Series) else y[i])
        return np.array(X_lstm), np.array(y_lstm)

    def train_lstm(self, X_train, y_train, lookback=2):
        """Train Bidirectional LSTM with early stopping."""
        X_lstm, y_lstm = self.prepare_lstm_data(X_train, y_train, lookback)
        y_lstm_scaled = self.target_scaler.fit_transform(y_lstm.reshape(-1, 1)).flatten()
        self.lstm_model = Sequential()
        self.lstm_model.add(Bidirectional(LSTM(128, return_sequences=True), 
                                         input_shape=(lookback, X_train.shape[1])))
        self.lstm_model.add(Dropout(0.3))
        self.lstm_model.add(Bidirectional(LSTM(64)))
        self.lstm_model.add(Dropout(0.3))
        self.lstm_model.add(Dense(1, activation='linear'))
        self.lstm_model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.lstm_model.fit(X_lstm, y_lstm_scaled, epochs=100, batch_size=32, validation_split=0.2,
                            callbacks=[early_stopping], verbose=1)
        logging.info("LSTM model trained with Bidirectional layers.")
        print("LSTM model trained.")

    def train_prophet(self):
        """Train Prophet model on historical data."""
        df_prophet = self.data.reset_index()[['Timestamp', 'Close']].rename(columns={'Timestamp': 'ds', 'Close': 'y'})
        self.prophet_model = Prophet()
        self.prophet_model.fit(df_prophet)
        logging.info("Prophet model trained.")
        print("Prophet model trained.")

    def train_catboost(self, X_train, y_train):
        """Train CatBoost model."""
        self.cat_model.fit(X_train, y_train)
        logging.info("CatBoost model trained.")
        print("CatBoost model trained.")

    def train_all_models(self, test_size=0.1, lookback=2):
        """Train all models and evaluate with dynamic ensemble weighting."""
        X_scaled, y, dates = self.prepare_ml_data(lookback)
        if X_scaled is None:
            return None
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X_scaled, y, dates, test_size=test_size, shuffle=False
        )
        
        self.train_xgboost(X_train, y_train)
        xgb_pred_full = self.xgb_model.predict(X_test)
        
        self.train_lstm(X_train, y_train, lookback)
        X_test_lstm, y_test_lstm = self.prepare_lstm_data(X_test, y_test, lookback)
        lstm_pred_scaled = self.lstm_model.predict(X_test_lstm).flatten()
        lstm_pred = self.target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        
        self.train_prophet()
        future = self.prophet_model.make_future_dataframe(periods=len(y_test))
        forecast = self.prophet_model.predict(future)
        prophet_pred_full = forecast['yhat'].iloc[-len(y_test):].values
        
        self.train_catboost(X_train, y_train)
        cat_pred_full = self.cat_model.predict(X_test)
        
        start_idx = lookback
        xgb_pred = xgb_pred_full[start_idx:]
        prophet_pred = prophet_pred_full[start_idx:]
        cat_pred = cat_pred_full[start_idx:]
        y_test_aligned = y_test[start_idx:]
        lstm_pred = lstm_pred  # Already aligned from prepare_lstm_data
        dates_test_aligned = dates_test[start_idx:]
        
        # Dynamic ensemble weighting based on inverse RMSE
        xgb_rmse = np.sqrt(mean_squared_error(y_test_aligned, xgb_pred))
        lstm_rmse = np.sqrt(mean_squared_error(y_test_aligned, lstm_pred))
        prophet_rmse = np.sqrt(mean_squared_error(y_test_aligned, prophet_pred))
        cat_rmse = np.sqrt(mean_squared_error(y_test_aligned, cat_pred))
        total_inverse_rmse = (1 / xgb_rmse + 1 / lstm_rmse + 1 / prophet_rmse + 1 / cat_rmse)
        self.xgb_weight = (1 / xgb_rmse) / total_inverse_rmse
        self.lstm_weight = (1 / lstm_rmse) / total_inverse_rmse
        self.prophet_weight = (1 / prophet_rmse) / total_inverse_rmse
        self.cat_weight = (1 / cat_rmse) / total_inverse_rmse
        
        ensemble_pred = (self.xgb_weight * xgb_pred + 
                         self.lstm_weight * lstm_pred + 
                         self.prophet_weight * prophet_pred + 
                         self.cat_weight * cat_pred)
        
        rmse = np.sqrt(mean_squared_error(y_test_aligned, ensemble_pred))
        logging.info(f"Ensemble RMSE: ${rmse:.2f} (Weights - XGBoost: {self.xgb_weight:.2f}, "
                     f"LSTM: {self.lstm_weight:.2f}, Prophet: {self.prophet_weight:.2f}, CatBoost: {self.cat_weight:.2f})")
        print(f"Ensemble RMSE: ${rmse:.2f} (Weights - XGBoost: {self.xgb_weight:.2f}, "
              f"LSTM: {self.lstm_weight:.2f}, Prophet: {self.prophet_weight:.2f}, CatBoost: {self.cat_weight:.2f})")
        
        self.plot_predictions(dates_test_aligned, y_test_aligned, ensemble_pred)
        return X_test[start_idx:], y_test_aligned, ensemble_pred, dates_test_aligned

    def generate_detailed_explanation(self, last_data, df):
        """Generate a detailed and actionable explanation based on technical indicators."""
        macd_hist = last_data['MACD_Hist'].values[0]
        rsi = last_data['RSI'].values[0]
        close = last_data['Close'].values[0]
        bb_upper = last_data['BB_Upper'].values[0]
        bb_lower = last_data['BB_Lower'].values[0]
        vwap = last_data['VWAP'].values[0]
        atr = last_data['ATR'].values[0]
        adx_val = last_data['ADX'].values[0]

        trend = "bullish" if macd_hist > 0 else "bearish"
        momentum = ("overbought (high risk of pullback)" if rsi > 70 else
                    "oversold (potential bounce)" if rsi < 30 else
                    "neutral")
        volatility = "high (potential breakout)" if close > bb_upper or close < bb_lower else "normal"
        vwap_position = "bullish (above VWAP)" if close > vwap else "bearish (below VWAP)"
        volatility_range = f"${atr:.2f}"
        trend_strength = "strong (ADX>25)" if adx_val > 25 else "weak (ADX<25)"

        support_level = df['Low'][-14:].min()
        resistance_level = df['High'][-14:].max()

        candle_signal = "No clear reversal patterns"
        if 'BullEngulf' in last_data.columns and last_data['BullEngulf'].values[0] == 100:
            candle_signal = "Bullish Engulfing (bullish reversal)"
        elif 'BullEngulf' in last_data.columns and last_data['BullEngulf'].values[0] == -100:
            candle_signal = "Bearish Engulfing (bearish reversal)"

        explanation = (
            f"Trend: {trend}, {trend_strength}. "
            f"Momentum: {momentum}. "
            f"Volatility: {volatility}, range {volatility_range}. "
            f"Price: {vwap_position}. "
            f"Support: ${support_level:.2f}, Resistance: ${resistance_level:.2f}. "
            f"Candle: {candle_signal}."
        )
        return explanation

    def predict_next_day(self, lookback=2):
        """Predict the next day's price with ensemble and detailed explanation."""
        if self.data is None or self.xgb_model is None or self.lstm_model is None or self.prophet_model is None or self.cat_model is None:
            logging.error("No trained models or data available.")
            print("No trained models or data available. Run train_all_models() first.")
            return None
        
        df_pred = self.data.copy()
        
        for lag in range(1, lookback + 1):
            df_pred[f'Close_lag_{lag}'] = df_pred['Close'].shift(lag)
            df_pred[f'Volume_lag_{lag}'] = df_pred['Volume'].shift(lag)
        
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA25', 'SMA50', 'EMA12', 'EMA26', 
            'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ROC', 'StochK', 'StochD', 'ATR', 
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CMF', 'VWAP', 'ForceIndex13', 'CCI', 
            'ADX', 'DI+', 'DI-', 'KC_Upper', 'KC_Lower', 'Returns', 'LogReturns', 'BullishRegime', 
            'Volatility', 'DC_Upper', 'DC_Lower', 'BullEngulf'
        ] + [f'Close_lag_{lag}' for lag in range(1, lookback + 1)] + \
            [f'Volume_lag_{lag}' for lag in range(1, lookback + 1)]
        
        last_data = df_pred.tail(1)
        
        if last_data[features].isnull().any().any():
            logging.error("Insufficient data due to NaNs in features.")
            print("Insufficient data for prediction due to NaNs in features.")
            return None
        
        X_last = last_data[features]
        X_last_scaled = self.scaler.transform(X_last)
        
        xgb_pred = self.xgb_model.predict(X_last_scaled)[0]
        
        lstm_data = df_pred[features].tail(lookback)
        if lstm_data.isnull().any().any():
            logging.error("Insufficient data for LSTM due to NaNs.")
            print("Insufficient data for LSTM due to NaNs in the last lookback rows.")
            return None
        X_lstm = self.scaler.transform(lstm_data)
        X_lstm = X_lstm.reshape((1, lookback, len(features)))
        lstm_pred_scaled = self.lstm_model.predict(X_lstm, verbose=0)[0][0]
        lstm_pred = self.target_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
        
        future = self.prophet_model.make_future_dataframe(periods=1)
        forecast = self.prophet_model.predict(future)
        prophet_pred = forecast['yhat'].iloc[-1]
        
        cat_pred = self.cat_model.predict(X_last_scaled)[0]
        
        prediction = (self.xgb_weight * xgb_pred + 
                      self.lstm_weight * lstm_pred + 
                      self.prophet_weight * prophet_pred + 
                      self.cat_weight * cat_pred)
        
        explanation = self.generate_detailed_explanation(last_data, df_pred)
        
        logging.info(f"XGBoost: ${xgb_pred:.2f}, LSTM: ${lstm_pred:.2f}, Prophet: ${prophet_pred:.2f}, "
                     f"CatBoost: ${cat_pred:.2f}, Composite: ${prediction:.2f}, Explanation: {explanation}")
        print(f"XGBoost Prediction: ${xgb_pred:.2f}, LSTM Prediction: ${lstm_pred:.2f}, "
              f"Prophet Prediction: ${prophet_pred:.2f}, CatBoost Prediction: ${cat_pred:.2f}")
        print(f"Composite Prediction: ${prediction:.2f}")
        print(f"TA Explanation: {explanation}")
        return prediction

    def plot_predictions(self, dates, actual, predicted):
        """Visualize actual vs predicted prices."""
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual, label='Actual Price', color='blue')
        plt.plot(dates, predicted, label='Predicted Price', color='red', linestyle='--')
        plt.title('Bitcoin Actual vs. Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        plt.show()
        logging.info("Prediction plot generated.")

if __name__ == "__main__":
    btc = UltimateBitcoinForecaster()
    btc.check_gpu()  # Verify GPU usage
    btc.fetch_bitcoin_data(api_key='xxxxxxxxxxxxxxxxxx')  # Replace with your Alpha Vantage API key
    if btc.data is not None:
        btc.calculate_indicators()
        result = btc.train_all_models(test_size=0.1, lookback=2)
        if result:
            X_test, y_test, y_pred, dates_test = result
            btc.predict_next_day(lookback=2)