import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import logging
import os
import pandas_ta as ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from prophet import Prophet
import optuna
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.python.client import device_lib
import warnings
import pickle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, filename="bitcoin_forecast.log", 
                    format='%(asctime)s %(message)s')

class UltimateBitcoinForecaster:
    def __init__(self):
        """Initialize the Bitcoin forecasting tools with additional metrics storage."""
        self.data = None
        self.xgb_model = None
        self.lstm_model = None
        self.prophet_model = None
        self.cat_model = None
        self.scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.stacking_model = LinearRegression()
        self.regime_models = {}
        self.best_xgb_params = None
        self.best_lstm_params = None
        self.best_prophet_params = None
        self.best_cat_params = None
        # New instance variables to store final performance metrics
        self.final_rmse = None
        self.final_mape = None
        self.final_dir_acc = None

    def check_gpu(self):
        """Ensure NVIDIA RTX 3050 GPU is used if available."""
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU detected. Proceeding with CPU.")
            logging.info("No GPU detected. Proceeding with CPU.")
        else:
            devices = device_lib.list_local_devices()
            gpu_details = [d for d in devices if d.device_type == 'GPU']
            if gpu_details:
                gpu_name = gpu_details[0].name
                gpu_memory = gpu_details[0].memory_limit / (1024 ** 2)
                print(f"Using GPU: {gpu_name} with {gpu_memory:.0f} MB memory")
                logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.0f} MB memory")
            else:
                print(f"Using GPU: {gpus[0].name}")
                logging.info(f"Using GPU: {gpus[0].name}")

    def fetch_bitcoin_data(self, days=5000, api_key=None):
        """Fetch Bitcoin price history from TwelveData API (up to 5000 days)."""
        if not api_key:
            logging.error("API key is required.")
            print("API key is required.")
            return None
        
        symbol = "BTC/USD"
        interval = "1day"
        max_days = min(days, 5000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': max_days,
            'apikey': api_key
        }
        
        try:
            response = requests.get("https://api.twelvedata.com/time_series", params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'values' not in data:
                logging.error("No 'values' in API response.")
                print("No data found.")
                return None
            
            records = [
                {
                    'Timestamp': pd.to_datetime(item['datetime']),
                    'Open': float(item['open']),
                    'High': float(item['high']),
                    'Low': float(item['low']),
                    'Close': float(item['close']),
                    'Volume': 0  # Placeholder; volume fetched separately
                }
                for item in data['values']
            ]
            
            df = pd.DataFrame(records).set_index('Timestamp').sort_index()
            self.data = df.tail(max_days)
            logging.info(f"Fetched {len(self.data)} days of data.")
            print(f"Fetched {len(self.data)} days of data.")
            return self.data
        
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            print(f"Error fetching data: {e}")
            return None

    def fetch_volume_from_binance(self, days=5000):
        """Fetch volume data from Binance API and merge with existing data."""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'limit': days
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            volume_df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            volume_df['Timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('Timestamp', inplace=True)
            volume_df = volume_df['volume'].astype(float)
            if self.data is not None:
                self.data['Volume'] = volume_df.reindex(self.data.index, method='nearest')
                logging.info("Volume data fetched from Binance and merged.")
                print("Volume data fetched from Binance and merged.")
            else:
                logging.error("No existing data to merge volume with.")
                print("No existing data to merge volume with.")
        except Exception as e:
            logging.error(f"Error fetching volume data: {e}")
            print(f"Error fetching volume data: {e}")

    def add_cycle_features(self):
        """Add Bitcoin halving cycle features to the dataset."""
        df = self.data.copy()
        halving_dates = [
            pd.Timestamp('2012-11-28'),
            pd.Timestamp('2016-07-09'),
            pd.Timestamp('2020-05-11'),
            pd.Timestamp('2024-04-19')
        ]
        
        df['HalvingCycle'] = 0
        df['DaysSinceHalving'] = 0
        df['DaysToNextHalving'] = 0
        
        for i, date in enumerate(halving_dates):
            if i < len(halving_dates) - 1:
                next_date = halving_dates[i + 1]
                mask = (df.index >= date) & (df.index < next_date)
                df.loc[mask, 'HalvingCycle'] = i + 1
                df.loc[mask, 'DaysSinceHalving'] = (df.index[mask] - date).days
                df.loc[mask, 'DaysToNextHalving'] = (next_date - df.index[mask]).days
            else:
                mask = df.index >= date
                df.loc[mask, 'HalvingCycle'] = i + 1
                df.loc[mask, 'DaysSinceHalving'] = (df.index[mask] - date).days
                next_approx = date + pd.Timedelta(days=1460)
                df.loc[mask, 'DaysToNextHalving'] = (next_approx - df.index[mask]).days
        
        first_halving = halving_dates[0]
        mask_before = df.index < first_halving
        df.loc[mask_before, 'HalvingCycle'] = 0
        df.loc[mask_before, 'DaysSinceHalving'] = (df.index[mask_before] - pd.Timestamp('2009-01-03')).days
        df.loc[mask_before, 'DaysToNextHalving'] = (first_halving - df.index[mask_before]).days
        
        avg_cycle_length = 1460
        df['CyclePosition'] = df['DaysSinceHalving'] / avg_cycle_length
        df['CyclePosition_sin'] = np.sin(2 * np.pi * df['CyclePosition'])
        df['CyclePosition_cos'] = np.cos(2 * np.pi * df['CyclePosition'])
        
        self.data = df
        logging.info("Added halving cycle features to dataset")
        return df

    def calculate_indicators(self):
        """Calculate technical indicators with feature importance tracking."""
        if self.data is None:
            logging.error("No data available for indicators.")
            print("No data available. Please run fetch_bitcoin_data() first.")
            return None
        
        df = self.data.copy()
        print("Calculating technical indicators...")
        
        if 'Volume' not in df.columns or (df['Volume'] == 0).all():
            print("Volume data is missing or all zeros. Skipping volume-based indicators.")
            volume_based_indicators = ['OBV', 'CMF', 'VWAP', 'ForceIndex13']
        else:
            volume_based_indicators = []
        
        df['LogClose'] = np.log(df['Close'])
        df['LogReturns'] = df['LogClose'].diff()
        
        df['SMA7'] = ta.sma(df['Close'], length=7)
        df['SMA25'] = ta.sma(df['Close'], length=25)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['EMA12'] = ta.ema(df['Close'], length=12)
        df['EMA26'] = ta.ema(df['Close'], length=26)
        
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ROC'] = ta.roc(df['Close'], length=14)
        
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['StochK'] = stoch['STOCHk_14_3_3']
        df['StochD'] = stoch['STOCHd_14_3_3']
        
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0']
        
        if 'OBV' not in volume_based_indicators:
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
        if 'CMF' not in volume_based_indicators:
            df['CMF'] = ta.adosc(df['High'], df['Low'], df['Close'], df['Volume'], fast=3, slow=10)
        if 'VWAP' not in volume_based_indicators:
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        if 'ForceIndex13' not in volume_based_indicators:
            df['ForceIndex'] = df['Close'].diff(1) * df['Volume']
            df['ForceIndex13'] = ta.ema(df['ForceIndex'], length=13)
        
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
        
        dmi = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = dmi['ADX_14']
        df['DI+'] = dmi['DMP_14']
        df['DI-'] = dmi['DMN_14']
        
        keltner = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
        df['KC_Upper'] = keltner['KCUe_20_2.0']
        df['KC_Lower'] = keltner['KCLe_20_2.0']
        
        df['Returns'] = df['Close'].pct_change()
        df['BullishRegime'] = (df['Close'] > df['SMA50']).astype(int)
        df['Volatility'] = df['Close'].rolling(window=14).std()
        df['DC_Upper'] = df['High'].rolling(window=20).max()
        df['DC_Lower'] = df['Low'].rolling(window=20).min()
        
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        df['BullEngulf'] = 0
        df.loc[(df['Close'].shift(1) < df['Open'].shift(1)) & 
               (df['Close'] > df['Open']) & 
               (df['Close'] > df['Open'].shift(1)) & 
               (df['Open'] < df['Close'].shift(1)), 'BullEngulf'] = 100
        df.loc[(df['Close'].shift(1) > df['Open'].shift(1)) & 
               (df['Close'] < df['Open']) & 
               (df['Close'] < df['Open'].shift(1)) & 
               (df['Open'] > df['Close'].shift(1)), 'BullEngulf'] = -100
        
        self.data = df.dropna()
        self.analyze_feature_correlations(df)
        logging.info(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        print(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        return df

    def analyze_feature_correlations(self, df):
        """Analyze and log feature correlations to identify redundant features."""
        target_corr = df.corr()['Close'].sort_values(ascending=False)
        logging.info(f"Top 10 correlated features: {target_corr[:10].to_dict()}")
        
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(upper.index[i], upper.columns[j], upper.iloc[i, j]) 
                    for i, j in zip(*np.where(upper > 0.95))]
        if high_corr:
            logging.info(f"Highly correlated features (>0.95): {high_corr}")

    def detect_market_regimes(self, window=90):
        """Detect market regimes (bull, bear, accumulation) based on price action."""
        df = self.data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=window).std()
        
        if 'ADX' not in df.columns:
            adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            df['ADX'] = adx['ADX_14']
        
        df['Regime'] = 0
        bull_condition = (df['Returns'].rolling(window=window).mean() > 0) & \
                         (df['Volatility'] > df['Volatility'].rolling(window=window*3).mean()) & \
                         (df['ADX'] > 25)
        df.loc[bull_condition, 'Regime'] = 1
        bear_condition = (df['Returns'].rolling(window=window).mean() < 0) & \
                         (df['Volatility'] > df['Volatility'].rolling(window=window*3).mean()) & \
                         (df['ADX'] > 25)
        df.loc[bear_condition, 'Regime'] = 2
        accum_condition = (df['Volatility'] < df['Volatility'].rolling(window=window*3).mean()) & \
                          (df['ADX'] < 20)
        df.loc[accum_condition, 'Regime'] = 3
        
        df['Regime'] = df['Regime'].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        self.data = df
        logging.info("Detected market regimes based on price action")
        return df

    def prepare_ml_data(self, lookback=30):
        """Prepare features and target with lagged features for predicting returns."""
        if self.data is None:
            logging.error("No indicators available for ML data.")
            print("No indicators available. Run calculate_indicators() first.")
            return None

        df = self.data.copy()
        print("Preparing ML data with lookback...")

        df['Target'] = df['Close'].pct_change().shift(-1)
        df = df.dropna(subset=['Target'])

        for lag in range(1, lookback + 1):
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

        base_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA25', 'SMA50', 'EMA12', 'EMA26',
            'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ROC', 'StochK', 'StochD', 'ATR',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CMF', 'VWAP', 'ForceIndex13', 'CCI',
            'ADX', 'DI+', 'DI-', 'KC_Upper', 'KC_Lower', 'Returns', 'LogReturns', 'BullishRegime',
            'Volatility', 'DC_Upper', 'DC_Lower', 'BullEngulf', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'Month_sin', 'Month_cos', 'LogClose',
            'HalvingCycle', 'DaysSinceHalving', 'DaysToNextHalving', 'CyclePosition_sin', 'CyclePosition_cos'
        ]

        available_features = [f for f in base_features if f in df.columns]
        features = available_features + [f'Close_lag_{lag}' for lag in range(1, lookback + 1)] + \
                  [f'Volume_lag_{lag}' for lag in range(1, lookback + 1)]

        df = df.dropna(subset=features)

        if df.empty:
            logging.error("No data left after dropping NaNs.")
            print("Error: No data left after dropping NaNs.")
            return None

        X = df[features]
        y = df['Target']
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        logging.info(f"Prepared {len(X)} samples with {len(features)} features.")
        print(f"Prepared {len(X)} samples with {len(features)} features.")
        return X_scaled, y_scaled, df.index

    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost hyperparameters with narrowed search range."""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_val_inv = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_val_inv, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        self.best_xgb_params = study.best_params
        return study.best_params

    def optimize_lstm(self, X_train, y_train, X_val, y_val, lookback):
        """Optimize LSTM hyperparameters."""
        def objective(trial):
            units1 = trial.suggest_int('units1', 64, 256)
            units2 = trial.suggest_int('units2', 32, 128)
            units3 = trial.suggest_int('units3', 16, 64)
            dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
            dropout2 = trial.suggest_float('dropout2', 0.1, 0.5)
            
            X_lstm_train, y_lstm_train = self.prepare_lstm_data(X_train, y_train, lookback)
            X_lstm_val, y_lstm_val = self.prepare_lstm_data(X_val, y_val, lookback)
            
            inputs = Input(shape=(lookback, X_train.shape[1]))
            lstm1 = Bidirectional(LSTM(units1, return_sequences=True))(inputs)
            dropout_layer1 = Dropout(dropout1)(lstm1)
            lstm2 = Bidirectional(LSTM(units2, return_sequences=True))(dropout_layer1)
            attention_output = Attention()([lstm2, lstm2])
            dropout_layer2 = Dropout(dropout2)(attention_output)
            lstm3 = LSTM(units3)(dropout_layer2)
            outputs = Dense(1, activation='linear')(lstm3)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='nadam', loss='mse')
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=64, 
                      validation_data=(X_lstm_val, y_lstm_val), callbacks=[early_stopping], verbose=0)
            
            preds_scaled = model.predict(X_lstm_val).flatten()
            preds = self.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            y_val_inv = self.target_scaler.inverse_transform(y_lstm_val.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_val_inv, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        self.best_lstm_params = study.best_params
        return study.best_params

    def optimize_prophet(self, df_train, df_val):
        """Optimize Prophet hyperparameters."""
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            }
            model = Prophet(**params)
            model.add_seasonality(name='halving_cycle', period=1460, fourier_order=5)
            model.fit(df_train)
            forecast = model.predict(df_val)
            mse = mean_squared_error(df_val['y'], forecast['yhat'])
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        self.best_prophet_params = study.best_params
        return study.best_params

    def optimize_catboost(self, X_train, y_train, X_val, y_val):
        """Optimize CatBoost hyperparameters."""
        def objective(trial):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42,
                'verbose': 0
            }
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_val_inv = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_val_inv, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        self.best_cat_params = study.best_params
        return study.best_params

    def prepare_lstm_data(self, X, y, lookback):
        """Prepare data for LSTM with lookback window."""
        X_lstm = []
        y_lstm = []
        for i in range(lookback, len(X)):
            X_lstm.append(X[i-lookback:i])
            y_lstm.append(y[i])
        return np.array(X_lstm), np.array(y_lstm)

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with best parameters."""
        if self.best_xgb_params is None:
            self.best_xgb_params = {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 100, 'subsample': 0.8}
        self.xgb_model = xgb.XGBRegressor(**self.best_xgb_params)
        self.xgb_model.fit(X_train, y_train)
        logging.info("XGBoost model trained.")

    def train_lstm(self, X_train, y_train, lookback):
        """Train LSTM model with best parameters."""
        if self.best_lstm_params is None:
            self.best_lstm_params = {'units1': 128, 'units2': 64, 'units3': 32, 'dropout1': 0.2, 'dropout2': 0.2}
        
        X_lstm, y_lstm = self.prepare_lstm_data(X_train, y_train, lookback)
        inputs = Input(shape=(lookback, X_train.shape[1]))
        lstm1 = Bidirectional(LSTM(self.best_lstm_params['units1'], return_sequences=True))(inputs)
        dropout1 = Dropout(self.best_lstm_params['dropout1'])(lstm1)
        lstm2 = Bidirectional(LSTM(self.best_lstm_params['units2'], return_sequences=True))(dropout1)
        attention_output = Attention()([lstm2, lstm2])
        dropout2 = Dropout(self.best_lstm_params['dropout2'])(attention_output)
        lstm3 = LSTM(self.best_lstm_params['units3'])(dropout2)
        outputs = Dense(1, activation='linear')(lstm3)
        
        self.lstm_model = Model(inputs=inputs, outputs=outputs)
        self.lstm_model.compile(optimizer='nadam', loss='mse')
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=64, callbacks=[early_stopping], verbose=0)
        logging.info("LSTM model trained.")

    def train_prophet(self, df_train):
        """Train Prophet model with best parameters."""
        if self.best_prophet_params is None:
            self.best_prophet_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            }
        self.prophet_model = Prophet(**self.best_prophet_params)
        self.prophet_model.add_seasonality(name='halving_cycle', period=1460, fourier_order=5)
        self.prophet_model.fit(df_train)
        logging.info("Prophet model trained.")

    def train_catboost(self, X_train, y_train):
        """Train CatBoost model with best parameters."""
        if self.best_cat_params is None:
            self.best_cat_params = {'depth': 6, 'learning_rate': 0.1, 'iterations': 500, 'l2_leaf_reg': 3}
        self.cat_model = CatBoostRegressor(**self.best_cat_params, verbose=0)
        self.cat_model.fit(X_train, y_train)
        logging.info("CatBoost model trained.")

    def train_regime_specific_models(self, X_train, y_train, lookback):
        """Train regime-specific XGBoost models."""
        df = self.data.iloc[-len(X_train):].copy()
        regimes = df['Regime'].unique()
        regime_models = {}
        
        for regime in regimes:
            regime_idx = df[df['Regime'] == regime].index
            regime_train_idx = [i for i, idx in enumerate(df.index) if idx in regime_idx]
            if len(regime_train_idx) < lookback + 1:
                continue
            X_regime = X_train[regime_train_idx]
            y_regime = y_train[regime_train_idx]
            model = xgb.XGBRegressor(**self.best_xgb_params)
            model.fit(X_regime, y_regime)
            regime_models[regime] = model
            logging.info(f"Trained regime-specific model for regime {regime}")
        return regime_models

    def optimize_hyperparameters(self, X_scaled, y_scaled, dates, lookback):
        """Optimize hyperparameters for all models."""
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        dates_train = dates[:split_idx]
        dates_val = dates[split_idx:]
        
        print("Optimizing XGBoost...")
        self.optimize_xgboost(X_train, y_train, X_val, y_val)
        print("Optimizing LSTM...")
        self.optimize_lstm(X_train, y_train, X_val, y_val, lookback)
        print("Optimizing Prophet...")
        df_train = pd.DataFrame({'ds': dates_train, 'y': self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()})
        df_val = pd.DataFrame({'ds': dates_val, 'y': self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()})
        self.optimize_prophet(df_train, df_val)
        print("Optimizing CatBoost...")
        self.optimize_catboost(X_train, y_train, X_val, y_val)

    def get_aligned_predictions(self, X_test, dates_test, lookback):
        """Get predictions from all models aligned by date."""
        X_lstm, _ = self.prepare_lstm_data(X_test, np.zeros(len(X_test)), lookback)
        
        xgb_pred = self.xgb_model.predict(X_test)
        lstm_pred = self.lstm_model.predict(X_lstm).flatten()
        prophet_df = pd.DataFrame({'ds': dates_test})
        prophet_forecast = self.prophet_model.predict(prophet_df)
        prophet_pred = prophet_forecast['yhat'].values
        cat_pred = self.cat_model.predict(X_test)
        
        min_len = min(len(xgb_pred[lookback:]), len(lstm_pred), len(prophet_pred[lookback:]), len(cat_pred[lookback:]))
        
        predictions = {
            'XGBoost': xgb_pred[lookback:lookback+min_len],
            'LSTM': lstm_pred[:min_len],
            'Prophet': prophet_pred[lookback:lookback+min_len],
            'CatBoost': cat_pred[lookback:lookback+min_len],
            'dates': dates_test[lookback:lookback+min_len]
        }
        return predictions

    def train_all_models(self, test_size=0.1, lookback=30):
        """Train all models with time series validation and store final metrics."""
        X_scaled, y_scaled, dates = self.prepare_ml_data(lookback)
        if X_scaled is None:
            return None
        
        self.optimize_hyperparameters(X_scaled, y_scaled, dates, lookback)
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {'XGBoost': [], 'LSTM': [], 'Prophet': [], 'CatBoost': [], 'Ensemble': []}
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
            dates_train, dates_test = dates[train_idx], dates[test_idx]
            
            self.train_xgboost(X_train, y_train)
            self.train_lstm(X_train, y_train, lookback)
            df_train = pd.DataFrame({'ds': dates_train, 'y': self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()})
            self.train_prophet(df_train)
            self.train_catboost(X_train, y_train)
            
            predictions = self.get_aligned_predictions(X_test, dates_test, lookback)
            
            for model, pred in predictions.items():
                if model == 'dates':
                    continue
                pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                y_test_inv = self.target_scaler.inverse_transform(y_test[lookback:lookback+len(pred)].reshape(-1, 1)).flatten()
                rmse = np.sqrt(mean_squared_error(y_test_inv, pred))
                mape = mean_absolute_percentage_error(y_test_inv + 1e-10, pred + 1e-10)
                dir_acc = np.mean(np.sign(pred) == np.sign(y_test_inv))
                cv_scores[model].append({'RMSE': rmse, 'MAPE': mape, 'Dir_Acc': dir_acc})
            
            meta_features = np.column_stack([predictions[model] for model in ['XGBoost', 'LSTM', 'Prophet', 'CatBoost']])
            meta_features = self.target_scaler.inverse_transform(meta_features)
            y_test_inv = self.target_scaler.inverse_transform(y_test[lookback:lookback+len(predictions['XGBoost'])].reshape(-1, 1)).flatten()
            self.stacking_model.fit(meta_features, y_test_inv)
            ensemble_pred = self.stacking_model.predict(meta_features)
            rmse = np.sqrt(mean_squared_error(y_test_inv, ensemble_pred))
            mape = mean_absolute_percentage_error(y_test_inv + 1e-10, ensemble_pred + 1e-10)
            dir_acc = np.mean(np.sign(ensemble_pred) == np.sign(y_test_inv))
            cv_scores['Ensemble'].append({'RMSE': rmse, 'MAPE': mape, 'Dir_Acc': dir_acc})
        
        for model, scores in cv_scores.items():
            avg_rmse = np.mean([s['RMSE'] for s in scores])
            avg_mape = np.mean([s['MAPE'] for s in scores])
            avg_dir_acc = np.mean([s['Dir_Acc'] for s in scores])
            print(f"{model} - Avg RMSE: {avg_rmse:.4f}, Avg MAPE: {avg_mape:.4f}, Avg Dir Acc: {avg_dir_acc:.4f}")
        
        # Final train-test split and model training
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        self.train_xgboost(X_train, y_train)
        self.train_lstm(X_train, y_train, lookback)
        df_train = pd.DataFrame({'ds': dates_train, 'y': self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()})
        self.train_prophet(df_train)
        self.train_catboost(X_train, y_train)
        self.regime_models = self.train_regime_specific_models(X_train, y_train, lookback)
        
        predictions = self.get_aligned_predictions(X_test, dates_test, lookback)
        
        meta_features = np.column_stack([predictions[model] for model in ['XGBoost', 'LSTM', 'Prophet', 'CatBoost']])
        meta_features = self.target_scaler.inverse_transform(meta_features)
        y_test_inv = self.target_scaler.inverse_transform(y_test[lookback:lookback+len(predictions['XGBoost'])].reshape(-1, 1)).flatten()
        self.stacking_model.fit(meta_features, y_test_inv)
        ensemble_pred = self.stacking_model.predict(meta_features)
        
        # Store final metrics
        self.final_rmse = np.sqrt(mean_squared_error(y_test_inv, ensemble_pred))
        self.final_mape = mean_absolute_percentage_error(y_test_inv + 1e-10, ensemble_pred + 1e-10)
        self.final_dir_acc = np.mean(np.sign(ensemble_pred) == np.sign(y_test_inv))
        
        logging.info(f"Final Ensemble RMSE: {self.final_rmse:.4f}, MAPE: {self.final_mape:.4f}, Directional Accuracy: {self.final_dir_acc:.4f}")
        print(f"Final Ensemble RMSE: {self.final_rmse:.4f}, MAPE: {self.final_mape:.4f}, Directional Accuracy: {self.final_dir_acc:.4f}")
        
        self.plot_predictions(predictions['dates'], y_test_inv, ensemble_pred)
        self.save_models()
        return X_test[lookback:], y_test_inv, ensemble_pred, predictions['dates']

    def plot_predictions(self, dates, actual, predicted):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual, label='Actual Returns', color='blue')
        plt.plot(dates, predicted, label='Predicted Returns', color='orange')
        plt.title('Bitcoin Returns: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('bitcoin_forecast.png')
        plt.close()
        logging.info("Prediction plot saved as bitcoin_forecast.png")

    def save_models(self):
        """Save trained models to disk."""
        with open('xgb_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        self.lstm_model.save('lstm_model.h5')
        with open('prophet_model.pkl', 'wb') as f:
            pickle.dump(self.prophet_model, f)
        with open('cat_model.pkl', 'wb') as f:
            pickle.dump(self.cat_model, f)
        with open('stacking_model.pkl', 'wb') as f:
            pickle.dump(self.stacking_model, f)
        with open('regime_models.pkl', 'wb') as f:
            pickle.dump(self.regime_models, f)
        logging.info("All models saved to disk.")

    def predict_with_cycle_awareness(self, lookback=30, days_ahead=30):
        """Generate future predictions with cycle awareness, prices, and uncertainty."""
        if self.data is None or self.xgb_model is None:
            print("Models or data not available. Train models first.")
            return None
        
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        last_features = self.data.tail(lookback).copy()
        predictions = []
        
        for i in range(days_ahead):
            X_scaled, _, _ = self.prepare_ml_data(lookback)
            last_X = X_scaled[-1].reshape(1, -1)
            regime = self.data['Regime'].iloc[-1]
            pred_scaled = self.regime_models.get(regime, self.xgb_model).predict(last_X)[0]
            pred = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred)
            
            new_row = last_features.iloc[-1].copy()
            new_row['Close'] = new_row['Close'] * (1 + pred)
            new_row.name = future_dates[i]
            
            last_features = pd.concat([last_features, pd.DataFrame([new_row])])
            last_features = last_features.tail(lookback)
            
            self.data = pd.concat([self.data, pd.DataFrame([new_row])])
            
            self.calculate_indicators()
            self.detect_market_regimes()
        
        # Create DataFrame with predicted returns
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Returns': predictions})
        
        # Calculate predicted prices
        last_known_price = self.data['Close'].iloc[-days_ahead-1]  # Price before predictions
        future_df['Price_Factor'] = (1 + future_df['Predicted_Returns']).cumprod()
        future_df['Predicted_Price'] = last_known_price * future_df['Price_Factor']
        
        # Calculate 95% confidence intervals using historical volatility
        historical_returns = self.data['Returns'].dropna()
        std_dev = historical_returns.std()
        future_df['Lower_Return'] = future_df['Predicted_Returns'] - 1.96 * std_dev
        future_df['Upper_Return'] = future_df['Predicted_Returns'] + 1.96 * std_dev
        future_df['Lower_Factor'] = (1 + future_df['Lower_Return']).cumprod()
        future_df['Lower_Price'] = last_known_price * future_df['Lower_Factor']
        future_df['Upper_Factor'] = (1 + future_df['Upper_Return']).cumprod()
        future_df['Upper_Price'] = last_known_price * future_df['Upper_Factor']
        
        # Drop temporary columns
        future_df.drop(['Price_Factor', 'Lower_Factor', 'Upper_Factor'], axis=1, inplace=True)
        
        # Display model performance and disclaimer
        print("\n--- Model Performance on Test Set ---")
        print(f"RMSE: {self.final_rmse:.4f}")
        print(f"MAPE: {self.final_mape:.4f}")
        print(f"Directional Accuracy: {self.final_dir_acc:.4f}")
        print("\n**Disclaimer**: Predicting cryptocurrency prices is extremely challenging due to their high volatility and the influence of unpredictable factors. These predictions should be treated with caution and not used as the sole basis for investment decisions.")
        
        # Display predictions in a structured table
        print("\n--- Future Predictions ---")
        print(future_df[['Date', 'Predicted_Returns', 'Predicted_Price', 'Lower_Price', 'Upper_Price']])
        
        # Save predictions to CSV
        future_df.to_csv('bitcoin_future_predictions.csv', index=False)
        print("\nPredictions saved to 'bitcoin_future_predictions.csv'")
        
        return future_df

if __name__ == "__main__":
    btc = UltimateBitcoinForecaster()
    btc.check_gpu()
    # Replace with your actual TwelveData API key
    btc.fetch_bitcoin_data(days=5000, api_key='xxxxxxxxxxxxxxxxxxxxxxxxx') 
    btc.fetch_volume_from_binance(days=5000)
    if btc.data is not None:
        btc.add_cycle_features()
        btc.calculate_indicators()
        btc.detect_market_regimes()
        result = btc.train_all_models(test_size=0.1, lookback=30)
        if result:
            X_test, y_test, y_pred, dates_test = result
            btc.predict_with_cycle_awareness(lookback=30)