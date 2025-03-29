import pandas as pd
import numpy as np
np.NaN = np.nan  # For pandas_ta compatibility
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import pandas_ta as ta
import mplfinance as mpf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class BitcoinTechnicalAnalysisML:
    def __init__(self):
        """Initialize the Bitcoin price prediction tools."""
        self.data = None  # Store Bitcoin data
        self.api_url = "https://www.alphavantage.co/query"
        # XGBoost model with GPU acceleration (set 'device' to 'cpu' if GPU unavailable)
        self.model = xgb.XGBRegressor(n_estimators=100, tree_method='hist', device='cuda', random_state=42)
        self.scaler = StandardScaler()  # Normalize features
        
    def fetch_bitcoin_data(self, days=700, vs_currency='usd', api_key=None):
        """Fetch Bitcoin price history from Alpha Vantage API."""
        if not api_key:
            print("API key is required for Alpha Vantage.")
            self.data = None
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
                print("No data found in the response. Check API key or parameters.")
                self.data = None
                return None
            
            records = []
            for date_str, values in time_series.items():
                date = pd.to_datetime(date_str)
                record = {
                    'Timestamp': date,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['5. volume'])
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            df.set_index('Timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            if len(df) > days:
                df = df[-days:]
            
            self.data = df
            print(f"Fetched {len(self.data)} days of Bitcoin data from {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}.")
            return self.data
            
        except requests.exceptions.HTTPError as e:
            print(f"Failed to fetch data: {e}")
            self.data = None
            return None
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            self.data = None
            return None
        except KeyError as e:
            print(f"Error parsing data: Key {e} not found in response.")
            self.data = None
            return None
    
    def calculate_indicators(self):
        """Calculate technical indicators for price prediction."""
        if self.data is None:
            print("No data available. Please run fetch_bitcoin_data() first.")
            return None
        
        df = self.data.copy()
        print("Calculating technical indicators...")
        
        # Simple Moving Averages
        df['SMA7'] = ta.sma(df['Close'], length=7)
        df['SMA25'] = ta.sma(df['Close'], length=25)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['SMA100'] = ta.sma(df['Close'], length=100)
        df['SMA200'] = ta.sma(df['Close'], length=200)
        
        # Exponential Moving Averages
        df['EMA12'] = ta.ema(df['Close'], length=12)
        df['EMA26'] = ta.ema(df['Close'], length=26)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Parabolic SAR
        df['SAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['StochK'] = stoch['STOCHk_14_3_3']
        df['StochD'] = stoch['STOCHd_14_3_3']
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0']
        
        # CCI
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
        
        # Volume Indicators
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['CMF'] = ta.adosc(df['High'], df['Low'], df['Close'], df['Volume'], fast=3, slow=10)
        
        # Force Index
        df['ForceIndex'] = df['Close'].diff(1) * df['Volume']
        df['ForceIndex13'] = ta.ema(df['ForceIndex'], length=13)
        
        # ATR
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Fibonacci Levels (using last 100 days)
        recent_high = df['High'].iloc[-100:].max()
        recent_low = df['Low'].iloc[-100:].min()
        df['Fib_0'] = recent_low
        df['Fib_23.6'] = recent_low + 0.236 * (recent_high - recent_low)
        df['Fib_38.2'] = recent_low + 0.382 * (recent_high - recent_low)
        df['Fib_50'] = recent_low + 0.5 * (recent_high - recent_low)
        df['Fib_61.8'] = recent_low + 0.618 * (recent_high - recent_low)
        df['Fib_100'] = recent_high
        
        # Additional Indicators for Enhanced Accuracy
        df['ROC'] = ta.roc(df['Close'], length=14)  # Rate of Change
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)  # Money Flow Index
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']  # Average Directional Index
        
        self.data = df
        print(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        return df
    
    def prepare_ml_data(self):
        """Prepare features and target for machine learning."""
        if self.data is None:
            print("No indicators available. Run calculate_indicators() first.")
            return None
        
        df = self.data.copy()
        print("Preparing ML data...")
        
        df['Target'] = df['Close'].shift(-1)  # Predict next day's close
        df = df.dropna()  # Drop rows with any NaN values
        print(f"Data shape after dropping NaNs: {df.shape}")
        
        if df.empty:
            print("Error: No data left after dropping NaNs. Try fetching more days or adjusting indicators.")
            return None
        
        # Updated feature list with new indicators
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA25', 'SMA50', 'SMA100', 'SMA200',
                    'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SAR',
                    'RSI', 'StochK', 'StochD', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'CCI', 'OBV',
                    'CMF', 'ForceIndex', 'ForceIndex13', 'ATR', 'ROC', 'MFI', 'ADX']
        
        X = df[features]
        y = df['Target']
        
        X_scaled = self.scaler.fit_transform(X)
        print(f"Prepared {len(X)} samples with {len(features)} features.")
        return X_scaled, y, X.index
    
    def train_model(self, test_size=0.2):
        """Train the XGBoost model."""
        result = self.prepare_ml_data()
        if result is None:
            print("Cannot train model without data.")
            return None
        X_scaled, y, dates = result
        
        if len(X_scaled) < 2:
            print("Not enough data to split. Fetch more days or adjust indicators.")
            return None
        
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X_scaled, y, dates, test_size=test_size, shuffle=False
        )
        print(f"Training on {len(X_train)} days, testing on {len(X_test)} days.")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: ${rmse:.2f}, MSE: {mse:.2f}")
        
        return X_test, y_test, y_pred, dates_test
    
    def predict_next_day(self):
        """Predict the next day's price and provide an explanation."""
        if self.data is None:
            print("No data available. Run earlier steps first.")
            return None
        
        last_data = self.data.tail(1)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA25', 'SMA50', 'SMA100', 'SMA200',
                    'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SAR',
                    'RSI', 'StochK', 'StochD', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'CCI', 'OBV',
                    'CMF', 'ForceIndex', 'ForceIndex13', 'ATR', 'ROC', 'MFI', 'ADX']
        
        X_last = last_data[features]
        X_last_scaled = self.scaler.transform(X_last)
        prediction = self.model.predict(X_last_scaled)[0]
        
        last_date = self.data.index[-1]
        next_date = last_date + timedelta(days=1)
        
        # Generate explanation
        explanation = self.generate_prediction_explanation(last_data)
        
        print(f"Last close ({last_date.strftime('%Y-%m-%d')}): ${last_data['Close'].values[0]:.2f}.")
        print(f"Predicted for {next_date.strftime('%Y-%m-%d')}: ${prediction:.2f}.")
        print(f"Explanation: {explanation}")
        return prediction
    
    def generate_prediction_explanation(self, last_data):
        """Generate a 2-3 line explanation based on key indicators."""
        macd_hist = last_data['MACD_Hist'].values[0]
        rsi = last_data['RSI'].values[0]
        bb_upper = last_data['BB_Upper'].values[0]
        bb_lower = last_data['BB_Lower'].values[0]
        close = last_data['Close'].values[0]
        
        # Trend analysis using MACD
        trend = "bullish" if macd_hist > 0 else "bearish"
        
        # Momentum analysis using RSI
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        else:
            momentum = "neutral"
        
        # Volatility analysis using Bollinger Bands
        if close > bb_upper:
            volatility = "high volatility with potential for a reversal"
        elif close < bb_lower:
            volatility = "high volatility with potential for a bounce"
        else:
            volatility = "normal volatility"
        
        explanation = (f"The prediction is based on a {trend} trend indicated by the MACD. "
                       f"The RSI suggests the market is {momentum}. "
                       f"Bollinger Bands indicate {volatility}.")
        return explanation
    
    def plot_predictions(self, X_test, y_test, y_pred, dates_test):
        """Plot actual vs predicted prices."""
        plt.figure(figsize=(14, 7))
        plt.plot(dates_test, y_test, label='Actual Price', color='blue')
        plt.plot(dates_test, y_pred, label='Predicted Price', color='red', linestyle='--')
        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        print("Plot: Blue = Actual, Red dashed = Predicted.")

if __name__ == "__main__":
    print("Starting Bitcoin price prediction...")
    btc = BitcoinTechnicalAnalysisML()
    # Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
    btc.fetch_bitcoin_data(days=700, api_key='XXXXXXXXXXXXXXXX')
    if btc.data is not None:
        btc.calculate_indicators()
        result = btc.train_model(test_size=0.2)
        if result is not None:
            X_test, y_test, y_pred, dates_test = result
            btc.plot_predictions(X_test, y_test, y_pred, dates_test)
            btc.predict_next_day()
    else:
        print("Failed to fetch data. Please check API key or network connection.")