"""
AI Engine - Indian Markets Only
Hybrid data fetching: Upstox (Primary) and nselib (Fallback) for Indian stocks (NSE/BSE) and Indices
"""

import pandas as pd
import numpy as np
import os
from nselib import capital_market
from nsepython import *
import upstox_client
from upstox_client.rest import ApiException
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import warnings

warnings.filterwarnings("ignore")

# Initialize APIs
# nselib doesn't require API key - free NSE data

class AIEngine:
    def __init__(self):
        self.models = {}
        # Map common index symbols to nselib names
        self.index_mapping = {
            '^NSEI': 'NIFTY 50',
            'NIFTY': 'NIFTY 50',
            '^NSEBANK': 'NIFTY BANK',
            'BANKNIFTY': 'NIFTY BANK',
            'NIFTY_BANK': 'NIFTY BANK',
            '^CNXIT': 'NIFTY IT',
            'NIFTY_IT': 'NIFTY IT'
        }
        
        # Initialize Upstox Client
        self.upstox_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self.upstox_config = upstox_client.Configuration()
        self.upstox_config.access_token = self.upstox_token
        self.upstox_api = upstox_client.HistoryApi(upstox_client.ApiClient(self.upstox_config))
        
        if self.upstox_token:
            print("Upstox Integration: Active in AIEngine")

    def fetch_data(self, ticker, period="5y"):
        """Fetches historical data using Upstox or nselib"""
        ticker = ticker.strip().upper()
        
        print(f"\n--- Fetching Data for {ticker} (Hybrid Approach) ---")
        
        # 1. Try Upstox for Indian Stocks first (if token provided)
        if self.upstox_token:
            df, stock = self._fetch_upstox(ticker)
            if df is not None and len(df) >= 50:
                print("  Success: Using Upstox Data")
                return df, stock

        # 2. Try nselib for Indian stocks as fallback
        df, stock = self._fetch_nselib(ticker)
        if df is not None and len(df) >= 50:
            print("  Success: Using nselib Fallback Data")
            return df, stock
        
        return None, None
    
    def _fetch_upstox(self, ticker):
        """Fetch professional grade data from Upstox"""
        if not self.upstox_token:
            return None, None
            
        try:
            print(f"  Trying Upstox for {ticker}...")
            clean_ticker = ticker.split('.')[0]
            
            # Map indices
            index_map = {
                'NIFTY 50': 'NSE_INDEX|Nifty 50',
                'NIFTY BANK': 'NSE_INDEX|Nifty Bank',
                'NIFTY IT': 'NSE_INDEX|Nifty IT'
            }
            
            upper_ticker = ticker.upper()
            instrument_key = f"NSE_EQ|{clean_ticker}"
            
            if upper_ticker in self.index_mapping:
                mapped_name = self.index_mapping[upper_ticker]
                instrument_key = index_map.get(mapped_name, f"NSE_INDEX|{mapped_name}")
            
            # High quality fetch settings
            today = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            api_response = self.upstox_api.get_historical_candle_data1(
                instrument_key, 
                'day', 
                today, 
                start_date,
                "2.0"
            )
            
            if api_response.status == 'success' and api_response.data.candles:
                candles = api_response.data.candles
                df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                
                # Reverse and clean as requested
                df = df[::-1].reset_index(drop=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                class MockTicker:
                    def __init__(self):
                        self.news = []
                
                return df, MockTicker()
                
        except Exception as e:
            print(f"  [!] Upstox failed: {str(e)[:60]}")
            
        return None, None

    
    def _fetch_nselib(self, ticker):
        """Fetch from nselib (NSE stocks & Indices - FREE)"""
        try:
            print(f"  Trying nselib...")
            
            # Clean ticker - remove suffixes
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('.BSE', '')
            
            # Map common index symbols to nselib names
            index_mapping = {
                '^NSEI': 'NIFTY 50',
                'NIFTY': 'NIFTY 50',
                '^NSEBANK': 'NIFTY BANK',
                'BANKNIFTY': 'NIFTY BANK',
                'NIFTY_BANK': 'NIFTY BANK',
                '^CNXIT': 'NIFTY IT',
                'NIFTY_IT': 'NIFTY IT'
            }
            
            if clean_ticker.upper() in index_mapping:
                clean_ticker = index_mapping[clean_ticker.upper()]
            
            # Identify if it's an index
            indices = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT', 'NIFTY AUTO', 'NIFTY PHARMA', 'NIFTY FMCG', 'NIFTY REALTY', 'NIFTY METAL', 'NIFTY ENERGY', 'NIFTY INFRA', 'NIFTY PSE', 'NIFTY CPSE', 'NIFTY NEXT 50']
            is_index = any(idx in clean_ticker.upper() for idx in indices)
            
            print(f"  Fetching {'Index' if is_index else 'Equity'} data for: {clean_ticker}")
            
            # Date range for stable fetching
            end_date = datetime.now()
            start_date = end_date - timedelta(days=450)
            start_str = start_date.strftime('%d-%m-%Y')
            end_str = end_date.strftime('%d-%m-%Y')
            
            try:
                data = None
                if is_index:
                    try:
                        print(f"  Attempting nsepython index_history for: {clean_ticker}")
                        data = index_history(clean_ticker, start_str, end_str)
                    except Exception as nie:
                        print(f"  [!] nsepython index failed: {nie}")
                        data = capital_market.index_data(clean_ticker, start_str, end_str)
                        
                    column_mapping = {
                        'HistoricalDate': 'Date', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume',
                        'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume',
                        'Index Name': 'Index', 'INDEX_NAME': 'Index'
                    }
                else:
                    data = capital_market.price_volume_and_deliverable_position_data(clean_ticker, start_str, end_str)
                    column_mapping = {
                        'Date': 'Date', 'OpenPrice': 'Open', 'HighPrice': 'High', 'LowPrice': 'Low', 'ClosePrice': 'Close', 'TotalTradedQuantity': 'Volume'
                    }
                
                if data is not None and not data.empty:
                    df = pd.DataFrame(data)
                    
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df.rename(columns={old_col: new_col}, inplace=True)
                    
                    # Fallback for volume
                    if 'Volume' not in df.columns:
                        vol_cols = ['TTL_TRD_QNTY', 'TURNOVER', 'TotalTradedQuantity', 'TURN OVER']
                        for vc in vol_cols:
                            if vc in df.columns:
                                df.rename(columns={vc: 'Volume'}, inplace=True)
                                break
                        
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_cols):
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        df = df.sort_index()
                        
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if df[col].dtype == object:
                                df[col] = df[col].str.replace(',', '')
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df = df.dropna()
                        
                        if len(df) > 50:
                            print(f"  [OK] nselib: {len(df)} rows")
                            
                            class MockTicker:
                                def __init__(self):
                                    self.news = []
                            
                            return df, MockTicker()
                        else:
                            print(f"  [X] Insufficient data: Only {len(df)} rows")
                    else:
                        print(f"  [X] Missing required columns in response")
                else:
                    print(f"  [X] No data returned from nselib")
                    
            except Exception as inner_e:
                print(f"  [X] nselib call failed: {str(inner_e)[:60]}")
                
        except Exception as e:
            print(f"  [X] nselib unexpected error: {str(e)[:60]}")
        
        return None, None

    def add_indicators(self, df):
        """Adds Advanced Technical Indicators for ML Features"""
        if 'Close' not in df.columns:
            return None

        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['MA20'] + 2*df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['MA20'] - 2*df['Close'].rolling(window=20).std()
        
        # Target: 1 if Price Up Tomorrow, 0 if Down
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def analyze_sentiment(self, ticker_obj):
        """Analyzes news sentiment using NLP"""
        try:
            if not ticker_obj: 
                return 0, "Neutral"
            
            news = ticker_obj.news
            if not news:
                return 0, "Neutral"
            
            polarity_sum = 0
            count = 0
            
            for article in news[:7]: 
                title = article.get('title', '')
                blob = TextBlob(title)
                polarity_sum += blob.sentiment.polarity
                count += 1
            
            avg_polarity = polarity_sum / count if count > 0 else 0
            
            if avg_polarity > 0.15: 
                return avg_polarity, "Bullish"
            if avg_polarity < -0.15: 
                return avg_polarity, "Bearish"
            return avg_polarity, "Neutral"
        except Exception as e:
            print(f"Sentiment Error: {e}")
            return 0, "Neutral"

    def predict_price_movement(self, ticker):
        """Main function to train model and predict next move"""
        df, stock = self.fetch_data(ticker)
        
        if df is None:
            return {"error": f"Could not find data for symbol '{ticker}'"}

        df = self.add_indicators(df)
        if df is None or len(df) < 30:
             return {"error": "Not enough data after calculating indicators."}
        
        # Features for ML
        features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volume']
        
        # Ensure all columns exist
        missing_cols = [c for c in features if c not in df.columns]
        if missing_cols:
             return {"error": f"Missing data columns: {missing_cols}"}

        X = df[features]
        y = df['Target']
        
        # Train/Test Split
        split = int(len(df) * 0.85)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Model: Random Forest Classifier
        model = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Accuracy Score
        accuracy = model.score(X_test, y_test)
        
        # Prediction
        latest_data = X.iloc[[-1]]
        prediction = model.predict(latest_data)[0]
        prob = model.predict_proba(latest_data)[0][1] 
        
        # Sentiment
        sentiment_score, sentiment_label = self.analyze_sentiment(stock)
        
        # Hybrid Signal
        final_signal = "HOLD"
        if prob > 0.6 and sentiment_score > -0.1:
            final_signal = "BUY"
        elif prob < 0.4 and sentiment_score < 0.1:
            final_signal = "SELL"
            
        return {
            "current_price": round(df['Close'].iloc[-1], 2),
            "prediction": "UP" if prediction == 1 else "DOWN",
            "confidence": round(prob * 100, 1),
            "accuracy": round(accuracy * 100, 1),
            "sentiment": sentiment_label,
            "sentiment_score": round(sentiment_score, 2),
            "signal": final_signal,
            "rsi": round(df['RSI'].iloc[-1], 2),
            "macd": round(df['MACD'].iloc[-1], 2)
        }
