import pandas as pd
import numpy as np
import os
import requests
from nselib import capital_market
from nsepython import *
import yfinance as yf
import upstox_client
from upstox_client.rest import ApiException
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from .news_handler import get_google_news
import warnings
import json

warnings.filterwarnings("ignore")

class MockTicker:
    def __init__(self, name=""):
        self.name = name
        self.news = []

class AdvancedAIEngine:
    def __init__(self):
        self.models = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Map common index symbols to nselib names
        self.index_mapping = {
            '^NSEI': 'NIFTY 50',
            'NIFTY': 'NIFTY 50',
            '^NSEBANK': 'NIFTY BANK',
            'BANKNIFTY': 'NIFTY BANK',
            'NIFTY_BANK': 'NIFTY BANK',
            '^CNXIT': 'NIFTY IT',
            'NIFTY_IT': 'NIFTY IT',
            '^BSESN': 'SENSEX',
            'SENSEX': 'SENSEX'
        }
        
        # Initialize Upstox Client
        self.upstox_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        self.upstox_config = upstox_client.Configuration()
        self.upstox_config.access_token = self.upstox_token
        self.upstox_api = upstox_client.HistoryApi(upstox_client.ApiClient(self.upstox_config))
        
        if self.upstox_token:
            print("Upstox Integration: Active in AdvancedAIEngine")
            
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            'Authorization': f'Bearer {self.upstox_token}',
            'Accept': 'application/json'
        }
        
    def get_current_price(self, ticker):
        """
        Get the most recent price for a ticker efficiently.
        Prioritizes: Upstox -> nselib -> yfinance
        """
        try:
            ticker = ticker.strip().upper().replace('.NS', '').replace('.BO', '')
            
            # 1. Try Upstox (Best for real-time)
            if self.upstox_token:
                try:
                    import upstox_client
                    config = upstox_client.Configuration()
                    config.access_token = self.upstox_token
                    market_api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
                    
                    # Determine Instrument Key
                    is_index = any(idx in ticker for idx in ['NIFTY', 'BANK', 'SENSEX'])
                    if is_index:
                        symbol = "NSE_INDEX|Nifty 50" if "50" in ticker else "NSE_INDEX|Nifty Bank"
                    else:
                        symbol = f"NSE_EQ|{ticker}"
                    
                    api_response = market_api.get_full_market_quote(symbol)
                    if api_response.status == 'success' and symbol in api_response.data:
                        price = api_response.data[symbol].last_price
                        if price: return round(float(price), 2)
                except Exception as e:
                    print(f"  [!] Upstox LTP Error for {ticker}: {e}")


            # 3. Fallback: yfinance (Last resort)
            import yfinance as yf
            yf_ticker = ticker + ".NS"
            # Using a safer approach with fast_info if possible
            stock = yf.Ticker(yf_ticker)
            price = stock.fast_info.get('lastPrice')
            
            if not price:
                # One last try with download
                data = yf.download(yf_ticker, period='1d', interval='1m', progress=False, threads=False)
                if not data.empty:
                    price = data['Close'].iloc[-1]
            
            if price: return round(float(price), 2)
            
            return None
        except Exception as e:
            print(f"  [!] LTP Critical Error for {ticker}: {e}")
            return None

    def get_market_status(self):
        """
        Get current Indian Market Status (Open/Closed/Holiday) using Time-Based logic (IST)
        and manual holiday check for robust performance.
        """
        try:
            # 1. Calculate IST Time (UTC + 5:30)
            from datetime import datetime, timezone, timedelta
            utc_now = datetime.now(timezone.utc)
            ist_offset = timedelta(hours=5, minutes=30)
            now = utc_now + ist_offset
            
            # 2. Weekend Check
            if now.weekday() >= 5: # Saturday=5, Sunday=6
                return "CLOSED (WEEKEND)"
            
            # 3. Manual Holiday Check (2026 Sample)
            # You can expand this list or fetch it from a JSON
            holidays_2026 = [
                "2026-01-26", # Republic Day
                "2026-03-06", # Holi (Approx)
                "2026-04-02", # Good Friday
                "2026-05-01", # Maharashtra Day
                "2026-08-15", # Independence Day
                "2026-10-02", # Gandhi Jayanti
                "2026-10-21", # Dussehra (Approx)
                "2026-11-09"  # Diwali (Approx)
            ]
            
            today_str = now.strftime('%Y-%m-%d')
            if today_str in holidays_2026:
                return "HOLIDAY"
            
            # 4. Market Hours: 09:15 to 15:30 IST
            # Using total minutes from midnight for easier comparison
            curr_min = now.hour * 60 + now.minute
            open_min = 9 * 60 + 15
            close_min = 15 * 60 + 30
            
            if curr_min < open_min:
                return "PRE-OPEN"
            elif curr_min > close_min:
                return "CLOSED"
            else:
                return "OPEN"
                
        except Exception as e:
            print(f"  [!] Market Status Logic Error: {e}")
            return "UNKNOWN"
            
    def fetch_data_upstox(self, ticker, timeframe='1M'):
        """
        Fetch historical data using Upstox API with timeframe support
        """
        if not self.upstox_token:
            return None, None
            
        print(f"\n--- Fetching via Upstox: {ticker} ({timeframe}) ---")
        
        # Clean ticker
        clean_ticker = ticker.split('.')[0]
        instrument_key = ""
        
        # Check mapping for indices
        index_map = {
            'NIFTY 50': 'NSE_INDEX|Nifty 50',
            'NIFTY BANK': 'NSE_INDEX|Nifty Bank',
            'NIFTY IT': 'NSE_INDEX|Nifty IT',
            'NIFTY AUTO': 'NSE_INDEX|Nifty Auto',
            'NIFTY 500': 'NSE_INDEX|Nifty 500',
            'SENSEX': 'BSE_INDEX|SENSEX'
        }
        
        upper_ticker = ticker.upper()
        if upper_ticker in self.index_mapping:
            instrument_key = index_map.get(self.index_mapping[upper_ticker])
        elif upper_ticker in index_map:
            instrument_key = index_map[upper_ticker]
        
        if not instrument_key:
            # Check if it's a BSE stock (ends with .BO from user request logic)
            if ticker.upper().endswith('.BO'):
                instrument_key = f"BSE_EQ|{clean_ticker}"
            else:
                instrument_key = f"NSE_EQ|{clean_ticker}"
        
        print(f"  [DEBUG] Upstox Instrument Key: {instrument_key}")
        
        try:
            # Map timeframes to Upstox intervals and history windows
            # format: (interval, history_days)
            tf_map = {
                '1min': ('1minute', 2),
                '5min': ('5minute', 7),
                '1W': ('day', 30),
                '1M': ('day', 500),
                '1Y': ('day', 400),
                'ALL': ('day', 1825)
            }
            
            interval, history_days = tf_map.get(timeframe, ('day', 31))
            
            # Special handling for Indices - they don't like long history via some scrapers
            # But Upstox is usually fine. We'll stick to requested tf.
            
            today = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=history_days)).strftime('%Y-%m-%d')
            
            api_response = self.upstox_api.get_historical_candle_data1(
                instrument_key, interval, today, start_date, "2.0"
            )
            
            if api_response.status == 'success' and api_response.data.candles:
                candles = api_response.data.candles
                df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                df = df[::-1].reset_index(drop=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                logging.info(f"  [OK] Fetched {len(df)} {interval} candles")
                
                return df, MockTicker(ticker)
                
        except Exception as e:
            print(f"  [!] Upstox failed: {e}")
            
        return None, None

    def get_upstox_backtest_data(self, ticker, from_date, to_date):
        """
        Fetch historical data from Upstox specifically for backtesting
        from_date: YYYY-MM-DD
        to_date: YYYY-MM-DD
        """
        if not self.upstox_token:
             logging.error("Upstox token missing in get_upstox_backtest_data")
             return None

        # Clean ticker
        clean_ticker = ticker.split('.')[0].replace('.NS', '').replace('.BO', '')
        instrument_key = ""
        
        index_map = {
            'NIFTY 50': 'NSE_INDEX|Nifty 50',
            'NIFTY BANK': 'NSE_INDEX|Nifty Bank',
            'SENSEX': 'BSE_INDEX|SENSEX',
            'FINNIFTY': 'NSE_INDEX|Nifty Fin Service',
            'MIDCPNIFTY': 'NSE_INDEX|Nifty Midcap 100'
        }
        
        # Normalize ticker for index matching
        upper_ticker = ticker.upper().replace('^NSEI', 'NIFTY 50').replace('^NSEBANK', 'NIFTY BANK').replace('^BSESN', 'SENSEX')
        
        if upper_ticker in self.index_mapping:
            instrument_key = index_map.get(self.index_mapping[upper_ticker])
        elif upper_ticker in index_map:
            instrument_key = index_map[upper_ticker]
        
        if not instrument_key:
            if ticker.upper().endswith('.BO'):
                instrument_key = f"BSE_EQ|{clean_ticker}"
            else:
                instrument_key = f"NSE_EQ|{clean_ticker}"

        print(f"  [UPSTOX-BT] Fetching {instrument_key} from {from_date} to {to_date}")
        
        try:
            api_response = self.upstox_api.get_historical_candle_data1(
                instrument_key, "day", to_date, from_date, "2.0"
            )
            
            if api_response.status == 'success' and api_response.data.candles:
                candles = api_response.data.candles
                df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
                df = df[::-1].reset_index(drop=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
                
        except Exception as e:
            print(f"  [!] Upstox Backtest API Error for {instrument_key}: {e}")
            logging.error(f"Upstox Backtest API Error: {e}")
            
        return None

    def fetch_data_nselib(self, ticker):
        """Fetch data from nselib for Indian stocks and indices (FREE)"""
        try:
            print(f"\n  Fetching from nselib for {ticker}...")
            
            # Clean ticker - remove suffixes
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('.BSE', '')
            
            if clean_ticker.upper() in self.index_mapping:
                clean_ticker = self.index_mapping[clean_ticker.upper()]
            
            # Identify if it's an index
            indices = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT', 'SENSEX', 'NIFTY AUTO', 'NIFTY PHARMA', 'NIFTY FMCG', 'NIFTY REALTY', 'NIFTY METAL', 'NIFTY ENERGY', 'NIFTY INFRA', 'NIFTY PSE', 'NIFTY CPSE', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200', 'NIFTY 500', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100', 'NIFTY SMALLCAP 100']
            
            is_index = any(idx in clean_ticker.upper() for idx in indices)
            
            print(f"  Fetching {'Index' if is_index else 'Equity'} data for: {clean_ticker}")
            
            # Smart History: Fetch only what's needed for indices to avoid Resource errors
            # Equities need 450 days for SMA 200, Indices use shorter history as requested
            history_days = 30 if is_index else 450
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=history_days)
            start_str = start_date.strftime('%d-%m-%Y')
            end_str = end_date.strftime('%d-%m-%Y')
            
            print(f"  Requesting {history_days}-day data from {start_str} to {end_str}...")
            
            try:
                data = None
                if is_index:
                    # Use nsepython for better index historical stability
                    try:
                        print(f"  Attempting nsepython index_history for: {clean_ticker}")
                        data = index_history(clean_ticker, start_str, end_str)
                    except Exception as nse_e:
                        print(f"  [!] nsepython index failed: {nse_e}")
                        # Next attempt yfinance as it is often more stable for indices
                        try:
                            yf_ticker = "^NSEI" if "50" in clean_ticker else ("^NSEBANK" if "BANK" in clean_ticker else None)
                            if yf_ticker:
                                print(f"  [!] Attempting yfinance fallback for index: {yf_ticker}")
                                yf_data = yf.download(yf_ticker, start=start_date, end=end_date)
                                if not yf_data.empty:
                                    data = yf_data.reset_index()
                        except Exception as yf_e:
                            print(f"  [!] yfinance index failed: {yf_e}")
                            # Final resort fallback for indices
                            data = capital_market.index_data(clean_ticker, start_str, end_str)
                    
                    column_mapping = {
                        'HistoricalDate': 'Date', 'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume',
                        'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume',
                        'Index Name': 'Index', 'INDEX_NAME': 'Index'
                    }
                else:
                    # nselib's price_volume_and_deliverable_position_data for equities
                    data = capital_market.price_volume_and_deliverable_position_data(clean_ticker, start_str, end_str)
                    column_mapping = {
                        'Date': 'Date', 'OpenPrice': 'Open', 'HighPrice': 'High', 'LowPrice': 'Low', 'ClosePrice': 'Close', 'TotalTradedQuantity': 'Volume'
                    }
                
                if data is not None and not data.empty:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Log columns for debugging
                    print(f"  Columns found: {df.columns.tolist()}")
                    
                    # Check which columns exist and rename them
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df.rename(columns={old_col: new_col}, inplace=True)
                    
                    # Fallback for Volume if the specific mapping failed
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
                        
                        # Convert to numeric
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            # Remove commas from strings if any
                            if df[col].dtype == object:
                                df[col] = df[col].str.replace(',', '')
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df = df.dropna()
                        
                        if len(df) > 50:
                            print(f"  [OK] SUCCESS: nselib - {len(df)} data points")
                            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
                            return df, None
                        else:
                            print(f"  [X] Insufficient data: Only {len(df)} points")
                            return None, None
                    else:
                        print(f"  [X] Missing required columns in response. Found: {df.columns.tolist()}")
                        return None, None
                else:
                    print(f"  [X] No data returned from nselib")
                    return None, None
                    
            except Exception as e:
                error_msg = str(e)
                print(f"  [X] nselib error: {error_msg[:120]}")
                return None, None
                
        except Exception as e:
            print(f"  [X] Unexpected error: {str(e)[:100]}")
            return None, None
    
    def fetch_data(self, ticker, timeframe='1M'):
        """
        Orchestrate data fetching from multiple sources
        """
        ticker = ticker.strip().upper()
        
        # User preference: Convert to BSE for analysis consistency
        original_ticker = ticker
        is_index = any(idx in ticker for idx in ['NIFTY', 'BANKNIFTY', 'SENSEX', '^NSEI', '^NSEBANK', 'SENSEX', 'BSESN'])
        # Suffix handling for Indian stocks
        if not is_index and not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            # Default to BSE as requested by user
            ticker = ticker + ".BO"
            print(f"  [INFO] Converting {original_ticker} to BSE ({ticker}) for consistency")
        
        print(f"  [FETCH] Attempting to fetch data for: {ticker}")

        print(f"\n{'='*60}")
        print(f"FETCHING DATA FOR: {ticker} ({timeframe})")
        print(f"{'='*60}")
        
        stock_name = original_ticker
        
        # 1. Try Upstox for Indian Stocks first (if token provided)
        if self.upstox_token:
            df, ticker_obj = self.fetch_data_upstox(ticker.split('.')[0], timeframe=timeframe)
            if df is not None and len(df) >= 5:
                print("  Success: Using Upstox Data")
                return df, MockTicker(original_ticker), ticker

        # 2. Try yfinance (Robust Fallback & Primary for BSE)
        try:
            print(f"--- Checking yfinance (Primary for {ticker}) ---")
            # Map timeframe to yfinance period/interval
            tf_map = {
                '1min': ('1d', '1m'),
                '5min': ('5d', '5m'),
                '1W': ('1mo', '1d'),
                '1M': ('6mo', '1d'),
                '1Y': ('2y', '1d'),
                'ALL': ('max', '1d')
            }
            period, interval = tf_map.get(timeframe, ('6mo', '1d'))
            
            # Map index symbols for yfinance
            yf_ticker = ticker
            if 'SENSEX' in ticker: 
                yf_ticker = "^BSESN"
                stock_name = "BSE SENSEX"
            elif 'NIFTY' in ticker and '50' in ticker: 
                yf_ticker = "^NSEI"
                stock_name = "NIFTY 50"
            elif 'BANK' in ticker: 
                yf_ticker = "^NSEBANK"
                stock_name = "NIFTY BANK"
            
            # Fetch data
            df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
            
            # Try to get name from yf Metadata (Optional/Fast)
            if not is_index:
                try:
                    t = yf.Ticker(yf_ticker)
                    stock_name = t.info.get('longName', original_ticker)
                except:
                    pass

            if not df.empty and len(df) >= 10: # Indices might have less on short TF
                print(f"  Success: Using yfinance Data ({len(df)} points)")
                return df, MockTicker(stock_name), ticker
        except Exception as e:
            print(f"  [!] yfinance failed: {e}")

        # 3. Try nselib (Last resort Fallback)
        print("--- Checking nselib (Indian Market Fallback) ---")
        df, ticker_obj = self.fetch_data_nselib(ticker)
        if df is not None and len(df) >= 50:
            print("  Success: Using nselib Data")
            return df, MockTicker(stock_name), ticker
        
        # NO DATA AVAILABLE
        print(f"\n{'='*60}")
        print(f"[X] ERROR: Could not fetch data for {ticker}")
        print(f"{'='*60}")
        print("\nTried sources:")
        print("  - Upstox (Primary)")
        print("  - nselib (Indian stocks & indices fallback)")
        print("\nSuggestions:")
        print("  - For Indian stocks: Use symbol with or without .NS (e.g., RELIANCE, TCS.NS)")
        print("  - For Indian indices: Use name (e.g., NIFTY 50, NIFTY BANK)")
        print(f"{'='*60}\n")
        
        return None, None, None
    
    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators (Manual High-Precision)"""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Flatten MultiIndex columns if present (common with newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Clean column names to ensure we have Open, High, Low, Close, Volume
        col_map = {c.lower(): c for c in df.columns}
        for low, orig in col_map.items():
            if low == 'open' and 'Open' not in df.columns: df['Open'] = df[orig]
            if low == 'high' and 'High' not in df.columns: df['High'] = df[orig]
            if low == 'low' and 'Low' not in df.columns: df['Low'] = df[orig]
            if low == 'close' and 'Close' not in df.columns: df['Close'] = df[orig]
            if low == 'volume' and 'Volume' not in df.columns: df['Volume'] = df[orig]

        # Final check for required columns
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                print(f"  [!] Missing required column: {col}")
                return None

        # Moving Averages - Use min_periods=1 to prevent dropping rows on short data
        df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # Additional Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Target for ML
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Fill NaNs instead of dropping everything to keep data points for short-term analysis
        df = df.ffill().bfill()
        
        return df
    
    def calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    
    def ensemble_prediction(self, df):
        """Use ensemble of models for robust predictions"""
        features = ['SMA_10', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 
                   'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                   'BB_Width', 'BB_Upper', 'BB_Lower',
                   'Stoch_K', 'Stoch_D', 'ATR', 'Volume_Ratio', 'Momentum', 'ROC']
        
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 8:
            print(f"WARNING: Only {len(available_features)} features available")
            return None
        
        df_clean = df[available_features + ['Target']].dropna()
        
        if len(df_clean) < 5:
            print(f"WARNING: Not enough non-NaN samples ({len(df_clean)})")
            return None
            
        X = df_clean[available_features].values
        y = df_clean['Target'].values
        
        split = int(len(df) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=3,
            random_state=42
        )
        
        print("Training Random Forest model...")
        rf_model.fit(X_train, y_train)
        
        print("Training Gradient Boosting model...")
        gb_model.fit(X_train, y_train)
        
        latest = X[-1:].reshape(1, -1)
        rf_pred = rf_model.predict_proba(latest)[0][1]
        gb_pred = gb_model.predict_proba(latest)[0][1]
        
        rf_acc = rf_model.score(X_test, y_test)
        gb_acc = gb_model.score(X_test, y_test)
        
        total_acc = rf_acc + gb_acc
        rf_weight = rf_acc / total_acc if total_acc > 0 else 0.5
        gb_weight = gb_acc / total_acc if total_acc > 0 else 0.5
        
        ensemble_prob = (rf_pred * rf_weight + gb_pred * gb_weight)
        
        print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
        print(f"Gradient Boosting Accuracy: {gb_acc*100:.2f}%")
        print(f"Ensemble Prediction: {ensemble_prob*100:.1f}%")
        
        feature_importance = dict(zip(available_features, rf_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 Important Features: {[f[0] for f in top_features]}")
        
        return {
            'probability': ensemble_prob,
            'rf_accuracy': rf_acc,
            'gb_accuracy': gb_acc,
            'avg_accuracy': (rf_acc + gb_acc) / 2,
            'rf_weight': rf_weight,
            'gb_weight': gb_weight,
            'top_features': top_features
        }
    
    def analyze_sentiment(self, ticker):
        """Advanced sentiment analysis from Google News RSS"""
        try:
            news_list = get_google_news(ticker)
            if not news_list:
                return 0, "Neutral", []
            
            sentiments = []
            for item in news_list:
                analysis = TextBlob(item['title'])
                sentiments.append(analysis.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            if avg_sentiment > 0.01:
                label = "Bullish"
            elif avg_sentiment < -0.01:
                label = "Bearish"
            else:
                label = "Neutral"
            
            return round(float(avg_sentiment), 3), label, news_list
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0, "Neutral", []
    
    def calculate_risk_metrics(self, df):
        """Calculate risk and volatility metrics"""
        returns = df['Close'].pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252 - 0.02) / volatility if volatility > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        var_95 = np.percentile(returns, 5)
        
        return {
            'volatility': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'var_95': round(var_95 * 100, 2)
        }

    def get_upstox_option_chain(self, instrument_key, target_expiry=None):
        """Fetches option chain data with Greeks from Upstox"""
        
        # Helper to reload token if missing or invalid
        def reload_token():
            # In production (Render), keys are in the OS environment.
            # Only try loading from .env if the key isn't already there.
            token = os.getenv("UPSTOX_ACCESS_TOKEN")
            if not token:
                from dotenv import load_dotenv
                load_dotenv()
                token = os.getenv("UPSTOX_ACCESS_TOKEN")
            
            self.upstox_token = token
            self.headers = {
                'Authorization': f'Bearer {self.upstox_token}',
                'Accept': 'application/json'
            }
            return self.upstox_token

        def clean_num(val):
            try:
                return float(val) if val is not None else 0.0
            except: 
                return 0.0
        if not self.upstox_token:
            reload_token()

        # Proactive reload if the token looks stale or we want to ensure latest from .env
        reload_token()

        print(f"Fetching Greeks for: {instrument_key}")
            
        if not self.upstox_token:
            print("  [ERROR] No Upstox token found after reload attempt")
            return None
            
        try:
            # 1. Fetch expiry dates
            expiry_url = f"{self.base_url}/option/contract"
            params = {'instrument_key': instrument_key}
            
            exp_resp = requests.get(expiry_url, headers=self.headers, params=params)
            exp_data = exp_resp.json()
            
            # Check for token error and try one more time if so
            if exp_data.get('status') == 'error' and any(e.get('error_code') == 'UDAPI100050' for e in exp_data.get('errors', [])):
                print("  [INFO] Detected Invalid Token error. Attempting dynamic reload...")
                reload_token()
                exp_resp = requests.get(expiry_url, headers=self.headers, params=params)
                exp_data = exp_resp.json()

            if exp_data.get('status') != 'success':
                print(f"  [ERROR] Expiry API failed for {instrument_key}: {exp_data.get('errors')}")
                return None
                
            if not exp_data.get('data'):
                print(f"  [ERROR] No contract data found for {instrument_key}")
                return None
                
            # Extract unique expiry dates and pick the nearest one that is today or in the future
            all_expiries = sorted(list(set([item['expiry'] for item in exp_data['data']])))
            today_str = datetime.now().strftime('%Y-%m-%d')
            future_expiries = [e for e in all_expiries if e >= today_str]
            
            if not future_expiries:
                print(f"  [ERROR] No current/future expiry dates found for {instrument_key}.")
                if not all_expiries: return None
                nearest_expiry = all_expiries[0]
                future_expiries = [nearest_expiry]
            else:
                # If a specific expiry was requested, use it; otherwise use the nearest
                nearest_expiry = target_expiry if (target_expiry and target_expiry in future_expiries) else future_expiries[0]
                
            print(f"  [OK] Using Expiry: {nearest_expiry}")
            
            # 2. Fetch the chain
            chain_url = f"{self.base_url}/option/chain"
            params = {
                'instrument_key': instrument_key,
                'expiry_date': nearest_expiry
            }
            
            response = requests.get(chain_url, headers=self.headers, params=params)
            data = response.json()
            
            if data.get('status') == 'success':
                options = data['data']
                if not options:
                    print("  [ERROR] Chain data 'data' field is empty")
                    return None
                    
                total_pe_oi = 0
                total_ce_oi = 0
                chain_summary = []
                underlying_price = options[0].get('underlying_spot_price', 0) if options else 0
                
                # Sort options by strike price
                options = sorted(options, key=lambda x: x['strike_price'])
                
                # Find ATM index
                atm_index = 0
                min_diff = float('inf')
                for i, op in enumerate(options):
                    diff = abs(op['strike_price'] - underlying_price)
                    if diff < min_diff:
                        min_diff = diff
                        atm_index = i
                
                # Take range around ATM
                start_idx = max(0, atm_index - 10)
                end_idx = min(len(options), atm_index + 11)
                selected_options = options[start_idx:end_idx]

                # Calculate totals from ALL options
                for op in options:
                    ce_data = op.get('call_options', {})
                    pe_data = op.get('put_options', {})
                    total_ce_oi += ce_data.get('market_data', {}).get('oi', 0)
                    total_pe_oi += pe_data.get('market_data', {}).get('oi', 0)

                # Process selected ATM strikes
                for op in selected_options:
                    ce_data = op.get('call_options', {})
                    pe_data = op.get('put_options', {})
                    ce_oi = clean_num(ce_data.get('market_data', {}).get('oi', 0))
                    pe_oi = clean_num(pe_data.get('market_data', {}).get('oi', 0))
                    
                    strike = op['strike_price']
                    ce_greeks = ce_data.get('option_greeks', {})
                    pe_greeks = pe_data.get('option_greeks', {})
                    

                    chain_summary.append({
                        'strike': strike,
                        'ce_oi': ce_oi,
                        'pe_oi': pe_oi,
                        'ce_ltp': clean_num(ce_data.get('market_data', {}).get('ltp')),
                        'pe_ltp': clean_num(pe_data.get('market_data', {}).get('ltp')),
                        'ce_iv': clean_num(ce_greeks.get('iv')),
                        'pe_iv': clean_num(pe_greeks.get('iv')),
                        'ce_delta': clean_num(ce_greeks.get('delta')),
                        'pe_delta': clean_num(pe_greeks.get('delta')),
                        'ce_theta': clean_num(ce_greeks.get('theta')),
                        'pe_theta': clean_num(pe_greeks.get('theta')),
                        'ce_gamma': clean_num(ce_greeks.get('gamma')),
                        'pe_gamma': clean_num(pe_greeks.get('gamma')),
                        'ce_vega': clean_num(ce_greeks.get('vega')),
                        'pe_vega': clean_num(pe_greeks.get('vega'))
                    })

                pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
                print(f"  [SUCCESS] Returned {len(chain_summary)} strikes")
                
                return {
                    'expiry': nearest_expiry,
                    'all_expiries': future_expiries,
                    'total_ce_oi': total_ce_oi,
                    'total_pe_oi': total_pe_oi,
                    'pcr': round(pcr, 4),
                    'underlying_price': underlying_price,
                    'items': chain_summary
                }
            else:
                print(f"  [ERROR] Chain API failed for {instrument_key}: {data.get('errors')}")
        except Exception as e:
            print(f"  [CRITICAL] Exception for {instrument_key}: {str(e)}")
            
        return None
    
    def comprehensive_analysis(self, ticker, timeframe='1M'):
        """Main function for complete AI analysis - REAL DATA ONLY"""
        print(f"\n{'='*60}")
        print(f"TRADE X - ADVANCED ANALYSIS: {ticker} ({timeframe})")
        print(f"{'='*60}\n")
        
        result = self.fetch_data(ticker, timeframe=timeframe)
        
        # Safety check: if fetch_data failed to return a valid result
        if result is None or result[0] is None:
            return {
                "success": False,
                "error": f"Could not fetch real market data for '{ticker}'. Please verify the ticker symbol is listed on BSE (e.g., RELIANCE, TCS, INFY) and try again.",
                "suggestions": [
                    "For Indian stocks: Use symbol directly (we search BSE first)",
                    "For Indian indices: Use name (e.g., NIFTY 50, SENSEX)",
                    "Ensure you have a stable internet connection"
                ]
            }
        
        df, stock_obj, final_ticker = result
        
        # Determine if it's an index or short timeframe
        final_ticker_upper = final_ticker.upper()
        is_index = any(idx in final_ticker_upper for idx in ['NIFTY', 'BANKNIFTY', 'SENSEX', '^NSEI', '^NSEBANK', 'BSESN'])
        is_short_tf = timeframe in ['1min', '5min', '1W']
        
        min_data = 5 if (is_index or is_short_tf) else 100
        
        if len(df) < min_data:
            return {"error": f"Insufficient data for analysis. Only {len(df)} data points available."}
        
        try:
            print(f"\nProcessing {len(df)} real data points...")
            
            df = self.calculate_advanced_indicators(df)
            if df is None:
                return {"error": "Analysis failed: Technical indicators could not be calculated due to missing market data."}
            
            min_after = 5 if (is_index or is_short_tf) else 50
            if len(df) < min_after:
                return {"error": f"Analysis failed: Not enough stable data points (Need {min_after}, have {len(df)})"}
            
            ensemble_result = self.ensemble_prediction(df)
            
            if ensemble_result is None:
                return {"error": "Failed to generate AI predictions - market patterns are currently too volatile/insufficient."}
            
            # Use original ticker for better news search
            sentiment_score, sentiment_label, news_analysis = self.analyze_sentiment(ticker)
            risk_metrics = self.calculate_risk_metrics(df)
            
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0

            # --- ADD NEW UPSTOX DATA ---
            opt_chain = None
            if self.upstox_token:
                # Map to Upstox instrument key
                instrument_key = f"NSE_EQ|{final_ticker.split('.')[0]}"
                if is_index:
                    index_map = {
                        'NIFTY 50': 'NSE_INDEX|Nifty 50',
                        'NIFTY BANK': 'NSE_INDEX|Nifty Bank',
                        'SENSEX': 'BSE_INDEX|SENSEX'
                    }
                    if 'NIFTY 50' in final_ticker_upper or '^NSEI' in final_ticker_upper:
                        instrument_key = 'NSE_INDEX|Nifty 50'
                    elif 'BANK' in final_ticker_upper or '^NSEBANK' in final_ticker_upper:
                        instrument_key = 'NSE_INDEX|Nifty Bank'
                    elif 'SENSEX' in final_ticker_upper or 'BSESN' in final_ticker_upper:
                        instrument_key = 'BSE_INDEX|SENSEX'
                elif final_ticker_upper.endswith('.BO'):
                    instrument_key = f"BSE_EQ|{final_ticker.split('.')[0]}"
                
                opt_chain = self.get_upstox_option_chain(instrument_key)
            # ---------------------------
        except Exception as e:
            print(f"  [!] Error during processing components: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Processing Error: {str(e)}"}
        
        indicators = {
            'RSI': round(df['RSI'].iloc[-1], 2),
            'MACD': round(df['MACD'].iloc[-1], 2),
            'MACD_Signal': round(df['MACD_Signal'].iloc[-1], 2),
            'Stoch_K': round(df['Stoch_K'].iloc[-1], 2),
            'ATR': round(df['ATR'].iloc[-1], 2),
            'BB_Width': round(df['BB_Width'].iloc[-1], 4),
            'Volume_Ratio': round(df['Volume_Ratio'].iloc[-1], 2)
        }
        
        # --- CONSENSUS LOGIC (Hybrid Technical + Sentiment) ---
        prob = ensemble_result['probability']
        
        # Normalize sentiment to 0-1 range (0=Bearish, 0.5=Neutral, 1=Bullish)
        norm_sentiment = (sentiment_score + 1) / 2
        
        # Weighted Consensus: 65% Technical ML, 35% News Sentiment
        consensus_prob = (prob * 0.65) + (norm_sentiment * 0.35)
        
        signal = "HOLD"
        signal_strength = "Weak"
        
        if consensus_prob > 0.60:
            signal = "BUY"
            signal_strength = "Strong" if consensus_prob > 0.70 else "Moderate"
        elif consensus_prob < 0.40:
            signal = "SELL"
            signal_strength = "Strong" if consensus_prob < 0.30 else "Moderate"
        
        # Consistency Check: If Technicals & Sentiment diverge, mark as 'Vulnerable' or 'Neutral'
        is_divergent = (prob > 0.6 and sentiment_score < -0.1) or (prob < 0.4 and sentiment_score > 0.1)
        if is_divergent:
            signal = "HOLD"
            signal_strength = "Contradictory"
        
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        
        sma_50 = df['SMA_50'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1] if 'SMA_200' in df.columns else sma_50
        
        trend = "Bullish" if current_price > sma_50 > sma_200 else ("Bearish" if current_price < sma_50 < sma_200 else "Neutral")
        
        print(f"[OK] Analysis Complete!")
        print(f" Signal: {signal} ({signal_strength})")
        print(f" Confidence: {prob*100:.1f}%")
        print(f" Sentiment: {sentiment_label}")
        
        chart_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(100).copy()
        chart_df['Date'] = chart_df.index.strftime('%Y-%m-%d')
        chart_data = chart_df.to_dict('records')
        
        return {
            'ticker': final_ticker,
            'stock_name': stock_obj.name if hasattr(stock_obj, 'name') else final_ticker,
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'prediction': 'BULLISH' if consensus_prob > 0.5 else 'BEARISH',
            'confidence': round(consensus_prob * 100, 1),
            'technical_prob': round(prob * 100, 1),
            'accuracy': round(ensemble_result['avg_accuracy'] * 100, 1),
            'signal': signal,
            'signal_strength': signal_strength,
            'sentiment': sentiment_label,
            'sentiment_score': round(sentiment_score, 2),
            'news': news_analysis[:5],
            'indicators': indicators,
            'risk_metrics': risk_metrics,
            'support': round(recent_low, 2),
            'resistance': round(recent_high, 2),
            'trend': trend,
            'chart_data': chart_data,
            'option_chain_upstox': opt_chain
        }
