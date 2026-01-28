from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from core.advanced_ai import AdvancedAIEngine
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market
import json
import traceback
import os
import datetime
import threading
import time
import logging
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("foursight.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the expected Excel/CSV file path for local data sync
EXCEL_DATA_PATH = os.path.join(os.getcwd(), 'Book1.xlsx')
CSV_DATA_PATH = os.path.join(os.getcwd(), 'Book1.xlsx')

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize AI Engine
ai_engine = AdvancedAIEngine()

app.secret_key = os.environ.get('SECRET_KEY', 'premium_trading_secret_key_123')

@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        data = request.form
        email = data.get('email')
        password = data.get('password')
        
        # Simple mock authentication
        if email and password:
            session['user'] = email
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        data = request.form
        # Mock registration
        session['user'] = data.get('email')
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout redirect"""
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    """
    Comprehensive stock analysis endpoint
    Returns: AI predictions, technical indicators, sentiment, risk metrics
    """
    try:
        data = request.json
        ticker = data.get('ticker', '').strip().upper()
        timeframe = data.get('timeframe', '1M')
        
        if not ticker:
            return jsonify({
                'success': False,
                'error': 'Please provide a valid ticker symbol'
            })
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {ticker} (Period: {timeframe})")
        print(f"{'='*60}\n")
        
        # Run comprehensive AI analysis
        result = ai_engine.comprehensive_analysis(ticker, timeframe=timeframe)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            })
        
        # Success response
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        print(f"\nERROR in /api/analyze:")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """
    Execute a backtest strategy on historical data
    """
    # Heavy imports moved to top level

    try:
        data = request.json
        ticker = data.get('ticker', 'NIFTY').upper()
        strategy = data.get('strategy', 'EMA_CROSS')
        start_date = data.get('start_date')
        end_date = data.get('end_date', datetime.datetime.now().strftime('%Y-%m-%d'))
        initial_capital_raw = data.get('initial_capital')
        initial_capital = float(initial_capital_raw) if initial_capital_raw and str(initial_capital_raw).strip() != "" else 100000.0

        # 0. Clean Ticker & Resolve Mapping
        # Strip any existing suffixes to start fresh
        base_ticker = ticker.strip().upper().replace('.NS', '').replace('.BO', '').replace('.BSE', '').replace('.NSE', '')
        
        ticker_map = {
            'NIFTY': '^NSEI', 'NIFTY50': '^NSEI', 'NIFTY 50': '^NSEI',
            'BANKNIFTY': '^NSEBANK', 'NIFTYBANK': '^NSEBANK',
            'FINNIFTY': 'NIFTY_FIN_SERVICE.NS',
            'SENSEX': '^BSESN'
        }
        
        yf_ticker = ticker_map.get(base_ticker, base_ticker)
        if yf_ticker == base_ticker and not yf_ticker.startswith('^'):
            yf_ticker = base_ticker + ".NS"

        # 1. Start Date & Buffer Logic
        from dateutil import parser as date_parser
        original_start_dt = None
        buffer_start = None
        
        try:
            now = datetime.datetime.now()
            # If end_date is today or not provided, check if it's weekend
            if not end_date or end_date == now.strftime('%Y-%m-%d'):
                # If Saturday or Sunday, use Friday
                if now.weekday() >= 5: # 5=Sat, 6=Sun
                    days_to_subtract = now.weekday() - 4
                    now = now - datetime.timedelta(days=days_to_subtract)
                end_date = now.strftime('%Y-%m-%d')
            else:
                end_date_dt = date_parser.parse(end_date)
                end_date = end_date_dt.strftime('%Y-%m-%d')

            if start_date:
                original_start_dt = date_parser.parse(start_date)
                buffer_start = (original_start_dt - datetime.timedelta(days=250)).strftime('%Y-%m-%d')
        except Exception as de:
            print(f"--- BACKTEST: Date error: {de} ---")
            if not original_start_dt:
                buffer_start = start_date

        # 2. Fetch Data with Triple Fallback (Upstox -> Yahoo -> nselib)
        df = None
        source = "None"
        
        # A. Try Upstox High-Fidelity Data first (Best for Indian Stocks)
        print(f"--- BACKTEST: [1/3] Upstox Attempt for {base_ticker} ---")
        if not ai_engine.upstox_token:
            print("  [!] Upstox token missing, skipping step 1")
        else:
            try:
                df = ai_engine.get_upstox_backtest_data(base_ticker, buffer_start or (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d'), end_date)
                if df is not None and not df.empty and len(df) >= 10:
                    source = "Upstox"
                    print(f"  [OK] source: {source}, rows: {len(df)}")
                else:
                    df = None 
            except Exception as ue:
                print(f"  [!] Upstox failed: {ue}")
                df = None

        # B. Fallback to Yahoo Finance (Good for Global + Fallback)
        if df is None:
            print(f"--- BACKTEST: [2/3] Yahoo Finance Attempt for {yf_ticker} ---")
            fetch_kwargs = {'end': end_date}
            if buffer_start: fetch_kwargs['start'] = buffer_start
            else: fetch_kwargs['period'] = '2y'

            try:
                df = yf.download(yf_ticker, **fetch_kwargs, progress=False)
                if df is None or df.empty or len(df) < 5:
                    # Try suffixes if raw fails
                    for suffix in ['.NS', '.BO', '']:
                        alt_ticker = base_ticker + suffix
                        if alt_ticker == yf_ticker: continue
                        print(f"  [!] Yahoo Fallback try {alt_ticker}...")
                        df = yf.download(alt_ticker, **fetch_kwargs, progress=False)
                        if df is not None and not df.empty and len(df) >= 10:
                            yf_ticker = alt_ticker
                            source = "Yahoo Finance"
                            break
                    else:
                        df = None
                else:
                    source = "Yahoo Finance"
                    print(f"  [OK] source: {source}, rows: {len(df)}")
            except Exception as ye:
                print(f"  [!] Yahoo error: {ye}")
                df = None

        # C. Tertiary Fallback: nselib (Ultra-Reliable for Indian Equities)
        if df is None or df.empty:
            print(f"--- BACKTEST: [3/3] nselib Tertiary Attempt for {base_ticker} ---")
            try:
                # nselib uses DD-MM-YYYY
                dt_now = datetime.datetime.now()
                # Use a very safe end date (yesterday) to avoid same-day archival issues
                safe_end = (dt_now - datetime.timedelta(days=1)).strftime('%d-%m-%Y')
                ns_start = (original_start_dt - datetime.timedelta(days=365)).strftime('%d-%m-%Y') if original_start_dt else (dt_now - datetime.timedelta(days=730)).strftime('%d-%m-%Y')
                
                # Check for Indices
                is_idx_nifty = base_ticker in ['NIFTY', 'NIFTY50', 'NSEI']
                is_idx_bank = base_ticker in ['BANKNIFTY', 'NSEBANK']
                
                ns_data = None
                if is_idx_nifty:
                    print(f"  Fetching nselib Index Data for NIFTY 50...")
                    ns_data = capital_market.index_data("NIFTY 50", ns_start, safe_end)
                elif is_idx_bank:
                    print(f"  Fetching nselib Index Data for NIFTY BANK...")
                    ns_data = capital_market.index_data("NIFTY BANK", ns_start, safe_end)
                else:
                    print(f"  Fetching nselib Equity Data for {base_ticker}...")
                    ns_data = capital_market.price_volume_and_deliverable_position_data(base_ticker, ns_start, safe_end)
                
                if ns_data is not None and not ns_data.empty:
                    df = pd.DataFrame(ns_data)
                    # Mapping: nselib columns are diverse
                    col_map = {
                        'Date': 'Date', 'HistoricalDate': 'Date',
                        'OpenPrice': 'Open', 'OPEN': 'Open',
                        'HighPrice': 'High', 'HIGH': 'High',
                        'LowPrice': 'Low', 'LOW': 'Low',
                        'ClosePrice': 'Close', 'CLOSE': 'Close',
                        'TotalTradedQuantity': 'Volume', 'VOLUME': 'Volume'
                    }
                    df.rename(columns=col_map, inplace=True)
                    
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    df = df.dropna(subset=['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Ensure numeric columns and clean commas
                    for c in ['Open', 'High', 'Low', 'Close']:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce')
                    
                    if 'Volume' in df.columns:
                        df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
                    else:
                        df['Volume'] = 0
                        
                    df = df.dropna(subset=['Close']).sort_index()
                    if not df.empty and len(df) >= 10:
                        source = "nselib"
                        print(f"  [OK] source: {source}, rows: {len(df)}")
                    else:
                        print(f"  [!] nselib returned too few rows ({len(df) if df is not None else 0})")
                        df = None
                else:
                    print(f"  [!] nselib returned empty response for {base_ticker}")
                    df = None
            except Exception as ne:
                print(f"  [!] nselib critical fail: {ne}")
                df = None

        if df is None or df.empty:
            return jsonify({
                'success': False, 
                'error': f'Backtest Data Error: Unable to retrieve history for "{base_ticker}" from Upstox, Yahoo, or NSE archives. '
                         f'This usually happens if the symbol is incorrect or the date range is too recent. '
                         f'Please try an earlier start date.'
            })

        print(f"--- BACKTEST: Final Dataset from {source} ({len(df)} rows) ---")

        if len(df) < 10:
             return jsonify({
                'success': False, 
                'error': f'Insufficient historical data for {base_ticker} (only {len(df)} days found). '
                         f'Backtesting requires at least 6-12 months of history for technical indicators (EMA/Bollinger) to work.'
            })

        # Flatten columns if multi-indexed (yfinance 0.2.0+ change)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure 'Close' is a Series and not a DF
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]

        # 3. Strategy Logic
        if strategy == 'EMA_CROSS':
            df['EMA_Fast'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_Slow'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['Signal'] = 0.0
            df['Signal'] = np.where(df['EMA_Fast'] > df['EMA_Slow'], 1.0, 0.0)
            df['Position'] = df['Signal'].diff()
            
        elif strategy == 'MEAN_REVERSION':
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD20'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + (df['STD20'] * 2)
            df['Lower'] = df['MA20'] - (df['STD20'] * 2)
            
            signal = 0.0
            signals = []
            for i in range(len(df)):
                curr_close = df['Close'].iloc[i]
                if curr_close < df['Lower'].iloc[i]:
                    signal = 1.0
                elif curr_close > df['Upper'].iloc[i]:
                    signal = 0.0
                signals.append(signal)
            df['Signal'] = signals
            df['Position'] = df['Signal'].diff()

        elif strategy == 'RSI_STRATEGY':
            # RSI Calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            signal = 0.0
            signals = []
            for i in range(len(df)):
                curr_rsi = df['RSI'].iloc[i]
                if curr_rsi < 30: # Oversold
                    signal = 1.0
                elif curr_rsi > 70: # Overbought
                    signal = 0.0
                signals.append(signal)
            df['Signal'] = signals
            df['Position'] = df['Signal'].diff()

        elif strategy == 'MACD_CROSS':
            # MACD Calculation
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            df['Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1.0, 0.0)
            df['Position'] = df['Signal'].diff()

        elif strategy == 'SUPERTREND':
            # Supertrend Calculation (Abit complex manually)
            atr_period = 10
            multiplier = 3
            
            # ATR
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            atr = tr.ewm(span=atr_period, adjust=False).mean()
            
            # Bands
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Signal
            st_signals = []
            current_st = True # True = Bullish, False = Bearish
            for i in range(len(df)):
                if close.iloc[i] > upper_band.iloc[i-1] if i > 0 else 0:
                    current_st = True
                elif close.iloc[i] < lower_band.iloc[i-1] if i > 0 else 0:
                    current_st = False
                st_signals.append(1.0 if current_st else 0.0)
            
            df['Signal'] = st_signals
            df['Position'] = df['Signal'].diff()
        else:
            return jsonify({'success': False, 'error': 'Unsupported strategy'})

        # 4. Simulation
        capital = initial_capital
        position_shares = 0
        trades = []
        equity_curve = []

        for date, row in df.iterrows():
            current_price = float(row['Close'])
            is_active = original_start_dt is None or date >= original_start_dt
            
            # Execute Trades (Only if we are in the active backtest period)
            if is_active:
                if row['Position'] == 1.0 and capital > current_price: # Buy
                    position_shares = capital // current_price
                    cost = position_shares * current_price
                    capital -= cost
                    trades.append({
                        'type': 'BUY',
                        'date': date.strftime('%Y-%m-%d'),
                        'price': round(current_price, 2),
                        'shares': position_shares
                    })
                elif row['Position'] == -1.0 and position_shares > 0: # Sell
                    revenue = position_shares * current_price
                    capital += revenue
                    
                    profit = 0
                    if len(trades) > 0 and trades[-1]['type'] == 'BUY':
                        profit = revenue - (trades[-1]['shares'] * trades[-1]['price'])
                    
                    trades.append({
                        'type': 'SELL',
                        'date': date.strftime('%Y-%m-%d'),
                        'price': round(current_price, 2),
                        'shares': position_shares,
                        'profit': round(profit, 2)
                    })
                    position_shares = 0

                # Track Equity (Only during active period)
                total_value = capital + (position_shares * current_price)
                equity_curve.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': round(total_value, 2)
                })

        # Ensure we have some equity data
        if not equity_curve:
             return jsonify({'success': False, 'error': 'No trading signals generated during this period. Try a longer timeframe.'})
        final_value = equity_curve[-1]['value'] if equity_curve else initial_capital
        total_profit = final_value - initial_capital
        roi = (total_profit / initial_capital) * 100

        # Metrics
        returns = pd.Series([e['value'] for e in equity_curve]).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty and returns.std() != 0 else 0
        
        # Drawdown
        values = np.array([e['value'] for e in equity_curve])
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        max_drawdown = np.min(drawdowns) * 100 if drawdowns.size > 0 else 0

        win_trades = [t for t in trades if t.get('type') == 'SELL' and t.get('profit', 0) > 0]
        total_closed_trades = len([t for t in trades if t.get('type') == 'SELL'])
        win_rate = (len(win_trades) / total_closed_trades * 100) if total_closed_trades > 0 else 0

        return jsonify({
            'success': True,
            'metrics': {
                'total_profit': round(total_profit, 2),
                'roi': round(roi, 2),
                'sharpe': round(float(sharpe), 2),
                'max_drawdown': round(max_drawdown, 2),
                'win_rate': round(win_rate, 2),
                'total_trades': total_closed_trades
            },
            'equity_curve': equity_curve,
            'trades': trades
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get-latest-price/<ticker>')
def get_latest_price(ticker):
    """Fetch only the latest price for high-frequency updates"""
    try:
        import yfinance as yf
        # Quick clean up
        clean_ticker = ticker.upper().strip()
        if not clean_ticker.endswith('.NS') and not clean_ticker.endswith('.BO') and not clean_ticker.startswith('^'):
            # Check if it was search as BSE
            if ':' in clean_ticker:
                symbol = clean_ticker.split(':')[-1]
                clean_ticker = symbol + (".BO" if "BSE" in clean_ticker else ".NS")
            else:
                clean_ticker += ".BO" # Default to BSE as requested

        stock = yf.Ticker(clean_ticker)
        # Use fast_info for speed
        price = stock.fast_info.get('last_price') or stock.fast_info.get('lastPrice')
        
        if price is None:
            # Fallback for some tickers
            df = stock.history(period='1d')
            if not df.empty:
                price = df['Close'].iloc[-1]

        return jsonify({
            'success': True,
            'price': round(price, 2) if price else 0,
            'time': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# In-memory storage for user portfolios (email as key)
user_portfolios = {}

@app.route('/api/portfolio', methods=['GET', 'POST', 'DELETE'])
def portfolio():
    """
    Portfolio management endpoint
    """
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    email = session['user']
    if email not in user_portfolios:
        user_portfolios[email] = []

    if request.method == 'GET':
        import yfinance as yf
        holdings = user_portfolios[email]
        total_invested = 0
        total_current_value = 0
        
        enriched_holdings = []
        for h in holdings:
            try:
                ticker = h['ticker']
                # Add .BO if it looks like an Indian stock but has no suffix
                yf_ticker = ticker
                if not any(ticker.endswith(s) for s in ['.NS', '.BO']) and not ticker.startswith('^'):
                    yf_ticker = ticker + ".BO"
                
                stock = yf.Ticker(yf_ticker)
                curr_price = stock.fast_info.get('last_price') or stock.history(period='1d')['Close'].iloc[-1]
                
                invested = h['quantity'] * h['avg_price']
                current_value = h['quantity'] * curr_price
                pnl = current_value - invested
                pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                
                total_invested += invested
                total_current_value += current_value
                
                enriched_holdings.append({
                    **h,
                    'current_price': round(curr_price, 2),
                    'pnl': round(pnl, 2),
                    'pnl_percent': round(pnl_pct, 2),
                    'type': h.get('type', 'Holding')
                })
            except Exception as e:
                print(f"Error enriching {h['ticker']}: {e}")
                enriched_holdings.append({**h, 'current_price': h['avg_price'], 'pnl': 0, 'pnl_percent': 0})

        total_pnl = total_current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        return jsonify({
            'success': True,
            'portfolio': {
                'total_invested': round(total_invested, 2),
                'total_value': round(total_current_value, 2),
                'total_pnl': round(total_pnl, 2),
                'pnl_percent': round(total_pnl_pct, 2),
                'holdings': enriched_holdings
            }
        })
    
    if request.method == 'DELETE':
        data = request.json
        ticker = data.get('ticker', '').upper().strip()
        asset_type = data.get('type', 'Holding')
        
        if not ticker:
            return jsonify({'success': False, 'error': 'Ticker is required'})
            
        initial_len = len(user_portfolios[email])
        user_portfolios[email] = [h for h in user_portfolios[email] if not (h['ticker'] == ticker and h.get('type') == asset_type)]
        
        if len(user_portfolios[email]) < initial_len:
            return jsonify({'success': True, 'message': f'Removed {ticker} from {asset_type}s'})
        else:
            return jsonify({'success': False, 'error': 'Stock not found in portfolio'})

    # POST - Add or Update holding
    data = request.json
    ticker = data.get('ticker', '').upper().strip()
    quantity = float(data.get('quantity', 0))
    avg_price = float(data.get('avg_price', 0))
    asset_type = data.get('type', 'Holding') # Holding or Position
    
    if not ticker or quantity <= 0 or avg_price <= 0:
        return jsonify({'success': False, 'error': 'Invalid stock data'})

    # Find if exists with SAME type
    found = False
    for h in user_portfolios[email]:
        if h['ticker'] == ticker and h.get('type') == asset_type:
            # Update average price: (old_q * old_p + new_q * new_p) / (old_q + new_q)
            new_total_q = h['quantity'] + quantity
            new_avg_p = (h['quantity'] * h['avg_price'] + quantity * avg_price) / new_total_q
            h['quantity'] = new_total_q
            h['avg_price'] = round(new_avg_p, 2)
            found = True
            break
    
    if not found:
        user_portfolios[email].append({
            'ticker': ticker,
            'quantity': quantity,
            'avg_price': avg_price,
            'type': asset_type
        })

    return jsonify({
        'success': True,
        'message': f'Successfully added/updated {ticker} in portfolio'
    })

@app.route('/api/screener', methods=['POST'])
def screener():
    """
    Stock screener endpoint
    Filter stocks based on criteria
    """
    try:
        criteria = request.json
        
        # Mock screener results
        results = [
            {
                'ticker': 'AAPL',
                'price': 180.50,
                'change_pct': 2.5,
                'volume': 50000000,
                'market_cap': 2800000000000,
                'pe_ratio': 28.5,
                'signal': 'BUY'
            },
            {
                'ticker': 'TCS.NS',
                'price': 3500,
                'change_pct': 1.2,
                'volume': 2000000,
                'market_cap': 1280000000000,
                'pe_ratio': 32.1,
                'signal': 'HOLD'
            }
        ]
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    Backtesting endpoint
    Test trading strategy on historical data
    """
    try:
        data = request.json
        ticker = data.get('ticker')
        strategy = data.get('strategy', 'buy_hold')
        period = data.get('period', '1y')
        
        # Mock backtest results
        return jsonify({
            'success': True,
            'backtest': {
                'ticker': ticker,
                'strategy': strategy,
                'period': period,
                'initial_capital': 10000,
                'final_value': 12500,
                'total_return': 25.0,
                'sharpe_ratio': 1.5,
                'max_drawdown': -8.5,
                'win_rate': 65.0,
                'total_trades': 45,
                'winning_trades': 29,
                'losing_trades': 16
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/market/status')
def market_status():
    """
    Get live Indian Market Status
    """
    try:
        status = ai_engine.get_market_status()
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market/movers', methods=['GET'])
def market_movers():
    """
    Get top gainers and losers
    """
    return jsonify({
        'success': True,
        'gainers': [
            {'ticker': 'NVDA', 'price': 450.50, 'change': 8.5},
            {'ticker': 'AMD', 'price': 120.30, 'change': 6.2},
            {'ticker': 'TSLA', 'price': 245.80, 'change': 5.1}
        ],
        'losers': [
            {'ticker': 'META', 'price': 320.10, 'change': -4.2},
            {'ticker': 'NFLX', 'price': 480.50, 'change': -3.8},
            {'ticker': 'AMZN', 'price': 155.20, 'change': -2.9}
        ]
    })

@app.route('/api/option-chain', methods=['GET'])
def get_option_chain():
    """
    Fetch option chain data - Supports Upstox, NSE, and local Excel (Book1.xlsx)
    """
    from nsepython import nse_optionchain_scrapper
    try:
        symbol = request.args.get('symbol', 'NIFTY').upper()
        target_expiry = request.args.get('expiry') # Optional specific expiry request
        
        # --- NEW: EXCEL SYNC (Book1.xlsx) ---
        excel_data = []
        excel_summary = None
        if os.path.exists(EXCEL_DATA_PATH):
            try:
                df_xl = pd.read_excel(EXCEL_DATA_PATH)
                # Identify Chain Name (Underlying)
                chain_name = df_xl['underlying'].iloc[0] if 'underlying' in df_xl.columns else 'N/A'
                nearest_expiry = df_xl['expiryDates'].iloc[0] if 'expiryDates' in df_xl.columns else 'N/A'
                
                # Robust Column Mapping
                # CE OI might be missing or named differently. PE OI is 'openInterest'
                cols = df_xl.columns.tolist()
                ce_oi_col = None
                pe_oi_col = 'openInterest' if 'openInterest' in cols else None
                
                # If we only have one 'openInterest', and it's surrounded by PE columns, CE is likely missing
                # But let's check if there's any Unnamed or hidden OI column
                for c in cols:
                    if ('Interest' in c or c.upper() == 'OI') and c != pe_oi_col and c != 'changeinOpenInterest' and c != 'changeinOpenInterest.1':
                        ce_oi_col = c
                        break

                for _, row in df_xl.iterrows():
                    strike = row.get('strikePrice', row.get('strikePrice.1', row.get('strikePrice.2', 0)))
                    
                    ce_buys = float(str(row.get('totalBuyQuantity', 0)).replace(',', ''))
                    ce_sells = float(str(row.get('totalSellQuantity', 0)).replace(',', ''))
                    pe_buys = float(str(row.get('totalBuyQuantity.1', 0)).replace(',', ''))
                    pe_sells = float(str(row.get('totalSellQuantity.1', 0)).replace(',', ''))
                    
                    ce_movement = "Bullish" if ce_buys > ce_sells else "Bearish"
                    pe_movement = "Bearish" if pe_buys > pe_sells else "Bullish"
                    
                    # Fix: If ce_oi_col is none, we might be missing data, but let's try to map what we can
                    excel_data.append({
                        'strikePrice': strike,
                        'expiry': str(row.get('expiryDate', row.get('expiryDates', 'N/A'))).split(' ')[0],
                        'CE': {
                            'openInterest': row.get(ce_oi_col, 0) if ce_oi_col else 0,
                            'changeinOpenInterest': row.get('changeinOpenInterest', 0),
                            'pChangeOI': row.get('pchangeinOpenInterest', 0),
                            'volume': row.get('totalTradedVolume', 0),
                            'lastPrice': row.get('lastPrice', 0),
                            'totalBuyQty': ce_buys,
                            'totalSellQty': ce_sells,
                            'movement': ce_movement
                        },
                        'PE': {
                            'openInterest': row.get(pe_oi_col, 0) if pe_oi_col else 0,
                            'changeinOpenInterest': row.get('changeinOpenInterest.1', 0),
                            'pChangeOI': row.get('pchangeinOpenInterest.1', 0),
                            'volume': row.get('totalTradedVolume.1', 0),
                            'lastPrice': row.get('lastPrice.1', 0),
                            'totalBuyQty': pe_buys,
                            'totalSellQty': pe_sells,
                            'movement': pe_movement
                        }
                    })
                
                # Find Most and Least Movement Strikes
                df_xl['total_vol'] = df_xl.get('totalTradedVolume', 0) + df_xl.get('totalTradedVolume.1', 0)
                most_active = df_xl.loc[df_xl['total_vol'].idxmax()] if not df_xl.empty else None
                least_active = df_xl.loc[df_xl['total_vol'].idxmin()] if not df_xl.empty else None

                excel_summary = {
                    'source': 'Excel (Book1.xlsx)',
                    'chain_name': chain_name if chain_name != 'N/A' else symbol,
                    'expiry': str(nearest_expiry).split(' ')[0],
                    'total_rows': len(excel_data),
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'most_active_strike': float(most_active.get('strikePrice', 0)) if most_active is not None else 0,
                    'least_active_strike': float(least_active.get('strikePrice', 0)) if least_active is not None else 0,
                    'pcr': '--',
                    'underlying_value': '--'
                }
            except Exception as ex_err:
                print(f"Excel read error: {ex_err}")
        # ------------------------------------

        logger.info(f"Option Chain Requested: {symbol}")
        
        upstox_insights = None
        u_key = None
        if symbol == 'NIFTY': u_key = 'NSE_INDEX|Nifty 50'
        elif symbol == 'BANKNIFTY': u_key = 'NSE_INDEX|Nifty Bank'
        elif symbol == 'SENSEX': u_key = 'BSE_INDEX|SENSEX'
        elif symbol == 'FINNIFTY': u_key = 'NSE_INDEX|Nifty Fin Service'
        elif symbol == 'MIDCPNIFTY': u_key = 'NSE_INDEX|Nifty Midcap 100'

        if u_key:
            try:
                upstox_insights = ai_engine.get_upstox_option_chain(u_key, target_expiry=target_expiry)
            except Exception as ue:
                print(f"Upstox fetch error: {ue}")

        chain_data = None
        if symbol != 'SENSEX' and not excel_data:
            try:
                chain_data = nse_optionchain_scrapper(symbol)
            except Exception as ne:
                print(f"NSE fetch failed: {ne}")

        if upstox_insights:
            summary = {
                'chain_name': symbol,
                'total_ce_oi': upstox_insights.get('total_ce_oi', 0),
                'total_pe_oi': upstox_insights.get('total_pe_oi', 0),
                'pcr': upstox_insights.get('pcr', '--'),
                'underlying_value': upstox_insights.get('underlying_price', "See Header"),
                'expiry': upstox_insights.get('expiry', '--'),
                'timestamp': "Live (Upstox)"
            }
            processed_data = []
            for item in upstox_insights.get('items', []):
                processed_data.append({
                    'strikePrice': item.get('strike', 0),
                    'expiry': upstox_insights.get('expiry', 'N/A'),
                    'CE': {
                        'openInterest': item.get('ce_oi') or 0,
                        'lastPrice': item.get('ce_ltp') or 0,
                        'movement': 'N/A'
                    },
                    'PE': {
                        'openInterest': item.get('pe_oi') or 0,
                        'lastPrice': item.get('pe_ltp') or 0,
                        'movement': 'N/A'
                    }
                })
        else:
            summary = excel_summary or {}
            processed_data = excel_data or []

        if not processed_data and chain_data:
            try:
                summary = {
                    'total_ce_oi': chain_data['filtered']['CE']['totOI'],
                    'total_pe_oi': chain_data['filtered']['PE']['totOI'],
                    'pcr': round(chain_data['filtered']['PE']['totOI'] / chain_data['filtered']['CE']['totOI'], 2) if chain_data['filtered']['CE']['totOI'] > 0 else 0,
                    'underlying_value': chain_data['records']['underlyingValue'],
                    'timestamp': chain_data['records']['timestamp']
                }
                processed_data = chain_data['filtered']['data'][:20]
            except:
                pass


        if not processed_data:
             return jsonify({'success': False, 'error': 'Market data currently unavailable.'})

        return jsonify({
            'success': True,
            'summary': summary,
            'upstox_insights': upstox_insights,
            'data': processed_data,
            'all_expiries': upstox_insights.get('all_expiries', []) if upstox_insights else [],
            'excel_sync': (excel_summary is not None)
        })

    except Exception as e:
        print(f"Option Chain Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

    except Exception as e:
        print(f"Option Chain Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/market/indices', methods=['GET'])
def market_indices():
    """
    Get major market indices using Upstox (Primary) or nselib (Fallback)
    """
    upstox_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    results = []

    # 1. Try Upstox for high-quality real-time data
    if upstox_token:
        try:
            import upstox_client
            config = upstox_client.Configuration()
            config.access_token = upstox_token
            market_api = upstox_client.MarketQuoteApi(upstox_client.ApiClient(config))
            
            # Request quotes for major indices and large caps
            # Indices: NSE_INDEX|Nifty 50, NSE_INDEX|Nifty Bank, etc.
            # Large Caps: NSE_EQ|RELIANCE, NSE_EQ|TCS, NSE_EQ|HDFCBANK
            items = [
                "NSE_INDEX|Nifty 50", "NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty IT",
                "NSE_EQ|RELIANCE", "NSE_EQ|TCS", "NSE_EQ|HDFCBANK", 
                "NSE_EQ|INFY", "NSE_EQ|ICICIBANK", "NSE_EQ|ZOMATO",
                "NSE_EQ|ADANIENT", "NSE_EQ|MARUTI", "NSE_EQ|SUNPHARMA",
                "NSE_EQ|WIPRO", "NSE_EQ|HINDUNILVR", "NSE_EQ|AXISBANK",
                "NSE_EQ|TITAN", "NSE_EQ|ULTRACEMCO", "NSE_EQ|LT"
            ]
            index_keys = ",".join(items)
            api_response = market_api.get_full_market_quote(index_keys)
            
            if api_response.status == 'success':
                for key, data in api_response.data.items():
                    # Handle both Index and Equity naming
                    display_name = key.split('|')[1].upper()
                    results.append({
                        'name': display_name,
                        'value': data.last_price,
                        'change': round(data.net_change_percentage, 2)
                    })
                print(f"  [OK] Fetched {len(results)} items for banner from Upstox")
        except Exception as e:
            print(f"Upstox indices error: {e}")

    # 2. Fallback to nselib if Upstox failed or no token
    if not results:
        from nselib import capital_market
        try:
            indices_data = capital_market.market_watch_all_indices()
            if indices_data is not None and not indices_data.empty:
                major_indices = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT', 'NIFTY NEXT 50', 'NIFTY MIDCAP 100']
                for idx, row in indices_data.iterrows():
                    index_name = row.get('index', '')
                    if index_name in major_indices:
                        results.append({
                            'name': index_name,
                            'value': row.get('last', 0),
                            'change': row.get('percChange', 0)
                        })
        except Exception as e:
            print(f"nselib fallback error: {e}")

    # 3. Last Resort Fallback (Extended Mock for Premium Look)
    if not results:
        results = [
            {'name': 'NIFTY 50', 'value': 21820.50, 'change': 0.85},
            {'name': 'NIFTY BANK', 'value': 48550.30, 'change': 1.12},
            {'name': 'RELIANCE', 'value': 2950.40, 'change': 1.45},
            {'name': 'TCS', 'value': 3820.15, 'change': -0.32},
            {'name': 'HDFCBANK', 'value': 1650.75, 'change': 0.65},
            {'name': 'ZOMATO', 'value': 125.40, 'change': 2.10},
            {'name': 'ADANIENT', 'value': 3150.20, 'change': -1.05},
            {'name': 'INFY', 'value': 1580.45, 'change': 0.25},
            {'name': 'MARUTI', 'value': 10250.00, 'change': 0.88},
            {'name': 'TATAMOTORS', 'value': 950.30, 'change': 1.15}
        ]
        
    return jsonify({
        'success': True,
        'indices': results
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Foursight AI',
        'version': '2.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TRADE X - ADVANCED TRADING PLATFORM")
    print("="*60)
    print("Server starting on http://localhost:8000")
    print("AI Engine: READY")
    print("Features: Predictions | Sentiment | Risk Analysis")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8000, host='0.0.0.0')
