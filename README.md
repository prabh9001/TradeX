# 🚀 TRADE X - Advanced Trading Platform

## ✨ Premium Features

### 🎨 **Stunning UI Transformation**
- **Glassmorphism Design** - modern, premium dark theme with glass effects.
- **Smooth Animations** - Micro-interactions and transitions throughout.
- **Professional Color Scheme** - Carefully curated gradients and accents.
- **Responsive Layout** - Works beautifully on all screen sizes.

### 🤖 **Advanced AI Engine**
- **Ensemble Methods** - Combines Random Forest + Gradient Boosting for robust predictions.
- **Higher Accuracy** - Multiple models working together.
- **Confidence Scoring** - Know how reliable each prediction is.

### 📊 **Comprehensive Analytics**
- **30+ Technical Indicators** - RSI, MACD, Stochastic, ATR, Bollinger Bands, etc.
- **Multiple Moving Averages** - SMA & EMA (5, 10, 20, 50, 100, 200 periods).
- **Volume Analysis** - Volume ratios and trends.
- **Momentum Indicators** - ROC, Price momentum.

### 📰 **Advanced Sentiment Analysis**
- **News Sentiment Scoring** - AI-powered NLP analysis using actual news headlines.
- **Market Mood Indicators** - Clear Bullish/Bearish/Neutral signals.
- **Integrated News Feed** - Real-time news with automated sentiment tags.

### 📈 **Risk Management**
- **Volatility Analysis** - Annualized volatility metrics.
- **Sharpe Ratio** - Risk-adjusted return calculation.
- **Maximum Drawdown** - Historical peak-to-trough decline analysis.
- **Value at Risk (VaR)** - Statistical risk measurement (95% CI).

### 🎯 **Smart Trading Signals**
- **AI Signals** - Intelligent BUY/SELL/HOLD recommendations.
- **Signal Strength** - Strength classification (High/Medium/Low).
- **Hybrid Logic** - Combines ML predictions with technical and sentiment data.

### 📊 **Professional Charts**
- **Interactive Charts** - Full TradingView integration for professional analysis.
- **Multiple Timeframes** - Support for 1M, 5M, 1D, 1W, and more.
- **Live Updates** - High-frequency price updates every 2 seconds.

### 📋 **Integrated Portfolio & Options**
- **Database Portfolio** - Persistent storage for your holdings with live P&L.
- **Option Chain** - Live NSE Greeks (Delta, Theta, Gamma, Vega) and PCR analysis.
- **Strategy Backtester** - Test strategies against historical data using triple-fallback data sources.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file (refer to `.env.example`):
```env
UPSTOX_ACCESS_TOKEN=your_token_here
SECRET_KEY=your_flask_secret_key
```

### 3. Run the Application
```bash
python app.py
```

### 4. Open Browser
Navigate to: **http://localhost:8000**

---

## 📖 How to Use

1. **Enter Stock Symbol** - Type any ticker (e.g., RELIANCE, TCS, AAPL).
2. **Click Analyze** - The AI Engine will process technical, news, and historical data.
3. **Review Results** - Analyze the AI prediction, indicators, and risk metrics.
4. **Actionable Signals** - Use the smart signals for informed decision making.

## 🎨 Design Highlights

### Color Palette
- **Primary Blue**: `#3b82f6` - Actions & highlights
- **Success Green**: `#10b981` - Positive metrics
- **Danger Red**: `#ef4444` - Negative metrics
- **Warning Gold**: `#f59e0b` - Neutral/caution

### Typography
- **Primary Font**: Inter - Clean, modern sans-serif
- **Secondary Font**: Outfit - Elegant headers
- **Monospace**: JetBrains Mono - For numbers and data

---

## 🔧 Technical Stack

- **Backend**: Flask 3.0, scikit-learn, SQLAlchemy
- **Data APIs**: yfinance, nselib, Upstox
- **Frontend**: Pure HTML/CSS/JS (Vanilla), Chart.js, TradingView Widgets
- **AI/NLP**: Random Forest, Gradient Boosting, TextBlob

---

## 🎨 Customization

Edit `static/css/style.css` to customize:
- Colors (`:root` variables)
- Spacing
- Border radius
- Animations
- Fonts

---

**Built with ❤️ for Modern Traders**
