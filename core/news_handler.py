import feedparser
from textblob import TextBlob
import datetime

def get_google_news(symbol):
    """
    Fetches news from Google News RSS for a specific stock
    and analyzes sentiment using TextBlob.
    """
    try:
        # 1. Clean Symbol for Search Query
        # If symbol is 'TCS.NS', we change query to "TCS share price India"
        clean_symbol = symbol.replace('.NS', '').replace('BSE:', '').replace('.BO', '')
        
        # We add "stock news India" to ensure we get financial news, not random info
        query = f"{clean_symbol} stock news India"
        
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        # 2. Google News RSS URL (Free & Public)
        # 'hl=en-IN' ensures we get English news for India
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        # 3. Fetch the Feed
        feed = feedparser.parse(rss_url)
        
        news_list = []
        
        # 4. Process Top 10 Articles (increaed from 5 for better coverage)
        for entry in feed.entries[:10]:
            title = entry.title
            link = entry.link
            published = entry.published if hasattr(entry, 'published') else "Recent"
            
            # --- AI SENTIMENT ANALYSIS ---
            analysis = TextBlob(title)
            polarity = analysis.sentiment.polarity
            
            # Determine Label (Positive/Negative/Neutral)
            if polarity > 0.05: # Adjusted threshold for better sensitivity
                sentiment = "Positive"
            elif polarity < -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            news_list.append({
                "title": title,
                "link": link,
                "sentiment": sentiment,
                "date": published
            })
            
        return news_list

    except Exception as e:
        print(f"Error fetching Google News: {e}")
        return []
