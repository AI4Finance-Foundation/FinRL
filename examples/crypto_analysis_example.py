import ccxt
import pandas as pd
from datetime import datetime, timedelta

def analyze_crypto_prices(exchange_id='binance', symbol='BTC/USDT'):
    """
    Analyze cryptocurrency prices without needing an account
    """
    # Initialize exchange (public API only)
    exchange = getattr(ccxt, exchange_id)()
    
    print(f"Fetching {symbol} price data from {exchange_id}...")
    
    # Fetch OHLCV data (Open, High, Low, Close, Volume)
    timeframe = '1d'  # 1 day candlesticks
    limit = 30  # Last 30 days
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print("\nPrice Analysis:")
        print(f"Current Price: ${df['close'].iloc[-1]:,.2f}")
        print(f"30-day High: ${df['high'].max():,.2f}")
        print(f"30-day Low: ${df['low'].min():,.2f}")
        print(f"30-day Volume: ${df['volume'].sum():,.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def setup_trading_client(exchange_id='binance', api_key=None, api_secret=None):
    """
    Setup a trading client with API keys
    """
    if api_key and api_secret:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            # Uncomment for paper trading
            # 'test': True  # Use testnet/sandbox if available
        })
        
        try:
            # Test API connection
            balance = exchange.fetch_balance()
            print("\nSuccessfully connected to exchange!")
            print("Available Balances:")
            # Show only non-zero balances
            for currency in balance['total']:
                if balance['total'][currency] > 0:
                    print(f"{currency}: {balance['total'][currency]}")
            return exchange
            
        except Exception as e:
            print(f"\nError connecting to exchange: {e}")
            print("Please check your API keys and permissions")
            return None
    else:
        print("\nNo API keys provided. Only price analysis will be available.")
        print("To enable trading:")
        print("1. Create an account on your chosen exchange (e.g., Binance)")
        print("2. Complete KYC verification")
        print("3. Generate API keys from your account settings")
        print("4. Enable trading permissions for the API keys")
        return None

def main():
    # Example usage for price analysis (no account needed)
    print("=== Analyzing Crypto Prices (No Account Required) ===")
    df = analyze_crypto_prices()
    
    # Example of setting up trading (requires API keys)
    print("\n=== Setting Up Trading Client ===")
    print("To enable trading, you would need to provide API keys:")
    exchange = setup_trading_client(
        api_key='YOUR_API_KEY',  # Replace with your API key
        api_secret='YOUR_SECRET'  # Replace with your API secret
    )

if __name__ == "__main__":
    main()
