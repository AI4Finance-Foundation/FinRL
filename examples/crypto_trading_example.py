from finrl.meta.data_processors.processor_ccxt import CCXTEngineer
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from datetime import datetime, timedelta

# Initialize data processor
ccxt_eng = CCXTEngineer()

# Define parameters
TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE = "2024-01-01"
TRADE_START_DATE = "2024-01-01"
TRADE_END_DATE = "2024-03-12"

# List of crypto pairs to trade
CRYPTO_PAIRS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

# Technical indicators to use
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "rsi",
    "cci",
    "dx",
]

def main():
    # Download and preprocess data
    print("Fetching cryptocurrency data...")
    train_data = ccxt_eng.data_fetch(
        start=TRAIN_START_DATE,
        end=TRAIN_END_DATE,
        pair_list=CRYPTO_PAIRS,
        period="1d"  # Using daily data, can be changed to '1m', '5m', '1h' etc.
    )
    
    # Create training environment
    train_env_config = {
        "price_array": train_data["close"].values,
        "tech_array": train_data[TECHNICAL_INDICATORS_LIST].values,
        "if_train": True,
    }
    
    env_instance = CryptoEnv(
        config=train_env_config,
        initial_capital=100000,  # Starting with 100k USDT
        buy_cost_pct=0.001,     # 0.1% trading fee
        sell_cost_pct=0.001,    # 0.1% trading fee
    )
    
    # Initialize agent
    agent = DRLAgent(env=env_instance)
    
    # Train PPO model
    print("Training model...")
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    trained_ppo = agent.train_model(model=model_ppo, 
                                  tb_log_name='ppo',
                                  total_timesteps=100000)
    
    print("Training finished!")
    
    # Save the model
    trained_ppo.save("ppo_crypto_trading")
    
    print("Model saved! You can now use it for trading or backtesting.")

if __name__ == "__main__":
    main()
