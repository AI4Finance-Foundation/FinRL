# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# install finrl library
# %pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git
# Alpaca keys
from __future__ import annotations

API_KEY = ""
API_SECRET = ""
API_BASE_URL = ""

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from alpaca_papertrading import AlpacaPaperTrading
from papertrading_common import train, test, alpaca_history, DIA_history

# Import Dow Jones 30 Symbols
ticker_list = DOW_30_TICKER
env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

train(
    start_date="2022-08-25",
    end_date="2022-08-31",
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl",  # current_working_dir
    break_step=1e5,
)

account_value_erl = test(
    start_date="2022-09-01",
    end_date="2022-09-02",
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    cwd="./papertrading_erl",
    net_dimension=ERL_PARAMS["net_dimension"],
)

train(
    start_date="2022-08-25",  # After tuning well, retrain on the training and testing sets
    end_date="2022-09-02",
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl_retrain",
    break_step=2e5,
)

action_dim = len(DOW_30_TICKER)
state_dim = (
    1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
)  # Calculate the DRL state dimension manually for paper trading. amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim

paper_trading_erl = AlpacaPaperTrading(
    ticker_list=DOW_30_TICKER,
    time_interval="1Min",
    drl_lib="elegantrl",
    agent="ppo",
    cwd="./papertrading_erl_retrain",
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=1e2,
)

paper_trading_erl.run()

# Check Portfolio Performance
# ## Get cumulative return
df_erl, cumu_erl = alpaca_history(
    key=API_KEY,
    secret=API_SECRET,
    url=API_BASE_URL,
    start="2022-09-01",  # must be within 1 month
    end="2022-09-12",
)  # change the date if error occurs

df_djia, cumu_djia = DIA_history(start="2022-09-01")
returns_erl = cumu_erl - 1
returns_dia = cumu_djia - 1
returns_dia = returns_dia[: returns_erl.shape[0]]

# plot and save
import matplotlib.pyplot as plt

plt.figure(dpi=1000)
plt.grid()
plt.grid(which="minor", axis="y")
plt.title("Stock Trading (Paper trading)", fontsize=20)
plt.plot(returns_erl, label="ElegantRL Agent", color="red")
# plt.plot(returns_sb3, label = 'Stable-Baselines3 Agent', color = 'blue' )
# plt.plot(returns_rllib, label = 'RLlib Agent', color = 'green')
plt.plot(returns_dia, label="DJIA", color="grey")
plt.ylabel("Return", fontsize=16)
plt.xlabel("Year 2021", fontsize=16)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(78))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
ax.xaxis.set_major_formatter(
    ticker.FixedFormatter(["", "10-19", "", "10-20", "", "10-21", "", "10-22"])
)
plt.legend(fontsize=10.5)
plt.savefig("papertrading_stock.png")
