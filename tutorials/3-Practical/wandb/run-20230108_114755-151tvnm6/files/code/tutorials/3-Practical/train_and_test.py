from finrl.train import train, load_df
from finrl.test import test
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.plot import backtest_stats, get_daily_return, get_baseline # backtest_plot
from common import *
import datetime as dt
from finrl.plot import *
import pandas as pd

ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)
candle_time_interval = '5Min'  # '1Min'

env = StockTradingEnv

ERL_PARAMS = {"learning_rate": 3e-6, "batch_size": 2048, "gamma": 0.985,
              "seed": 312, "net_dimension": 512, "target_step": 5000, "eval_gap": 30,
              "eval_times": 1}

train_start_date = '2022-6-11'
train_end_date = '2022-8-11'
test_start_date = '2022-8-11'
test_end_date = '2022-9-1'
baseline_ticker = 'AXP'

model_name = 'ppo'
MODEL_IDX = f'{model_name}_{train_start_date}_{train_end_date}'


# if you want to use larger datasets (change to longer period), and it raises error,
# please try to increase "target_step". It should be larger than the episode steps.


def backtest_plot(
        account_value,
        baseline_df,
        value_col_name="account_value",
):
    df = deepcopy(account_value)
    # print('date', len(df))
    # print(type(df["date"][0]), df["date"][0])
    df["date"] = pd.to_datetime(df["date"])
    # df["date"] = pd.Timestamp(df["date"]).tz_localize("America/New_York")
    # df["date"] = df["date"].tz_localize("America/New_York")
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )


def train_and_test(
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
        baseline_ticker,
        model_name,
        MODEL_IDX,
):
    train(start_date=train_start_date,
          end_date=train_end_date,
          ticker_list=ticker_list,
          data_source='alpaca',
          time_interval=candle_time_interval,
          technical_indicator_list=INDICATORS,
          drl_lib='elegantrl',
          #       drl_lib='rllib',
          #       drl_lib='stable_baselines3',
          env=env,
          model_name=model_name,
          API_KEY=API_KEY,
          API_SECRET=API_SECRET,
          API_BASE_URL=API_BASE_URL,
          erl_params=ERL_PARAMS,
          cwd=f'./papertrading_erl/{MODEL_IDX}',  # current_working_dir
          break_step=1e7)

    account_value = test(start_date=test_start_date,
                         end_date=test_end_date,
                         ticker_list=ticker_list,
                         data_source='alpaca',
                         time_interval=candle_time_interval,
                         technical_indicator_list=INDICATORS,
                         drl_lib='elegantrl',
                         env=env,
                         model_name='ppo',
                         API_KEY=API_KEY,
                         API_SECRET=API_SECRET,
                         API_BASE_URL=API_BASE_URL,
                         #       erl_params=ERL_PARAMS,
                         cwd=f'./papertrading_erl/{MODEL_IDX}',  # current_working_dir
                         if_plot=True,  # to return a dataframe for backtest_plot
                         break_step=1e7)
    print("============== account_value ===========")
    print(account_value)

    # baseline stats
    print("==============Get Baseline Stats===========")
    # baseline_df_dji = get_baseline(
    #     ticker="^DJI",
    #     start=test_start_date,
    #     end=test_end_date)

    baseline_df = load_df(test_start_date, test_end_date)
    baseline_df = baseline_df[baseline_df['tic'] == baseline_ticker]
    baseline_df = baseline_df.loc[baseline_df['timestamp'].dt.time == dt.time(15, 59)]
    baseline_df['date'] = baseline_df['timestamp'].dt.date
    stats = backtest_stats(baseline_df, value_col_name='close')
    print(stats)

    print("==============Compare to Baseline===========")
    backtest_plot(account_value, baseline_df)


if __name__ == '__main__':
    train_and_test(train_start_date, train_end_date, test_start_date, test_end_date, baseline_ticker, model_name,
                   MODEL_IDX, )
