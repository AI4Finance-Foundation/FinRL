import os
from argparse import ArgumentParser

from finrl import config

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    TECHNICAL_INDICATORS_LIST,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
)

from finrl.finrl_meta.data_processor import DataProcessor

# construct environment
from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    if options.mode == "train":
        from finrl import train

        env = StockTradingEnv

        # demo for elegantrl
        kwargs = {}  # in current finrl_meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=TECHNICAL_INDICATORS_LIST,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            erl_params=ERL_PARAMS,
            break_step=1e5,
            kwargs=kwargs,
        )


if __name__ == "__main__":
    main()
