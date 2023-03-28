import datetime as dt
import multiprocessing
import json
import os
from finrl.train import train
from finrl.test import test
from finrl.config_tickers import *
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, get_daily_return, get_baseline
from train_and_test import backtest_plot, get_baseline
from common import *


def train_process(**kwargs):
    from utils.init_logger import init_logger

    save_path = kwargs["cwd"]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = os.path.join(save_path, "..", "process.log")
    init_logger(log_path)

    #     for key, value in kwargs.items():
    train(
        start_date=kwargs["start_date"],
        end_date=kwargs["end_date"],
        ticker_list=kwargs["ticker_list"],
        data_source=kwargs["data_source"],
        time_interval=kwargs["time_interval"],
        technical_indicator_list=kwargs["technical_indicator_list"],
        drl_lib=kwargs["drl_lib"],
        env=kwargs["env"],
        model_name=kwargs["model_name"],
        API_KEY=kwargs["API_KEY"],
        API_SECRET=kwargs["API_SECRET"],
        API_BASE_URL=kwargs["API_BASE_URL"],
        erl_params=kwargs["erl_params"],
        cwd=kwargs["cwd"],  # current_working_dir
        wandb=kwargs["wandb"],
        break_step=kwargs["break_step"],
    )


def test_process(**kwargs):
    from utils.init_logger import init_logger

    save_path = kwargs["cwd"]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = os.path.join(save_path, "..", "process.log")
    init_logger(log_path)

    #     for key, value in kwargs.items():
    account_value = test(
        start_date=kwargs["start_date"],
        end_date=kwargs["end_date"],
        ticker_list=kwargs["ticker_list"],
        data_source=kwargs["data_source"],
        time_interval=kwargs["time_interval"],
        technical_indicator_list=kwargs["technical_indicator_list"],
        drl_lib=kwargs["drl_lib"],
        env=kwargs["env"],
        model_name=kwargs["model_name"],
        API_KEY=kwargs["API_KEY"],
        API_SECRET=kwargs["API_SECRET"],
        API_BASE_URL=kwargs["API_BASE_URL"],
        #       erl_params=kwargs['erl_params'],
        cwd=kwargs["cwd"],  # current_working_dir
        if_plot=True,  # to return a dataframe for backtest_plot
        break_step=kwargs["break_step"],
    )
    # print("============== account_value ===========")
    # print(account_value)

    # baseline stats
    baseline_df = get_baseline(
        ticker=Baseline, start=kwargs["start_date"], end=kwargs["end_date"]
    )

    stats = backtest_stats(baseline_df, value_col_name="close")

    # %matplotlib inline
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    figs = backtest_plot(account_value, baseline_df)

    image_path = os.path.join(
        save_path,
        f"backtest_{TestStartDate}_{TestEndDate}_{TestTradeInterval}min.png",
    )
    figs.savefig(os.path.join(image_path))
        


if __name__ == "__main__":
    import streamlit as st

    save_dir = f"./log/"

    st.header("FinRL GUI", anchor=None)
    st.subheader("Train", anchor=None)
    StockPool = st.selectbox(
        "Select a stock pool",
        ["DOW_30_TICKER", "CHINESE_STOCK_TICKER", "NAS_100_TICKER", "SP_500_TICKER"],
    )
    NumStock = st.slider("Select the number of stocks to trade", 1, 50, 30)
    Algo = st.selectbox("Select an algorithm", ["PPO", "TD3", "SAC"])
    AlgoLib = st.selectbox(
        "Select an library", ["elegantrl", "rllib", "stable_baselines3"]
    )
    TrainStartDate = st.date_input(
        "Select a start date for training", value=dt.date(2022, 6, 11)
    ).strftime(
        "%Y-%-m-%-d"
    )  # datetime.data() to string
    TrainEndDate = st.date_input(
        "Select an end date for training", value=dt.date(2022, 8, 11)
    ).strftime("%Y-%-m-%-d")
    TrainTradeInterval = st.radio(
        "Select a trade interval", ["5Min", "1Min", "15Min", "30Min", "60Min"], key=1
    )
    Log = st.checkbox("Track log", value=True)

    ticker_list = eval(StockPool)[: int(NumStock)]
    env = StockTradingEnv
    # get the current date and time
    now = dt.datetime.now()
    time = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"
    MODEL_IDX = f"{Algo.lower()}_{TrainStartDate}_{TrainEndDate}_{time}"
    save_path = save_dir + f"{MODEL_IDX}/"
    # os.makedirs(save_path, exist_ok=True)

    if st.button("Train"):
        log_dict = {}

        ErlParams = {
            "learning_rate": 3e-6,
            "batch_size": 2048,
            "gamma": 0.985,
            "seed": 312,
            "net_dimension": 512,
            "target_step": 5000,
            "eval_gap": 30,
            "eval_times": 1,
        }
        st.text("Training Hyperparameters:")
        Hyperparams = st.json(ErlParams)
        st.text(f"Confs and Models saved to: {save_path}")
        training_args = {
            "start_date": TrainStartDate,
            "end_date": TrainEndDate,
            "ticker_list": ticker_list,
            "data_source": "alpaca",
            "time_interval": TrainTradeInterval,
            "technical_indicator_list": INDICATORS,
            "drl_lib": AlgoLib,
            "env": env,
            "model_name": Algo.lower(),
            "API_KEY": API_KEY,
            "API_SECRET": API_SECRET,
            "API_BASE_URL": API_BASE_URL,
            "erl_params": ErlParams,
            "cwd": os.path.join(save_path, "process/"),  # current_working_dir
            "wandb": False,  # wand be cannot be run in a subprocess
            "break_step": 1e7,
        }

        process = multiprocessing.Process(target=train_process, kwargs=training_args)
        process.daemon = (
            True  # let subprocess run on its own without blocking the main process
        )
        process.start()
        st.text(f"Process ID for current run: {process.pid}")

        # logging above info
        log_dict["StockPool"] = StockPool
        log_dict["NumStock"] = NumStock
        log_dict["TickerList"] = training_args["ticker_list"]
        log_dict["DataSource"] = training_args["data_source"]
        log_dict["IndicatorList"] = training_args["technical_indicator_list"]
        log_dict["Algo"] = Algo
        log_dict["AlgoLib"] = AlgoLib
        log_dict["TrainStartDate"] = TrainStartDate
        log_dict["TrainEndDate"] = TrainEndDate
        log_dict["TrainTradeInterval"] = TrainTradeInterval
        log_dict["ErlParams"] = ErlParams
        log_dict["BreakStep"] = training_args["break_step"]
        # log_dict['TrainArgs'] = training_args  # serialization error for json on dict of dict
        log_dict["ProcessID"] = process.pid

        print(log_dict)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = open(os.path.join(save_path, "conf.json"), "w")
        json.dump(log_dict, save_file)
        save_file.close()

        # process.join()  # if join(), the main process will wait for the subprocess to finish before continuing

        # train(start_date=TrainStartDate,
        # end_date=TrainEndDate,
        # ticker_list=ticker_list,
        # data_source='alpaca',
        # time_interval=TrainTradeInterval,
        # technical_indicator_list=INDICATORS,
        # drl_lib=AlgoLib,
        # env=env,
        # model_name=Algo.lower(),
        # API_KEY=API_KEY,
        # API_SECRET=API_SECRET,
        # API_BASE_URL=API_BASE_URL,
        # erl_params=ERL_PARAMS,
        # cwd=save_path,  # current_working_dir
        # wandb=False,
        # break_step=1e7)

    st.subheader("BackTest", anchor=None)
    TestStartDate = st.date_input(
        "Select a start date for backtest", value=dt.date(2022, 6, 1)
    ).strftime("%Y-%-m-%-d")
    TestEndDate = st.date_input(
        "Select an end date for backtest", value=dt.date(2022, 9, 1)
    ).strftime("%Y-%-m-%-d")
    TestTradeInterval = st.radio(
        "Select a trade interval", ["5Min", "1Min", "15Min", "30Min", "60Min"], key=2
    )
    Baseline = st.selectbox(
        "Select a baseline",
        [
            "^DJI",
        ],
        key=3,
    )
    # print(TestStartDate, TestEndDate, TestTradeInterval)
    # st.image(img, width=1000)

    # get all current ckpts
    ckpt_list = []

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for filename in os.listdir(save_dir):
        # check if the item is a directory
        if os.path.isdir(os.path.join(save_dir, filename)):
            # print(os.path.join(save_path, filename))
            ckpt_list.append(filename)
    SelectedCKPT = st.selectbox(
        "Select checkpoint", ckpt_list, key=4
    )  # select one ckpt
    save_path = save_dir + SelectedCKPT

    if st.button("BackTest"):  # test
        test_log_dict = {}

        # logging above info
        test_log_dict["StockPool"] = StockPool
        test_log_dict["NumStock"] = NumStock
        #     test_log_dict['TickerList'] = ticker_list
        test_log_dict["DataSource"] = "alpaca"
        #     test_log_dict['IndicatorList'] = INDICATORS
        test_log_dict["TestStartDate"] = TestStartDate
        test_log_dict["TestEndDate"] = TestEndDate
        test_log_dict["TestTradeInterval"] = TestTradeInterval
        Hyperparams = st.json(test_log_dict)

        #     print(test_log_dict)

        account_value = test(
            start_date=TestStartDate,
            end_date=TestEndDate,
            ticker_list=ticker_list,
            data_source="alpaca",
            time_interval=TestTradeInterval,
            technical_indicator_list=INDICATORS,
            drl_lib=AlgoLib,
            env=env,
            model_name=Algo.lower(),
            API_KEY=API_KEY,
            API_SECRET=API_SECRET,
            API_BASE_URL=API_BASE_URL,
            #       erl_params=ERL_PARAMS,
            cwd=save_path,  # current_working_dir
            if_plot=True,  # to return a dataframe for backtest_plot
            break_step=1e7,
        )

        # baseline stats
        print("==============Get Baseline Stats===========")
        baseline_df = get_baseline(
            ticker=Baseline, start=TestStartDate, end=TestEndDate
        )

        stats = backtest_stats(baseline_df, value_col_name="close")

        print("==============Compare to DJIA===========")
        # %matplotlib inline
        # S&P 500: ^GSPC
        # Dow Jones Index: ^DJI
        # NASDAQ 100: ^NDX
        figs = backtest_plot(account_value, baseline_df)
        from PIL import Image

        image_path = os.path.join(
            save_path,
            f"backtest_{TestStartDate}_{TestEndDate}_{TestTradeInterval}.png",
        )
        figs.savefig(os.path.join(image_path))
        image = Image.open(image_path)
        st.subheader("BackTest Results", anchor=None)
        st.image(image, caption="Results")

    st.subheader("Compare", anchor=None)
    CompareStartDate = st.date_input(
        "Select a start date for compare", value=dt.date(2022, 6, 1)
    ).strftime("%Y-%-m-%-d")
    CompareEndDate = st.date_input(
        "Select an end date for compare", value=dt.date(2022, 9, 1)
    ).strftime("%Y-%-m-%-d")
    CompareTradeInterval = st.radio(
        "Select a trade interval", ["5Min", "1Min", "15Min", "30Min", "60Min"], key=5
    )
    Baseline = st.selectbox(
        "Select a baseline",
        [
            "^DJI",
        ],
        key=6,
    )
      
    print(os.getcwd())
    # get all current ckpts
    ckpt_list = []

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for filename in os.listdir(save_dir):
        # check if the item is a directory
        if os.path.isdir(os.path.join(save_dir, filename)):
            ckpt_list.append(filename)
    # select multiple checkpoints
    SelectedCKPTs = st.multiselect("Select checkpoints", ckpt_list)

    if st.button("Compare"):  # test
        processes = []
        for SelectedCKPT in SelectedCKPTs:
            
            save_path = save_dir + SelectedCKPT

            compare_args = {
                "start_date": CompareStartDate,
                "end_date": CompareEndDate,
                "ticker_list": ticker_list,
                "data_source": "alpaca",
                "time_interval": CompareTradeInterval,
                "technical_indicator_list": INDICATORS,
                "drl_lib": AlgoLib,
                "env": env,
                "model_name": Algo.lower(),
                "API_KEY": API_KEY,
                "API_SECRET": API_SECRET,
                "API_BASE_URL": API_BASE_URL,
                "cwd": save_path,  # current_working_dir
                "if_plot": True,  # to return a dataframe for backtest_plot
                "break_step": 1e7,
            }

            process = multiprocessing.Process(target=test_process, kwargs=compare_args)

            process.daemon = (
                True  # let subprocess run on its own without blocking the main process
            )
            process.start()
            processes.append(process.pid)
        st.text(f"Process IDs for current comparison: {processes}")

        # logging above info


        # display results
        st.subheader("Compare Results", anchor=None)
