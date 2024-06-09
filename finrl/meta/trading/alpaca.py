import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
# import elegantrl.env as env
# import elegantrl.agent as agent
# import elegantrl.train as train
import numpy as np

import datetime
import gym
import threading
import time
import logbook
import pandas as pd 
import torch



from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.meta.paper_trading.common import AgentPPO

class AlpacaTrader:
    """
    A class representing an Alpaca trader.

    Parameters:
    - ticker_list (list): A list of ticker symbols for the stocks to trade.
    - time_interval (str): The time interval for trading data (e.g., "1Min", "5Min", "1Hour").
    - drl_lib (str): The deep reinforcement learning library to use (e.g., "elegantrl", "stable_baselines3").
    - agent: The agent to use for trading.
    - cwd (str): The current working directory.
    - net_dim (int): The dimension of the neural network.
    - state_dim (int): The dimension of the state.
    - action_dim (int): The dimension of the action.
    - API_KEY (str): The Alpaca API key.
    - API_SECRET (str): The Alpaca API secret.
    - API_BASE_URL (str): The Alpaca API base URL.
    - tech_indicator_list (list): A list of technical indicators to use.
    - turbulence_thresh (float): The turbulence threshold.
    - max_stock (float): The maximum number of stocks to hold.
    - latency (float): The latency for trading execution.

    Attributes:
    - drl_lib (str): The deep reinforcement learning library used.
    - logger (Logger): The logger for logging messages.
    - act (function): The function for taking actions.
    - device: The device used for training the agent.
    - alpaca: The Alpaca trading API.
    - time_interval (int): The time interval for trading data in seconds.
    - tech_indicator_list (list): A list of technical indicators used.
    - turbulence_thresh (float): The turbulence threshold.
    - max_stock (float): The maximum number of stocks to hold.
    - stocks (ndarray): An array representing the stocks currently held.
    - stocks_cd (ndarray): An array representing the stocks' cooldown.
    - cash: The amount of cash available.
    - stocks_df (DataFrame): A DataFrame representing the stocks' information.
    - asset_list (list): A list of assets.
    - price (ndarray): An array representing the stock prices.
    - stockUniverse (list): A list of stocks in the universe.
    - turbulence_bool (int): A flag indicating if turbulence is present.
    - equities (list): A list of equities.
    """

    def __init__(self,
                 ticker_list,
                 time_interval,
                 drl_lib,
                 agent,
                 cwd,
                 net_dim,
                 state_dim,
                 action_dim,
                 API_KEY,
                 API_SECRET,
                 API_BASE_URL,
                 tech_indicator_list,
                 turbulence_thresh=30,
                 max_stock=1e2,
                 latency=None,
            ):
        # Code for initializing the AlpacaTrader class

    def train(self, max_steps=1000000, batch_size=64, gamma=0.99, repeat_times=2, learning_rate=3e-4):
        """
        Train the ElegantRL agent on the Alpaca environment.

        Parameters:
        - max_steps (int): The maximum number of training steps.
        - batch_size (int): The batch size for training.
        - gamma (float): The discount factor.
        - repeat_times (int): The number of times to repeat the training.
        - learning_rate (float): The learning rate for training.
        """

        # Code for training the agent

    def is_market_open(self):
        """
        Check if the market is currently open.

        Returns:
        - bool: True if the market is open, False otherwise.
        """

        # Code for checking if the market is open

    def trade(self):
        """
        Execute a trade on the Alpaca platform.
        """

        # Code for executing a trade

    def get_state(self):
        """
        Get the current state for trading.

        Returns:
        - ndarray: The current state for trading.
        """

        # Code for getting the current state
class AlpacaTrader:
    """
    A class representing an Alpaca trader.

    Parameters:
    - ticker_list (list): A list of ticker symbols for the stocks to trade.
    - time_interval (str): The time interval for trading data (e.g., "1Min", "5Min", "1Hour").
    - drl_lib (str): The deep reinforcement learning library to use (e.g., "elegantrl", "stable_baselines3").
    - agent: The agent to use for trading.
    - cwd (str): The current working directory.
    - net_dim (int): The dimension of the neural network.
    - state_dim (int): The dimension of the state.
    - action_dim (int): The dimension of the action.
    - API_KEY (str): The Alpaca API key.
    - API_SECRET (str): The Alpaca API secret.
    - API_BASE_URL (str): The Alpaca API base URL.
    - tech_indicator_list (list): A list of technical indicators to use.
    - turbulence_thresh (float): The turbulence threshold.
    - max_stock (float): The maximum number of stocks to hold.
    - latency (float): The latency for trading execution.

    Attributes:
    - drl_lib (str): The deep reinforcement learning library used.
    - logger (Logger): The logger for logging messages.
    - act (function): The function for taking actions.
    - device: The device used for training the agent.
    - alpaca: The Alpaca trading API.
    - time_interval (int): The time interval for trading data in seconds.
    - tech_indicator_list (list): A list of technical indicators used.
    - turbulence_thresh (float): The turbulence threshold.
    - max_stock (float): The maximum number of stocks to hold.
    - stocks (ndarray): An array representing the stocks currently held.
    - stocks_cd (ndarray): An array representing the stocks' cooldown.
    - cash: The amount of cash available.
    - stocks_df (DataFrame): A DataFrame representing the stocks' information.
    - asset_list (list): A list of assets.
    - price (ndarray): An array representing the stock prices.
    - stockUniverse (list): A list of stocks in the universe.
    - turbulence_bool (int): A flag indicating if turbulence is present.
    - equities (list): A list of equities.
    """
    def __init__(self,
                 ticker_list,
                 time_interval,
                 drl_lib,
                agent,
                cwd,
                net_dim,
                state_dim,
                action_dim,
                 API_KEY,
                API_SECRET,
                API_BASE_URL,
                tech_indicator_list,
                turbulence_thresh=30,
                max_stock=1e2,
                latency=None,
            ):
        
        # load agent
        self.drl_lib = drl_lib
        self.logger = logbook.Logger(self.__class__.__name__)

        if agent == "ppo":
            if drl_lib == "elegantrl":
                agent_class = AgentPPO
                agent = agent_class(net_dim, state_dim, action_dim)
                actor = agent.act
                # load agent
                try:
                    cwd = cwd + "/actor.pth"
                    self.logger.info(f"load actor from: {cwd}")
                    actor.load_state_dict(
                        torch.load(cwd, map_location=lambda storage, loc: storage)
                    )
                    self.logger.info(f"Successfully load model {cwd}")
                    self.act = actor
                    self.device = agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")
            elif drl_lib == "stable_baselines3":
                from stable_baselines3 import PPO

                try:
                    # load agent
                    self.model = PPO.load(cwd)
                    self.logger.info(f"Successfully load model {cwd}")
                except:
                    raise ValueError("Fail to load agent!")
        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )
        
        # read trading time interval
        if time_interval == "1s":
            self.time_interval = 1
        elif time_interval == "5s":
            self.time_interval = 5
        elif time_interval == "1Min":
            self.time_interval = 60
        elif time_interval == "5Min":
            self.time_interval = 60 * 5
        elif time_interval == "15Min":
            self.time_interval = 60 * 15
        elif time_interval == "1Hour":
            self.time_interval = 60 * 1
        elif time_interval == "3Hour":
            self.time_interval = 60 * 3
        elif time_interval == "6Hour":
            self.time_interval = 60 * 6
        else:
            raise ValueError("Time interval input is NOT supported yet.")
        

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock


        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []


    
        
    def train(self, max_steps=1000000, batch_size=64, gamma=0.99, repeat_times=2, learning_rate=3e-4):
        """
        Train the ElegantRL agent on the Alpaca environment.

        Parameters:
        - max_steps (int): The maximum number of training steps.
        - batch_size (int): The batch size for training.
        - gamma (float): The discount factor.
        - repeat_times (int): The number of times to repeat the training.
        - learning_rate (float): The learning rate for training.
        """
        self.trainer.train(max_steps=max_steps, batch_size=batch_size, gamma=gamma, repeat_times=repeat_times, learning_rate=learning_rate)
        
    def is_market_open(self):
        """
        Check if the market is currently open.

        Returns:
        - bool: True if the market is open, False otherwise.
        """
        clock = self.api.get_clock()
        return clock.is_open
    
    def trade(self):
        """
        Execute a trade on the Alpaca platform.
        """
        state = self.get_state()

        if self.drl_lib == "elegantrl":
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
            action = (action * self.max_stock).astype(int)

        elif self.drl_lib == "rllib":
            action = self.agent.compute_single_action(state)

        elif self.drl_lib == "stable_baselines3":
            action = self.model.predict(state)[0]

        else:
            raise ValueError(
                "The DRL library input is NOT supported yet. Please check your input."
            )
        
        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 0  # stock_cd
            threads = []
            # self.logger.info ( f"stocks: {self.stocks}, univ: {self.stockUniverse}")
            self.logger.info ( f"action: {action}, min_action: {min_action}, ")
            for index in np.where(action < -min_action)[0]:  # sell_index:
                
                sell_num_shares = min(self.stocks[index], -action[index])
                # self.logger.info( f"holding: {self.stocks[index]}, action: {action[index]}, sell_num_shares: {sell_num_shares}")
                qty = abs(int(sell_num_shares))
                respSO = []
                
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "sell", respSO
                    )
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

            threads = []
            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(
                    tmp_cash // self.price[index], abs(int(action[index]))
                )
                if buy_num_shares != buy_num_shares:  # if buy_num_change = nan
                    qty = 0  # set to 0 quantity
                else:
                    qty = abs(int(buy_num_shares))
                respSO = []
                
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "buy", respSO
                    )
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

        else:  # sell all when turbulence
            threads = []
            positions = self.alpaca.list_positions()
            for position in positions:
                if position.side == "long":
                    orderSide = "sell"
                else:
                    orderSide = "buy"
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later

            for x in threads:  #  wait for all threads to complete
                x.join()

            self.stocks_cd[:] = 0
        
    def get_state(self):
        """
        Retrieves the current state of the trading environment.

        Returns:
            state (np.ndarray): The current state of the trading environment, represented as a numpy array.
        """
        alpaca = AlpacaProcessor(api=self.alpaca)
        price, tech, turbulence = alpaca.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval="1Min",  # "1Min", "5Min", "15Min", "1H", "1D
            tech_indicator_list=self.tech_indicator_list,
        )
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        tech = tech * 2**-7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool
        self.price = price

        amount = np.array(self.cash * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        state = np.hstack(
            (
                amount,
                turbulence,
                self.turbulence_bool,
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech,
            )
        ).astype(np.float32)
        state[np.isnan(state)] = 0.0
        state[np.isinf(state)] = 0.0
        # self.logger.info(len(self.stockUniverse))
        return state
        
    def submitOrder(self, qty, stock, side, resp):
        """
        Submits a market order for a given stock.

        Args:
            qty (int): The quantity of shares to be ordered.
            stock (str): The stock symbol to be ordered.
            side (str): The side of the order, either 'buy' or 'sell'.
            resp (list): A list to store the response of the order submission.

        Returns:
            None

        Raises:
            None
        """
        try:
            self.alpaca.submit_order(stock, qty, side, "market", "day")
            self.logger.info(f" Market order of | {qty} {stock} {side} | completed.")
            resp.append(True)
        except:
            self.logger.error(f"Order of | {qty} {stock} {side} | did not go through.")
            resp.append(False)

    def awaitMarketOpen(self):
        """
        Waits until the market is open.

        This method checks the market status using the Alpaca API and waits until the market is open.
        It continuously checks the market status every minute until it is open.

        Returns:
            None
        """
        isOpen = self.alpaca.get_clock().is_open
        while not isOpen:
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            self.logger.info(f"{timeToOpen} minutes til market open.")
            time.sleep(60)
            isOpen = self.alpaca.get_clock().is_open
    
    def run(self):
        """
        Run the trading loop, executing trades at the specified time interval.

        This method waits for the market to open, and then enters a loop where it repeatedly calls the `trade` method
        and sleeps for the specified time interval. The loop continues indefinitely until the program is terminated.
        """
        self.logger.info("Waiting for market to open...")
        self.awaitMarketOpen()
        self.logger.info("Market opened.")
        while True:
            self.trade()
            time.sleep(self.time_interval)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh