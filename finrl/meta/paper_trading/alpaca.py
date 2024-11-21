from __future__ import annotations

import datetime
import os
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd
import torch

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.meta.paper_trading.common import AgentPPO


class PaperTradingAlpaca:
    def __init__(
        self,
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
        self.agent = agent  # Store agent type for later use

        if agent == "ppo":
            if drl_lib == "elegantrl":
                agent_class = AgentPPO
                agent = agent_class(net_dim, state_dim, action_dim)
                actor = agent.act
                try:
                    cwd = cwd + "/actor.pth"
                    print(f"| load actor from: {cwd}")
                    actor.load_state_dict(
                        torch.load(cwd, map_location=lambda storage, loc: storage)
                    )
                    self.act = actor
                    self.device = agent.device
                except BaseException:
                    print(os.getcwd())
                    raise ValueError("Fail to load agent!")
            elif drl_lib == "rllib":
                from ray.rllib.agents import ppo
                from ray.rllib.agents.ppo.ppo import PPOTrainer

                config = ppo.DEFAULT_CONFIG.copy()
                config["env"] = StockEnvEmpty
                config["log_level"] = "WARN"
                config["env_config"] = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                }
                trainer = PPOTrainer(env=StockEnvEmpty, config=config)
                try:
                    trainer.restore(cwd)
                    self.agent = trainer
                    print("Restoring from checkpoint path", cwd)
                except:
                    raise ValueError("Fail to load agent!")
            elif drl_lib == "stable_baselines3":
                from stable_baselines3 import PPO

                try:
                    self.model = PPO.load(cwd)
                    print("Successfully load model", cwd)
                except:
                    raise ValueError("Fail to load agent!")
            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet. Please check your input."
                )

        elif agent == "ddpg":
            if drl_lib == "stable_baselines3":
                from stable_baselines3 import DDPG

                try:
                    self.model = DDPG.load(cwd)
                    print("Successfully loaded DDPG model", cwd)
                except:
                    raise ValueError("Failed to load DDPG model!")
            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet for DDPG. Please check your input."
                )

        elif agent == "td3":
            if drl_lib == "stable_baselines3":
                from stable_baselines3 import TD3

                try:
                    # load TD3 model
                    self.model = TD3.load(cwd)
                    print("Successfully loaded TD3 model", cwd)
                except:
                    raise ValueError("Failed to load TD3 model!")
            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet for TD3. Please check your input."
                )

        elif agent == "a2c":
            if drl_lib == "stable_baselines3":
                from stable_baselines3 import A2C

                try:
                    self.model = A2C.load(cwd)
                    print("Successfully load A2C model", cwd)
                except:
                    raise ValueError("Fail to load A2C agent!")
            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet for A2C. Please check your input."
                )

        else:
            print(f"Agent '{agent}' is not recognized.")
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except:
            raise ValueError(
                "Fail to connect to Alpaca. Please check account info and internet connection."
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

    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)

        # Wait for market to open.
        print("Waiting for market to open...")
        self.awaitMarketOpen()
        print("Market opened.")
        while True:
            clock = self.alpaca.get_clock()
            closingTime = clock.next_close.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            self.timeToClose = closingTime - currTime

            if self.timeToClose < (60 * 2):
                print("Market closing soon. Closing positions.")
                threads = []
                positions = self.alpaca.list_positions()
                for position in positions:
                    orderSide = "sell" if position.side == "long" else "buy"
                    qty = abs(int(float(position.qty)))
                    respSO = []
                    tSubmitOrder = threading.Thread(
                        target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                    )
                    tSubmitOrder.start()
                    threads.append(tSubmitOrder)

                for x in threads:
                    x.join()

                print("Sleeping until market close (15 minutes).")
                time.sleep(60 * 15)

            else:
                self.trade()
                last_equity = float(self.alpaca.get_account().last_equity)
                print("Equity: ", last_equity)
                self.equities.append([time.time(), last_equity])
                time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while not isOpen:
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            print(str(timeToOpen) + " minutes til market open.")
            time.sleep(60)
            isOpen = self.alpaca.get_clock().is_open

    def trade(self):
        state = self.get_state()
        print(f"State at trading time: {state}")

        if self.drl_lib == "elegantrl":
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
            action = (action * self.max_stock).astype(int)
            print(f"Elegantrl action: {action}")

        elif self.drl_lib == "rllib":
            action = self.agent.compute_single_action(state)
            print(f"RLlib action: {action}")

        elif self.drl_lib == "stable_baselines3":
            action = self.model.predict(state)[0]
            print(f"Stable Baselines action: {action}")

        else:
            raise ValueError(
                "The DRL library input is NOT supported yet. Please check your input."
            )

        # Additional debug for cash and price
        print(f"Cash available: {self.cash}")
        print(f"Stock prices: {self.price}")

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            threads = []

            # Selling loop
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty = abs(int(sell_num_shares))
                print(f"Attempting to sell {qty} shares of {self.stockUniverse[index]}")
                print(
                    f"Stock holdings: {self.stocks[index]}, Sell action: {action[index]}"
                )

                # Check if qty is reasonable
                if qty > 0:
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
                else:
                    print(
                        f"Skipping sell for {self.stockUniverse[index]} due to zero quantity."
                    )

            for x in threads:  #  wait for all threads to complete
                x.join()

            threads = []

            # Buying loop
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

                print(f"Attempting to buy {qty} shares of {self.stockUniverse[index]}")
                print(
                    f"Available cash: {self.cash}, Price of {self.stockUniverse[index]}: {self.price[index]}, Buy action: {action[index]}"
                )

                # Check if qty is reasonable
                if qty > 0:
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
                else:
                    print(
                        f"Skipping buy for {self.stockUniverse[index]} due to zero quantity or insufficient funds."
                    )

            for x in threads:  #  wait for all threads to complete
                x.join()
        else:
            print("Turbulence detected. Skipping trades to avoid risk.")
        state = self.get_state()
        print(f"State at trading time: {state}")

        # Action prediction based on DRL library
        if self.drl_lib == "elegantrl":
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
            action = (action * self.max_stock).astype(int)
            print(f"Elegantrl action: {action}")

        elif self.drl_lib == "stable_baselines3":
            action = self.model.predict(state)[0]
            print(f"Stable Baselines action: {action}")

        # Decision to trade based on action
        min_action = 1  # Reduced threshold for testing
        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            threads = []
            for index in np.where(action < -min_action)[0]:  # sell
                qty = min(self.stocks[index], -action[index])
                print(f"Attempting to sell {qty} of {self.stockUniverse[index]}")
                resp = []
                self.submitOrder(qty, self.stockUniverse[index], "sell", resp)
                print(f"Sell order response: {resp}")

            for index in np.where(action > min_action)[0]:  # buy
                qty = min(self.cash // self.price[index], abs(action[index]))
                if qty > 0:
                    print(
                        f"Attempting to buy {qty} of {self.stockUniverse[index]} with cash: {self.cash}"
                    )
                    resp = []
                    self.submitOrder(qty, self.stockUniverse[index], "buy", resp)
                    print(f"Buy order response: {resp}")
                else:
                    print(
                        f"Insufficient cash to buy {self.stockUniverse[index]}, qty: {qty}, cash: {self.cash}"
                    )

    def get_state(self):
        alpaca = AlpacaProcessor(api=self.alpaca)
        price, tech, turbulence = alpaca.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval="1Min",
            tech_indicator_list=self.tech_indicator_list,
        )
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        tech = tech * 2**-7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        print("stocks: ", stocks, self.stockUniverse)
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
        # print(len(self.stockUniverse))
        return state

    def submitOrder(self, qty, stock, side, resp):
        print(f"Order Attempt - Stock: {stock}, Quantity: {qty}, Side: {side}")
        if qty > 0:
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                print(
                    "Market order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | completed."
                )
                resp.append(True)
                print(
                    f"Order for {qty} shares of {stock} ({side}) submitted successfully."
                )
            except:
                print(
                    "Order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | did not go through."
                )
                resp.append(False)
                print(f"Order for {stock} ({side}) failed: {e}")
        else:
            """
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            """
            resp.append(True)
            print(f"Skipped order for {stock} due to zero quantity.")

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh


class StockEnvEmpty(gym.Env):
    # Empty Env used for loading rllib agent
    def __init__(self, config):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        self.env_num = 1
        self.max_step = 10000
        self.env_name = "StockEnvEmpty"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = False
        self.target_return = 9999
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        return

    def step(self, actions):
        return
