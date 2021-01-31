import unittest
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.env.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
import numpy as np
import pandas as pd


class TestStocktradingEnvCashpenalty(unittest.TestCase):
    def setUp(cls):
        cls.ticker_list = ["AAPL", "GOOG"]
        cls.df = YahooDownloader(
            start_date="2009-01-01", end_date="2021-01-01", ticker_list=cls.ticker_list
        ).fetch_data()
        print(f"df columns: {cls.df.columns}")
        cls.indicators = ["open", "close", "high", "low", "volume"]

    def test_trivial(self):
        print(f"got hree!")
        self.assertTrue(True)

    def test_zero_step(self):
        # Prove that zero actions results in zero stock buys, and no price changes
        init_amt = 1e6
        env = StockTradingEnvCashpenalty(
            df=self.df, initial_amount=init_amt, cache_indicator_data=False
        )
        _ = env.reset()

        # step with all zeros
        for i in range(2):

            actions = np.zeros(len(self.ticker_list))
            next_state, _, _, _ = env.step(actions)
            cash = next_state[0]
            holdings = next_state[1 : 1 + len(self.ticker_list)]
            asset_value = env.account_information["asset_value"][-1]
            total_assets = env.account_information["total_assets"][-1]
            self.assertEqual(cash, init_amt)
            self.assertEqual(0.0, np.sum(holdings))
            self.assertEqual(0.0, asset_value)
            self.assertEqual(init_amt, total_assets)
            self.assertEqual(i + 1, env.current_step)

    def test_cost_penalties(self):
        # TODO: Requesting contributions!
        pass

    def test_purchases(self):
        # TODO: Requesting contributions!
        pass

    def test_gains(self):
        # TODO: Requesting contributions!
        pass

    def validate_caching(self):
        # prove that results with or without caching don't change anything

        env_uncached = StockTradingEnvCashpenalty(
            df=self.df, initial_amount=init_amt, cache_indicator_data=False
        )
        env_cached = env = StockTradingEnvCashpenalty(
            df=self.df, initial_amount=init_amt, cache_indicator_data=True
        )
        _ = env_uncached.reset()
        _ = env_cached.reset()
        for i in range(10):
            actions = np.random.uniform(low=-1, high=1, size=2)
            print(f"actions: {actions}")
            un_state, un_reward, _, _ = env_uncached.step(actions)
            ca_state, ca_reward, _, _ = env_cached.step(actions)

            self.assertEqual(un_state, ca_state)
            self.assertEqual(un_reward, ca_reward)
