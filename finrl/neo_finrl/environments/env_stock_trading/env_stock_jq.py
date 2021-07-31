import numpy as np

class StockTradingEnv():
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, ary, initial_account=1e6, max_stock=1e2, transaction_fee_percent=1e-3, if_train=True,
                 ):
        self.stock_dim = 5
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        N = ary.shape[0]
        x = int(N*0.8)# ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)
        
        # reset
        self.day = 0
        if if_train:
            self.ary = ary[:x] 
        else:
            self.ary = ary[x:]
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.day_npy = self.ary[self.day]
        self.prices = self.day_npy[[5*x+3 for x in range(self.stock_dim)]]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack

        self.total_asset = self.account + (self.prices[:self.stock_dim] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21
        self.gamma_return = 0.0

        '''env information'''
        self.env_name = 'Stock_jq-v1'
        self.state_dim = 1 + 6 * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_return = 1.25  # convergence 1.5
        self.max_step = self.ary.shape[0]

    def reset(self):
        self.account = self.initial_account
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.prices = self.day_npy[[5*x+3 for x in range(self.stock_dim)]]
        self.total_asset = self.account + (self.prices[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)
        return state

    def step(self, action):
        action = action * self.max_stock
        self.prices = self.day_npy[[5*x+3 for x in range(self.stock_dim)]]
        """buy or sell stock"""
        for index in range(self.stock_dim):
            stock_action = action[index]
            adj = self.prices[index]
            if stock_action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, stock_action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day_npy = self.ary[self.day]
        self.prices = self.day_npy[[5*x+3 for x in range(self.stock_dim)]]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)

        next_total_asset = self.account + (self.prices[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_account
        return state, reward, done, None
