from __future__ import annotations

import numpy as np


class BitcoinEnv:  # custom env
    def __init__(
        self,
        data_cwd=None,
        price_ary=None,
        tech_ary=None,
        time_frequency=15,
        start=None,
        mid1=172197,
        mid2=216837,
        end=None,
        initial_account=1e6,
        max_stock=1e2,
        transaction_fee_percent=1e-3,
        mode="train",
        gamma=0.99,
    ):
        self.stock_dim = 1
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = 1
        self.gamma = gamma
        self.mode = mode
        self.load_data(
            data_cwd, price_ary, tech_ary, time_frequency, start, mid1, mid2, end
        )

        # reset
        self.day = 0
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        self.stocks = 0.0  # multi-stack

        self.total_asset = self.account + self.day_price[0] * self.stocks
        self.episode_return = 0.0
        self.gamma_return = 0.0

        """env information"""
        self.env_name = "BitcoinEnv4"
        self.state_dim = 1 + 1 + self.price_ary.shape[1] + self.tech_ary.shape[1]
        self.action_dim = 1
        self.if_discrete = False
        self.target_return = 10
        self.max_step = self.price_ary.shape[0]

    def reset(self) -> np.ndarray:
        self.day = 0
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        self.initial_account__reset = self.initial_account  # reset()
        self.account = self.initial_account__reset
        self.stocks = 0.0
        self.total_asset = self.account + self.day_price[0] * self.stocks

        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]
        state = np.hstack(
            (
                self.account * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)
        return state

    def step(self, action) -> (np.ndarray, float, bool, None):
        stock_action = action[0]
        """buy or sell stock"""
        adj = self.day_price[0]
        if stock_action < 0:
            stock_action = max(
                0, min(-1 * stock_action, 0.5 * self.total_asset / adj + self.stocks)
            )
            self.account += adj * stock_action * (1 - self.transaction_fee_percent)
            self.stocks -= stock_action
        elif stock_action > 0:
            max_amount = self.account / adj
            stock_action = min(stock_action, max_amount)
            self.account -= adj * stock_action * (1 + self.transaction_fee_percent)
            self.stocks += stock_action

        """update day"""
        self.day += 1
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        done = (self.day + 1) == self.max_step
        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]
        state = np.hstack(
            (
                self.account * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)

        next_total_asset = self.account + self.day_price[0] * self.stocks
        reward = (next_total_asset - self.total_asset) * 2**-16
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * self.gamma + reward
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0
            self.episode_return = next_total_asset / self.initial_account
        return state, reward, done, None

    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()
        episode_returns.append(1)
        btc_returns = list()  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(self.max_step):
                if i == 0:
                    init_price = self.day_price[0]
                btc_returns.append(self.day_price[0] / init_price)
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                episode_returns.append(self.total_asset / 1e6)
                if done:
                    break

        import matplotlib.pyplot as plt

        plt.plot(episode_returns, label="agent return")
        plt.plot(btc_returns, color="yellow", label="BTC return")
        plt.grid()
        plt.title("cumulative return")
        plt.xlabel("day")
        plt.xlabel("multiple of initial_account")
        plt.legend()
        plt.savefig(f"{cwd}/cumulative_return.jpg")
        return episode_returns, btc_returns

    def load_data(
        self, data_cwd, price_ary, tech_ary, time_frequency, start, mid1, mid2, end
    ):
        if data_cwd is not None:
            try:
                price_ary = np.load(f"{data_cwd}/price_ary.npy")
                tech_ary = np.load(f"{data_cwd}/tech_ary.npy")
            except BaseException:
                raise ValueError("Data files not found!")
        else:
            price_ary = price_ary
            tech_ary = tech_ary

        n = price_ary.shape[0]
        if self.mode == "train":
            self.price_ary = price_ary[start:mid1]
            self.tech_ary = tech_ary[start:mid1]
            n = self.price_ary.shape[0]
            x = n // int(time_frequency)
            ind = [int(time_frequency) * i for i in range(x)]
            self.price_ary = self.price_ary[ind]
            self.tech_ary = self.tech_ary[ind]
        elif self.mode == "test":
            self.price_ary = price_ary[mid1:mid2]
            self.tech_ary = tech_ary[mid1:mid2]
            n = self.price_ary.shape[0]
            x = n // int(time_frequency)
            ind = [int(time_frequency) * i for i in range(x)]
            self.price_ary = self.price_ary[ind]
            self.tech_ary = self.tech_ary[ind]
        elif self.mode == "trade":
            self.price_ary = price_ary[mid2:end]
            self.tech_ary = tech_ary[mid2:end]
            n = self.price_ary.shape[0]
            x = n // int(time_frequency)
            ind = [int(time_frequency) * i for i in range(x)]
            self.price_ary = self.price_ary[ind]
            self.tech_ary = self.tech_ary[ind]
        else:
            raise ValueError("Invalid Mode!")
