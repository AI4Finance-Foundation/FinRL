import pandas as pd
import numpy as np
from copy import copy
import datetime
from pprint import pprint


class Ledger:

    """
    This class allows for the buying and selling of assets,
    primarily now it will assist in determining tax implications of sales.

    In the future, it could be used to track environment information in an enhanced way.
    # at any date, we compute everything that did happen after the transactions, assuming that we close things on that day.
    This is a simplification as market orders could take time to fill, but for now this is good.

    self.d holds daily buys, sells,and tax implications of each asset used
    self.s holds scalars, and values like holdings of each asset class.
    """

    def __init__(self, assets, df=None, tax_threshold_days=365):
        self.assets = assets
        self.d = {a: {} for a in assets}
        self.s = {}
        self.tax_threshold_days = tax_threshold_days
        self.dates = []

    def date_from_str(self, datestr):
        return pd.to_datetime(datestr, infer_datetime_format=True).date()

    def date_to_str(self, date):
        return str(date)

    def coerce_date(self, date):
        """
        This function takes in a date object of many types and coerces it to a str representation
        """
        if isinstance(date, datetime.date):
            date = self.date_to_str(date)
        elif isinstance(date, datetime.datetime):
            date = self.date_to_str(date.date())
        elif isinstance(date, str):
            pass
        else:
            raise Exception("unknown date type")
        assert isinstance(date, str)
        return date

    def log_transactions(self, date, transactions, prices):
        """
        This function takes in transactions and prices, and computes tax implications of these transactions.
        It returns a dictionary of each asset and the tax implications of sale.
        It also records all of the transactions in self.d
        Args:
            date (str or datetime): Date to log for the transactions
            transactions (list(float)): Transactions corresponding to each asset. TODO: Allow dictionary actions
            prices (list(float)): Price per asset (in order of self.assets)
        returns:
            dict
                keys:
                    spend: Amount spent on share purchases
                    proceeds: Amount from selling shares
                    assets:
                        idx: Index of asset in self.assets
                            short profit: Profits that are 'short term'. Computed as price-avg cost (could be negative)
                            long porift: Profits that are 'long term'. Computed as price--avg cost (could be negative)
        """
        date = self.coerce_date(date)
        if date not in self.dates:
            self.dates.append(date)
            # carry over previous values and overwrite them.
            if len(self.dates) == 1:
                self.s[date] = {}
            else:
                self.s[date] = copy(self.s[self.dates[-2]])
            self.s[date]["transactions"] = transactions
            self.s[date]["prices"] = prices
        tax_d = {"assets": {}}
        # compute selling proceeds
        sells = -np.clip(transactions, -np.inf, 0)
        proceeds = np.dot(sells, prices)
        tax_d["proceeds"] = proceeds

        # compute buy costs
        buys = np.clip(transactions, 0, np.inf)
        spend = np.dot(buys, prices)
        tax_d["spend"] = spend

        for i, (a, t, p) in enumerate(zip(self.assets, transactions, prices)):
            buys, sells = max(0, t), -min(0, t)

            self.d[a][date] = {"buys": buys, "sells": sells, "price": p, "tax_used": 0}
            if sells > 0:  # consider order here
                lots = self.compute_tax_lots(a, sells, date)
                # compute profit/loss of this tax lot
                lots["short_profit"] = (prices[i] - lots["short_avg_price"]) * lots[
                    "short_term_shares"
                ]
                lots["long_profit"] = (prices[i] - lots["long_avg_price"]) * lots[
                    "long_term_shares"
                ]
                tax_d["assets"][i] = lots
        self._compute_holdings()
        return tax_d

    def _compute_holdings(self):
        """
        This function looks at assets held, and determines the total assets of each type held
        It also breaks these down by the nature of the asset (long term vs short term)

        """
        holdings = [0 for _ in self.assets]
        long_term_holdings, short_term_holdings = [0 for _ in self.assets], [
            0 for _ in self.assets
        ]
        for i, a in enumerate(self.assets):
            before_dates = [
                (
                    self.date_from_str(self.current_date)
                    - datetime.timedelta(self.tax_threshold_days)
                )
                > self.date_from_str(x)
                for x in self.dates
            ]
            for long_term, d in zip(
                before_dates, [self.date_to_str(x) for x in self.dates]
            ):
                holdings[i] += self.d[a][d]["buys"]
                holdings[i] -= self.d[a][d]["sells"]
                if long_term:
                    long_term_holdings[i] += self.d[a][d]["buys"]
                    long_term_holdings[i] -= self.d[a][d]["tax_used"]
                else:
                    short_term_holdings[i] += self.d[a][d]["buys"]
                    short_term_holdings[i] -= self.d[a][d]["tax_used"]

        assert min(holdings) >= 0
        self.s[self.current_date]["holdings"] = holdings
        self.s[self.current_date]["long_term_holdings"] = long_term_holdings
        self.s[self.current_date]["short_term_holdings"] = short_term_holdings

    def log_scalars(self, date, data: dict):
        """
        This function logs scalars for a given date.
        args:
            date (str or datetime): date to log for
            data (dict): Dictionary of keys and values to log into the scalar data structure.
        """
        date = self.coerce_date(date)
        if date != self.current_date:
            raise Exception("this shouldn't happen")
        for k, v in data.items():
            self.s[date][k] = v

    @property
    def long_term_holdings(self):
        # TODO: if we want long term holdings to be compute for next trading day as opposed to current one
        return self.s[self.current_date]["long_term_holdings"]

    @property
    def short_term_holdings(self):
        return self.s[self.current_date]["short_term_holdings"]

    @property
    def holdings(self):
        return self.s[self.current_date]["holdings"]

    @property
    def current_date(self):
        return self.dates[-1]

    @property
    def cash(self):
        if "cash" not in self.s[self.current_date]:
            return self.s[self.dates[-2]]["cash"]
        else:
            return self.s[self.current_date]["cash"]

    @property
    def asset_value(self):
        return self.s[self.current_date]["asset_value"]

    @property
    def total_value(self):
        return self.asset_value + self.cash

    def compute_tax_lots(self, asset, sells, sell_date):
        a_data = copy(self.d[asset])
        remaining_shares = sells
        long_shares = 0
        long_total_value = 0
        short_shares = 0
        short_total_value = 0
        # need to figure out average cost for long term and short term
        # ok, let's loop here and use up shares oldest to youngest
        dates = sorted(a_data)
        i = 0
        while remaining_shares > 0:

            date = dates[i]
            long_term = (
                self.date_from_str(sell_date)
                - datetime.timedelta(days=self.tax_threshold_days)
            ) > self.date_from_str(date)

            d = a_data[date]
            avail_shares = d["buys"] - d["tax_used"]
            if avail_shares > 0:
                # let's consume these
                if avail_shares < remaining_shares:
                    shares_consumed = avail_shares
                else:
                    shares_consumed = remaining_shares

                remaining_shares -= shares_consumed
                if long_term:
                    long_shares += shares_consumed
                    long_total_value += shares_consumed * d["price"]
                else:
                    short_shares += shares_consumed
                    short_total_value += shares_consumed * d["price"]

                a_data[date]["tax_used"] = shares_consumed
            i += 1

        self.d[asset] = a_data

        def get_avg_price(shares, value):
            if shares == 0:
                return 0
            else:
                return value / shares

        return {
            "asset": asset,
            "long_term_shares": long_shares,
            "long_avg_price": get_avg_price(long_shares, long_total_value),
            "short_term_shares": short_shares,
            "short_avg_price": get_avg_price(short_shares, short_total_value),
        }
