import unittest
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.env.accounting.ledger import Ledger
import numpy as np
import pandas as pd
import datetime


class TestLedger(unittest.TestCase):
    def test_trivial(self):
        self.assertTrue(True)

    def test_instantation_and_insert(self):
        n = 12

        l = Ledger(assets=["AAPL", "GOOG"], tax_threshold_days=7)
        datelist = [
            d.date()
            for d in pd.date_range(
                datetime.datetime.today() - datetime.timedelta(days=n), periods=n
            )
        ]
        for d in datelist[:10]:
            _ = l.log_date(d, [1.0, 1.0], [1.0, 1.0])
        """
        Since our tax horizon here is 7 days, we expect that the only long term shares
            are in the first 3 days of our 10. 
        So, selling 5 and 4 units respectively, we should get back 3 from long term for each sale here
        """
        tax_implications = l.log_date(str(datelist[10]), [-5, -4], [1.0,1.0])
        print(f"Tax implications: {tax_implications}")
        self.assertEqual(tax_implications[0]["long_term_shares"], 3)
        self.assertEqual(tax_implications[0]["short_term_shares"], 2)
        self.assertEqual(
            tax_implications[0]["short_term_shares"]
            + tax_implications[0]["long_term_shares"],
            5,
        )
        self.assertTrue(1 in tax_implications)
        """
        Since we have now exhausted our early purchases in terms of long term, we should get entirely short term shares here

        """
        trans2 = [-4, -4]
        tax_implications = l.log_date(str(datelist[11]), trans2, [1.0, 1.0])
        print(tax_implications)
        for i, v in enumerate(trans2):
            self.assertEqual(tax_implications[i]["short_term_shares"], -v)

        print(tax_implications)

    def test_count_holdings(self):
        n = 12

        l = Ledger(assets=["AAPL", "GOOG"], tax_threshold_days=7)
        datelist = [
            d.date()
            for d in pd.date_range(
                datetime.datetime.today() - datetime.timedelta(days=n), periods=n
            )
        ]
        for d in datelist[:10]:
            _ = l.log_date(d, [1.0, 1.0], [1.0,1.0])

        results = l.long_term_holdings
        self.assertTrue(len(results) == 2)
        self.assertEqual(results, [2.0, 2.0])
        print(results)

        # let's sell some stuff.
        # sell 2 units each. But we advance a day, so we get one additional unit
        # result shoudl be that after sale we have 1 long term tax unit and the rest short term.
        tax_imps = l.log_date(datelist[-2], [-2, -2], [1.0, 1.0])
        long_holdings = l.long_term_holdings
        self.assertEqual(long_holdings, [1.0, 1.0])
        print(f"final results: {results}")
        short_holdings  = l.short_term_holdings

        self.assertEqual(l.holdings,np.add(long_holdings, short_holdings).tolist() )


    def test_price_calculations(self):

        # need to test that price of old shares is computed correctly

        pass
