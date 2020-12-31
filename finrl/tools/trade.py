from time import time


def get_buy_and_hold_sharpe(test):
    test["daily_return"] = test["close"].pct_change(1)
    sharpe = (252 ** 0.5) * test["daily_return"].mean() / test["daily_return"].std()
    annual_return = ((test["daily_return"].mean() + 1) ** 252 - 1) * 100
    print("annual return: ", annual_return)

    print("sharpe ratio: ", sharpe)
    # return sharpe
