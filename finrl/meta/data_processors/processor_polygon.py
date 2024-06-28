class PolygonEngineer:
    def __init__(self):
        pass

    def data_fetch ( self, stock_list, num, unit, end_dt ):
        pass


    def preprocess ( df, stock_list ):
        pass



if __name__ == "__main__":
    import sys
    sys.path.append("..")


    TRADE_START_DATE = "2019-01-01"
    TRADE_END_DATE = "2021-08-01"


    e = PolygonEngineer()
    stock_list = ["AAPL"]
    num = 10
    unit = "day"
    end_dt = TRADE_END_DATE
    df = e.data_fetch(stock_list, num, unit, end_dt)