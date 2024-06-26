class PaperTradingFutu:
    def __init__(self, futu_api: FutuAPI):
        self.futu_api = futu_api
        self.account_id = None
        self.trading_market = None
        self.trading_env = None
        self.trading_account = None
        self.trading_account_id = None
        self.trading_order_id = None
        self.trading_order_status = None
        self.trading_order_status_message = None
        self.trading_order_status_code