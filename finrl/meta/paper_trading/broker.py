from abc import ABC, abstractmethod

class IBroker (ABC):
    @abstractmethod
    def submit_order(self, stock, qty, side, order_type, time_in_force):
        pass

    @abstractmethod
    def list_orders(self, status):
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        pass
    
    @abstractmethod
    def list_positions(self):
        pass

    @abstractmethod
    def get_account(self):
        pass

    @abstractmethod
    def get_clock(self):
        pass

    @abstractmethod
    def close_conn(self):
        pass

    @abstractmethod
    def fetch_latest_data(
        self,
        ticker_list,
        time_interval,
        tech_indicator_list,
    ):
        pass

    