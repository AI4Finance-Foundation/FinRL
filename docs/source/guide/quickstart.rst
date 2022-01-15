:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Quick Start
==================

Open ``main.py``

.. code-block:: python
   :linenos:

    import json
    import logging
    import os
    import time
    from argparse import ArgumentParser
    import datetime
    
    from finrl.config import config
    
    
    
    def build_parser():
        parser = ArgumentParser()
        parser.add_argument("--mode",dest="mode",
                            help="start mode, train, download_data"
                                 " backtest",
                            metavar="MODE", default="train")
        return parser
    
    
    def main():
        parser = build_parser()
        options = parser.parse_args()
        if not os.path.exists("./" + config.DATA_SAVE_DIR):
            os.makedirs("./" + config.DATA_SAVE_DIR)
        if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
            os.makedirs("./" + config.TRAINED_MODEL_DIR)
        if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
            os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
        if not os.path.exists("./" + config.RESULTS_DIR):
            os.makedirs("./" + config.RESULTS_DIR)
    
    
        if options.mode == "train":
            import finrl.autotrain.training
            finrl.autotrain.training.train_one()
    
        elif options.mode == "download_data":
            from finrl.marketdata.yahoodownloader import YahooDownloader
            df = YahooDownloader(start_date = config.START_DATE,
                                 end_date = config.END_DATE,
                                 ticker_list = config.DOW_30_TICKER).fetch_data()
            now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
            df.to_csv("./"+config.DATA_SAVE_DIR+"/"+now+'.csv')
    
    if __name__ == "__main__":
        main()


Run the library:

.. code-block:: python

    python main.py --mode=train

Choices for ``--mode``: start mode, train, download_data, backtest




