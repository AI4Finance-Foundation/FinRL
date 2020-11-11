import pathlib

#import finrl

import pandas as pd

#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
TESTING_DATA_FILE = "test.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)


## time_fmt = '%Y-%m-%d'
START_DATE = "2009-01-01"
END_DATE = "2020-11-01"


SINGLE_TICKER =['AAPL']

# Dow 30 constituents at 2019/01/01
DOW_30_TICKER = ['AAPL','MSFT','JPM','V','RTX','PG','GS','NKE','DIS','AXP',
                  'HD','INTC','WMT','IBM','MRK','UNH','KO','CAT','TRV','JNJ',
                  'CVX','MCD','VZ','CSCO','XOM','BA','MMM','PFE','WBA','DD']

# Nasdaq 100 constituents at 2019/01/01
nas_100_ticker = [
'AMGN'.'AAPL','AMAT','INTC','PCAR','PAYX','MSFT','ADBE','CSCO',
'XLNX','QCOM','COST','SBUX','FISV','CTXS','INTU','AMZN','EBAY','BIIB','CHKP',
'GILD','NLOK','CMCSA','FAST','ADSK','CTSH','CELG','NVDA','GOOGL','ISRG','VRTX',
'HSIC','BIDU','ATVI','ADP','ROST','ORLY','TFCFA','CERN','BKNG','MYL','MU','DLTR',
'ALXN','SIRI','MNST','AVGO','TXN','MDLZ','FB','ADI','WDC','REGN','LBTYK','VRSK',
'NFLX','TSLA','CHTR','MAR','ILMN','GOOGL','LRCX','EA','LBTYK','TFCFA','AAL','WBA',
'KHC','BMRN','JD','SWKS','INCY','PYPL','MXIM','TMUS','EXPE','TCOM','ULTA','CSX',
'NTES','MCHP','CTAS','KLAC','HAS','JBHT','IDXX','WYNN','MELI','ALGN','CDNS','WDAY',
'SNPS','ASML','TTWO','PEP','NXPI','XEL','AMD','NTAP','VRSN','LULU','WLTW','UAL'
]                  

