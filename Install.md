# install FinRL

## prepare for install
1. vpn is all you need if in china
2. python version >=3.7
3. pip remove zipline if your system installed zipline, zipline conflict with master branch RinRL
4. install git

## install step 
*  clone project 
```
$ git clone https://github.com/AI4Finance-LLC/FinRL-Library.git
```
*  install dependencies
```
$ cd FinRL-Library
$ pip install .
```
* test install
```
$ python FinRL_StockTrading_NeurIPS_2018.py
```
the following outputs appears, take it easy, install is successful
1. UserWarning: Module "zipline.assets" not found; multipliers will not be applied to position notionals.
  'Module "zipline.assets" not found; multipliers will not be applied'
  
if following outputs appears, please ensure your vpn enable for yahooo api fetch data
1. Failed download: 
 xxxx: No data found for this date range, symbol may be delisted

