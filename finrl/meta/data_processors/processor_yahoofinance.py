"""
FinRL Yahoo Finance 数据处理器

这是专门用于从Yahoo Finance API获取和处理股票数据的模块。
Yahoo Finance是全球最流行的免费金融数据源之一，提供实时和历史
股票价格、财务数据和市场统计信息。

数据处理功能：
1. 股票价格数据下载：支持多种时间间隔（1分钟到1月）
2. 数据清洗和标准化：处理缺失值、异常值和格式统一
3. 技术指标计算：集成stockstats库计算各种技术指标
4. 市场风险指标：计算VIX恐慌指数和市场波动度
5. 数据格式转换：转换为深度学习模型所需的数组格式

支持的数据类型：
- OHLCV数据：开盘价、最高价、最低价、收盘价、成交量
- 技术指标：MACD、RSI、布林带、移动平均线等
- 市场指标：VIX恐慌指数、市场波动度

金融数据特点：
- 时间序列性：数据按时间顺序排列，具有时间依赖性
- 多维性：同时包含价格、成交量、技术指标等多个维度
- 噪声性：市场数据包含大量噪声，需要适当的预处理
- 非平稳性：金融时间序列通常是非平稳的

参考来源：https://github.com/AI4Finance-LLC/FinRL
作者：AI4Finance Foundation
"""

from __future__ import annotations

import datetime
import time
from datetime import date
from datetime import timedelta
from sqlite3 import Timestamp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from stockstats import StockDataFrame as Sdf
from webdriver_manager.chrome import ChromeDriverManager

### 以下部分由 aymeric75 添加，用于网页爬虫功能


class YahooFinanceProcessor:
    """
    Yahoo Finance 数据处理器
    
    这个类提供了从Yahoo Finance API获取股票数据的完整解决方案。
    Yahoo Finance是最受欢迎的免费金融数据源，提供全球股票市场的
    实时和历史数据。
    
    主要特点：
    1. 免费使用：无需API密钥，开箱即用
    2. 数据丰富：支持全球主要交易所的股票数据
    3. 时间粒度灵活：从1分钟到1月的多种时间间隔
    4. 实时更新：提供准实时的市场数据
    5. 技术指标集成：内置多种技术分析指标
    
    数据质量说明：
    - 实时性：有15-20分钟延迟（免费版限制）
    - 准确性：数据质量高，适合研究和回测
    - 完整性：偶尔可能有缺失数据，需要清洗处理
    - 稳定性：作为免费服务，可能有访问限制
    
    使用场景：
    - 学术研究：免费获取历史数据进行学术分析
    - 策略回测：验证交易策略的历史表现
    - 模型训练：为机器学习模型提供训练数据
    - 实时监控：开发股票监控和分析工具
    """

    def __init__(self):
        """
        初始化Yahoo Finance处理器
        
        Yahoo Finance API是基于HTTP请求的，不需要认证，
        因此初始化过程很简单，主要是设置默认参数。
        """
        print("🌐 初始化Yahoo Finance数据处理器")
        print("  ✅ 无需API密钥，开箱即用")
        pass

    """
    数据下载方法说明
    
    参数说明：
    ----------
        start_date : str
            数据开始日期，格式：'YYYY-MM-DD'
        end_date : str
            数据结束日期，格式：'YYYY-MM-DD'
        ticker_list : list
            股票代码列表，如['AAPL', 'MSFT', 'GOOGL']
        time_interval : str
            时间间隔，支持：1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    
    示例：
    -------
    输入参数：
    ticker_list = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2021-12-31'
    time_interval = "1D"  # 日线数据

    输出数据格式：
        date        tic     open        high        low         close       volume
    0   2020-01-02  AAPL    74.059998   75.150002   73.797501   75.087502   135480400.0
    1   2020-01-02  MSFT    157.320007  158.139999  155.509995  156.529999   22834900.0
    2   2020-01-02  GOOGL   1347.010010 1347.010010 1337.000000 1339.390015   1715200.0
    ...
    
    数据列说明：
    - date: 交易日期
    - tic: 股票代码（ticker symbol）
    - open: 开盘价
    - high: 最高价
    - low: 最低价
    - close: 收盘价
    - volume: 成交量
    """

    ######## 以下代码由 aymeric75 添加 ###################

    def date_to_unix(self, date_str) -> int:
        """
        将日期字符串转换为Unix时间戳
        
        Unix时间戳是从1970年1月1日开始的秒数，在网络API中广泛使用。
        Yahoo Finance的某些API接口需要Unix时间戳格式的日期参数。
        
        Args:
            date_str (str): 日期字符串，格式为'YYYY-MM-DD'
        
        Returns:
            int: Unix时间戳（秒）
        
        示例：
            '2020-01-01' -> 1577836800
        """
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_stock_data(self, stock_name, period1, period2) -> pd.DataFrame:
        """
        使用网页爬虫获取单只股票的历史数据
        
        这个方法通过Selenium自动化浏览器来爬取Yahoo Finance网页上的
        股票历史数据。当API访问受限时，这种方法可以作为备选方案。
        
        技术实现：
        1. 使用Selenium启动无头Chrome浏览器
        2. 访问Yahoo Finance历史数据页面
        3. 处理可能的弹窗和广告
        4. 解析HTML表格数据
        5. 转换为pandas DataFrame格式
        
        Args:
            stock_name (str): 股票代码，如'AAPL'
            period1 (int): 开始时间的Unix时间戳
            period2 (int): 结束时间的Unix时间戳
        
        Returns:
            pd.DataFrame: 包含历史价格数据的DataFrame
        
        注意：
        - 网页爬虫可能不稳定，建议优先使用API方法
        - 需要安装Chrome浏览器和ChromeDriver
        - 爬虫速度较慢，不适合大量数据获取
        """
        print(f"  🕷️ 爬取{stock_name}的历史数据...")
        
        # 构建Yahoo Finance历史数据页面URL
        url = f"https://finance.yahoo.com/quote/{stock_name}/history/?period1={period1}&period2={period2}&filter=history"

        # Selenium WebDriver 设置
        options = Options()
        options.add_argument("--headless")  # 无头模式，提高性能
        options.add_argument("--disable-gpu")  # 禁用GPU，提高兼容性
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        try:
            # 访问URL
            driver.get(url)
            driver.maximize_window()
            time.sleep(5)  # 等待页面加载

            # 处理可能的Cookie同意弹窗
            try:
                RejectAll = driver.find_element(
                    By.XPATH, '//button[@class="btn secondary reject-all"]'
                )
                action = ActionChains(driver)
                action.click(on_element=RejectAll)
                action.perform()
                time.sleep(5)
                print("    ✅ 已处理Cookie弹窗")

            except Exception as e:
                print(f"    ℹ️ 未发现弹窗或处理失败: {e}")

            # 解析页面获取数据表格
            soup = BeautifulSoup(driver.page_source, "html.parser")
            table = soup.find("table")
            if not table:
                raise Exception("未找到数据表格")

            # 提取表头
            headers = [th.text.strip() for th in table.find_all("th")]
            headers[4] = "Close"        # 修正收盘价列名
            headers[5] = "Adj Close"    # 修正调整收盘价列名
            headers = ["date", "open", "high", "low", "close", "adjcp", "volume"]

            # 提取数据行
            rows = []
            for tr in table.find_all("tr")[1:]:  # 跳过表头行
                cells = [td.text.strip() for td in tr.find_all("td")]
                if len(cells) == len(headers):  # 只添加列数正确的行
                    rows.append(cells)

            # 创建DataFrame
            df = pd.DataFrame(rows, columns=headers)

            # 数据类型转换函数
            def safe_convert(value, dtype):
                """安全转换数据类型，处理格式化数字（如包含逗号的数字）"""
                try:
                    return dtype(value.replace(",", ""))
                except ValueError:
                    return value

            # 转换数值列的数据类型
            df["open"] = df["open"].apply(lambda x: safe_convert(x, float))
            df["high"] = df["high"].apply(lambda x: safe_convert(x, float))
            df["low"] = df["low"].apply(lambda x: safe_convert(x, float))
            df["close"] = df["close"].apply(lambda x: safe_convert(x, float))
            df["adjcp"] = df["adjcp"].apply(lambda x: safe_convert(x, float))
            df["volume"] = df["volume"].apply(lambda x: safe_convert(x, int))

            # 添加股票代码列
            df["tic"] = stock_name

            # 添加交易日序号列
            start_date = datetime.datetime.fromtimestamp(period1)
            df["date"] = pd.to_datetime(df["date"])
            df["day"] = (df["date"] - start_date).dt.days
            df = df[df["day"] >= 0]  # 排除开始日期之前的数据

            # 反转DataFrame行序（Yahoo返回的数据是倒序的）
            df = df.iloc[::-1].reset_index(drop=True)

            print(f"    ✅ 成功获取{len(df)}条{stock_name}数据记录")
            return df
            
        finally:
            # 确保浏览器被关闭
            driver.quit()

    def scrap_data(self, stock_names, start_date, end_date) -> pd.DataFrame:
        """
        批量爬取多只股票的历史数据
        
        这个方法对多只股票执行网页爬虫，获取它们的历史数据，
        然后合并成一个统一的DataFrame。
        
        Args:
            stock_names (list): 股票代码列表
            start_date (str): 开始日期，格式'YYYY-MM-DD'
            end_date (str): 结束日期，格式'YYYY-MM-DD'
        
        Returns:
            pd.DataFrame: 合并后的所有股票历史数据
        
        处理流程：
        1. 转换日期为Unix时间戳
        2. 逐只股票进行数据爬取
        3. 合并所有股票的数据
        4. 按日期和股票代码排序
        
        注意：
        - 爬虫过程可能较慢，请耐心等待
        - 部分股票可能爬取失败，会跳过并继续
        - 建议不要同时爬取过多股票，避免被网站限制
        """
        print(f"🕷️ 开始批量爬取{len(stock_names)}只股票数据...")
        
        # 转换日期格式
        period1 = self.date_to_unix(start_date)
        period2 = self.date_to_unix(end_date)

        all_dataframes = []
        total_stocks = len(stock_names)

        # 逐只处理股票
        for i, stock_name in enumerate(stock_names):
            try:
                print(
                    f"正在处理 {stock_name} ({i + 1}/{total_stocks})... "
                    f"进度: {(i + 1) / total_stocks * 100:.1f}%"
                )
                df = self.fetch_stock_data(stock_name, period1, period2)
                all_dataframes.append(df)
                
            except Exception as e:
                print(f"❌ 获取{stock_name}数据失败: {e}")

        # 合并所有数据
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df = combined_df.sort_values(by=["day", "tic"]).reset_index(drop=True)
            print(f"✅ 成功爬取并合并{len(combined_df)}条数据记录")
            return combined_df
        else:
            print("❌ 未能获取任何股票数据")
            return pd.DataFrame()

    ######## aymeric75 添加的代码结束 ###################

    def convert_interval(self, time_interval: str) -> str:
        """
        转换时间间隔格式
        
        将FinRL标准化的时间周期转换为Yahoo Finance API支持的格式。
        不同的数据源对时间间隔有不同的表示方法，这个函数确保兼容性。
        
        Args:
            time_interval (str): FinRL格式的时间间隔
        
        Returns:
            str: Yahoo Finance API格式的时间间隔
        
        支持的时间间隔：
        - 分钟级：1m, 2m, 5m, 15m, 30m, 60m, 90m
        - 小时级：1h
        - 日级：1d, 5d
        - 周级：1wk
        - 月级：1mo, 3mo
        
        使用说明：
        - 1m到30m：适用于短线交易和高频策略
        - 1h到1d：适用于日内交易策略
        - 1wk到1mo：适用于中长期投资策略
        """
        # Yahoo Finance支持的所有时间间隔
        yahoo_intervals = [
            "1m",    # 1分钟 - 超短线交易
            "2m",    # 2分钟
            "5m",    # 5分钟 - 短线交易常用
            "15m",   # 15分钟 - 日内交易常用
            "30m",   # 30分钟
            "60m",   # 60分钟 = 1小时
            "90m",   # 90分钟
            "1h",    # 1小时 - 日内策略
            "1d",    # 1天 - 最常用，适合中长期分析
            "5d",    # 5天
            "1wk",   # 1周 - 周线分析
            "1mo",   # 1月 - 月线分析
            "3mo",   # 3月 - 季度分析
        ]
        
        if time_interval in yahoo_intervals:
            return time_interval
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
        proxy: str | dict = None,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # Download and save the data in a pandas DataFrame
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        delta = timedelta(days=1)
        data_df = pd.DataFrame()
        for tic in ticker_list:
            current_tic_start_date = start_date
            while (
                current_tic_start_date <= end_date
            ):  # downloading daily to workaround yfinance only allowing  max 7 calendar (not trading) days of 1 min data per single download
                temp_df = yf.download(
                    tic,
                    start=current_tic_start_date,
                    end=current_tic_start_date + delta,
                    interval=self.time_interval,
                    proxy=proxy,
                )
                if temp_df.columns.nlevels != 1:
                    temp_df.columns = temp_df.columns.droplevel(1)

                temp_df["tic"] = tic
                data_df = pd.concat([data_df, temp_df])
                current_tic_start_date += delta

        data_df = data_df.reset_index().drop(columns=["Adj Close"])
        # convert the column names to match processor_alpaca.py as far as poss
        data_df.columns = [
            "timestamp",
            "close",
            "high",
            "low",
            "open",
            "volume",
            "tic",
        ]

        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        tic_list = np.unique(df.tic.values)
        NY = "America/New_York"

        trading_days = self.get_trading_days(start=self.start, end=self.end)
        # produce full timestamp index
        if self.time_interval == "1d":
            times = trading_days
        elif self.time_interval == "1m":
            times = []
            for day in trading_days:
                #                NY = "America/New_York"
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                for i in range(390):  # 390 minutes in trading day
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )

        # create a new dataframe with full timestamp series
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[
                df.tic == tic
            ]  # extract just the rows from downloaded data relating to this tic
            for i in range(tic_df.shape[0]):  # fill empty DataFrame using original data
                tmp_timestamp = tic_df.iloc[i]["timestamp"]
                if tmp_timestamp.tzinfo is None:
                    tmp_timestamp = tmp_timestamp.tz_localize(NY)
                else:
                    tmp_timestamp = tmp_timestamp.tz_convert(NY)
                tmp_df.loc[tmp_timestamp] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]
            # print("(9) tmp_df\n", tmp_df.to_string()) # print ALL dataframe to check for missing rows from download

            # if close on start date is NaN, fill data with first valid close
            # and set volume to 0.
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print("NaN data on start date, fill using first valid data.")
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_close = tmp_df.iloc[i]["close"]
                        tmp_df.iloc[0] = [
                            first_valid_close,
                            first_valid_close,
                            first_valid_close,
                            first_valid_close,
                            0.0,
                        ]
                        break

            # if the close price of the first row is still NaN (All the prices are NaN in this case)
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "Missing data for ticker: ",
                    tic,
                    " . The prices are all NaN. Fill with 0.",
                )
                tmp_df.iloc[0] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

            # fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
                    # print(tmp_df.iloc[i], " Filled NaN data with previous close and set volume to 0. ticker: ", tic)

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        #            print(("Data clean for ") + tic + (" is finished."))

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        #        print("Data clean all finished!")

        return new_df

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        print("cleaned_vix\n", cleaned_vix)
        vix = cleaned_vix[["timestamp", "close"]]
        print('cleaned_vix[["timestamp", "close"]\n', vix)
        vix = vix.rename(columns={"close": "VIXY"})
        print('vix.rename(columns={"close": "VIXY"}\n', vix)

        df = data.copy()
        print("df\n", df)
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start: str, end: str) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
        return trading_days

    # ****** NB: YAHOO FINANCE DATA MAY BE IN REAL-TIME OR DELAYED BY 15 MINUTES OR MORE, DEPENDING ON THE EXCHANGE ******
    def fetch_latest_data(
        self,
        ticker_list: list[str],
        time_interval: str,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(
            minutes=limit + 1
        )  # get the last rows up to limit

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )  # use start and end datetime to simulate the limit parameter
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(
            columns=["Adj Close"]
        )  # Alpaca data does not have 'Adj Close'

        data_df.columns = [  # convert to Alpaca column names lowercase
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb
