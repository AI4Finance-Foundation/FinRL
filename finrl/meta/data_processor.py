"""
FinRL 金融数据处理器

这是FinRL框架的核心数据处理模块，提供了统一的接口来处理来自不同数据源的金融数据。
该模块采用适配器模式，支持多种金融数据提供商，并提供标准化的数据处理流程。

主要功能：
1. 数据获取：从各种金融数据源下载历史股票数据
2. 数据清洗：处理缺失值、异常值和数据格式标准化
3. 特征工程：计算技术指标、市场波动度和风险指标
4. 数据转换：将DataFrame格式转换为深度学习模型所需的数组格式

支持的数据源：
- Yahoo Finance：免费的全球股票数据
- Alpaca：美股实时和历史数据API
- WRDS（Wharton Research Data Services）：学术级金融数据库

金融术语说明：
- OHLCV：开盘价(Open)、最高价(High)、最低价(Low)、收盘价(Close)、成交量(Volume)
- Technical Indicators：技术指标，基于价格和成交量计算的数学指标
- VIX：波动率指数，衡量市场恐慌程度
- Turbulence：市场波动度，用于风险控制和异常检测

作者：AI4Finance Foundation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)

# 导入各种数据源处理器


class DataProcessor:
    """
    金融数据处理器统一接口类

    该类提供了一个统一的接口来处理来自不同数据源的金融数据。
    采用适配器模式，根据指定的数据源自动选择相应的处理器。

    设计优势：
    1. 统一接口：无论使用哪种数据源，API调用方式都相同
    2. 易于扩展：添加新数据源只需实现相应的processor
    3. 数据标准化：确保不同数据源的输出格式一致
    4. 错误处理：提供统一的错误处理和异常管理

    Attributes:
        processor: 具体的数据源处理器实例
        tech_indicator_list: 技术指标列表，用于缓存
        vix: VIX指数数据，用于缓存
    """

    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        """
        初始化数据处理器

        根据指定的数据源类型创建相应的处理器实例。每种数据源
        都有其特定的API要求和数据格式，这里统一进行初始化。

        Args:
            data_source (str): 数据源类型，支持:
                - "alpaca": Alpaca股票交易API
                - "wrds": Wharton研究数据服务
                - "yahoofinance": Yahoo Finance免费API
            tech_indicator (list, optional): 技术指标列表，用于缓存优化
            vix (pd.DataFrame, optional): VIX数据，用于缓存优化
            **kwargs: 额外参数，通常包含API密钥等认证信息

        Raises:
            ValueError: 当数据源不支持或认证信息错误时抛出
        """
        print(f"🔧 初始化 {data_source} 数据处理器...")

        if data_source == "alpaca":
            # ========== Alpaca 数据源初始化 ==========
            # Alpaca是一个现代化的股票交易平台，提供免佣金交易和实时数据API
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                API_BASE_URL = kwargs.get("API_BASE_URL")

                if not all([API_KEY, API_SECRET, API_BASE_URL]):
                    raise ValueError("Alpaca需要API_KEY、API_SECRET和API_BASE_URL")

                self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
                print("  ✅ Alpaca API连接成功")

            except Exception as e:
                raise ValueError(
                    f"❌ Alpaca连接失败: {str(e)}\n" "请检查您的API密钥是否正确"
                )

        elif data_source == "wrds":
            # ========== WRDS 数据源初始化 ==========
            # WRDS是沃顿商学院的研究数据服务，提供高质量的学术级金融数据
            print("  📚 连接WRDS学术数据库...")
            self.processor = Wrds()
            print("  ✅ WRDS连接成功")

        elif data_source == "yahoofinance":
            # ========== Yahoo Finance 数据源初始化 ==========
            # Yahoo Finance提供免费的全球股票数据，是个人投资者和研究者的首选
            print("  🌐 连接Yahoo Finance免费API...")
            self.processor = YahooFinance()
            print("  ✅ Yahoo Finance连接成功")

        else:
            # 不支持的数据源
            supported_sources = ["alpaca", "wrds", "yahoofinance"]
            raise ValueError(
                f"❌ 不支持的数据源: '{data_source}'\n"
                f"支持的数据源: {supported_sources}"
            )

        # 初始化缓存变量，用于优化重复操作
        # 这些变量在使用缓存时避免重复计算技术指标
        self.tech_indicator_list = tech_indicator
        self.vix = vix

        print(f"✅ {data_source} 数据处理器初始化完成")

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        """
        下载金融数据

        从指定数据源下载股票的历史价格数据。返回的数据包含OHLCV
        （开盘价、最高价、最低价、收盘价、成交量）等基础信息。

        Args:
            ticker_list (list): 股票代码列表，如['AAPL', 'MSFT', 'GOOGL']
            start_date (str): 开始日期，格式'YYYY-MM-DD'
            end_date (str): 结束日期，格式'YYYY-MM-DD'
            time_interval (str): 时间间隔，如'1D'(日线)、'1H'(小时线)

        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame，列包括:
                - date: 日期
                - tic: 股票代码
                - open: 开盘价
                - high: 最高价
                - low: 最低价
                - close: 收盘价
                - volume: 成交量

        金融概念解释：
        - Ticker Symbol: 股票代码，如AAPL代表苹果公司
        - OHLCV: 股票价格的标准格式，包含了一个时间段内的核心信息
        - Time Interval: 数据频率，日线用于长期分析，分钟线用于短期交易
        """
        print(f"📊 下载数据: {len(ticker_list)} 只股票，{start_date} 到 {end_date}")

        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )

        print(f"  ✅ 成功下载 {len(df)} 条数据记录")
        return df

    def clean_data(self, df) -> pd.DataFrame:
        """
        清洗金融数据

        处理原始数据中的质量问题，包括缺失值、异常值、重复数据等。
        数据清洗是金融分析的关键步骤，确保后续分析的准确性。

        主要清洗步骤：
        1. 移除缺失值或使用合理方法填充
        2. 检测和处理异常值（如价格为负数）
        3. 去除重复记录
        4. 标准化数据格式和时区
        5. 确保数据的时间序列连续性

        Args:
            df (pd.DataFrame): 原始下载的数据

        Returns:
            pd.DataFrame: 清洗后的数据

        金融数据质量问题：
        - 停牌期间的缺失数据
        - 分股、合股等公司行为导致的价格跳跃
        - 交易所节假日导致的数据空白
        - 数据提供商的传输错误
        """
        print("🧹 清洗数据，处理缺失值和异常值...")

        original_count = len(df)
        df = self.processor.clean_data(df)
        final_count = len(df)

        print(f"  📊 数据清洗完成: {original_count} -> {final_count} 条记录")
        if original_count != final_count:
            print(f"  🗑️ 移除了 {original_count - final_count} 条异常数据")

        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        """
        添加技术指标

        计算各种技术指标并添加到数据中。技术指标是量化分析的核心工具，
        帮助识别市场趋势、动量和买卖信号。

        常见技术指标类型：
        1. 趋势指标：识别价格趋势方向
           - SMA/EMA: 简单/指数移动平均线
           - MACD: 移动平均收敛发散指标

        2. 动量指标：衡量价格变动速度
           - RSI: 相对强弱指数
           - CCI: 商品通道指数

        3. 波动率指标：衡量价格波动程度
           - Bollinger Bands: 布林带
           - ATR: 平均真实波幅

        Args:
            df (pd.DataFrame): 清洗后的价格数据
            tech_indicator_list (list): 要计算的技术指标列表

        Returns:
            pd.DataFrame: 包含技术指标的扩展数据

        技术分析原理：
        技术分析基于三个基本假设：
        1. 市场价格反映一切信息
        2. 价格呈趋势运动
        3. 历史会重复
        """
        print(f"📈 计算技术指标: {tech_indicator_list}")

        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        print(f"  ✅ 成功添加 {len(tech_indicator_list)} 个技术指标")
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        """
        添加市场波动度指标

        计算市场波动度（Turbulence Index），这是一个重要的风险指标，
        用于检测市场异常状态和系统性风险。

        波动度计算原理：
        1. 计算股票收益率的协方差矩阵
        2. 衡量当前市场状态与历史平均状态的偏离程度
        3. 高波动度通常预示着市场不稳定或危机

        应用场景：
        - 风险管理：高波动度时减少持仓或停止交易
        - 策略调整：根据市场状态调整交易策略
        - 危机预警：检测潜在的市场崩盘风险

        Args:
            df (pd.DataFrame): 包含价格和技术指标的数据

        Returns:
            pd.DataFrame: 添加了turbulence列的数据

        注意：波动度指标对数据质量要求较高，需要足够的历史数据窗口
        """
        print("🌊 计算市场波动度指标...")

        df = self.processor.add_turbulence(df)

        print("  ✅ 市场波动度指标计算完成")
        return df

    def add_vix(self, df) -> pd.DataFrame:
        """
        添加VIX恐慌指数

        VIX（Volatility Index）是衡量市场恐慌程度的重要指标，
        被称为"恐慌指数"或"投资者恐惧指标"。

        VIX指标特点：
        - 数值范围：通常在10-80之间
        - 低值（<20）：市场平静，投资者信心强
        - 中值（20-30）：市场存在一定不确定性
        - 高值（>30）：市场恐慌，投资者极度悲观

        投资意义：
        1. 反向指标：VIX高时往往是买入机会
        2. 风险管理：高VIX时应降低仓位
        3. 情绪指标：衡量市场整体情绪状态

        Args:
            df (pd.DataFrame): 股票数据

        Returns:
            pd.DataFrame: 添加了VIX列的数据

        历史经验：
        - 2008年金融危机：VIX峰值达到80+
        - 2020年疫情恐慌：VIX峰值达到82.69
        - 平静市场：VIX通常低于15
        """
        print("😱 添加VIX恐慌指数...")

        df = self.processor.add_vix(df)

        print("  ✅ VIX恐慌指数添加完成")
        return df

    def add_vixor(self, df) -> pd.DataFrame:
        """
        添加VIX衍生指标

        VIXor可能是VIX的变种或衍生指标，用于更精细地衡量
        市场波动率或特定的风险因子。

        Args:
            df (pd.DataFrame): 股票数据

        Returns:
            pd.DataFrame: 添加了VIXor指标的数据
        """
        print("📊 添加VIX衍生指标...")

        df = self.processor.add_vixor(df)

        print("  ✅ VIX衍生指标添加完成")
        return df

    def df_to_array(self, df, if_vix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将DataFrame转换为深度学习模型所需的数组格式

        将pandas DataFrame格式的数据转换为numpy数组，这是深度学习
        模型训练所必需的格式。同时进行数据预处理和异常值处理。

        转换过程：
        1. 提取价格数据（OHLC）
        2. 提取技术指标数据
        3. 提取波动度/VIX数据
        4. 处理NaN和无穷大值
        5. 确保数据类型和形状正确

        Args:
            df (pd.DataFrame): 包含所有特征的完整数据
            if_vix (bool): 是否包含VIX指数

        Returns:
            tuple: 包含三个numpy数组的元组
                - price_array: 价格数据数组 (时间, 股票, 价格特征)
                - tech_array: 技术指标数组 (时间, 股票, 指标)
                - turbulence_array: 波动度数组 (时间,)

        数据形状说明：
        - price_array: (T, N, 4) T=时间步，N=股票数，4=OHLC
        - tech_array: (T, N, I) I=技术指标数量
        - turbulence_array: (T,) 全市场的波动度时间序列

        异常值处理：
        技术指标中的NaN和无穷大值会被替换为0，这是因为：
        1. 部分指标在初期计算窗口不足
        2. 某些极端市场条件下可能产生异常值
        3. 0值不会对模型训练产生负面影响
        """
        print("🔄 转换数据格式为深度学习数组...")

        # 调用具体处理器的转换方法
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )

        print(f"  📏 数组形状:")
        print(f"    - 价格数组: {price_array.shape}")
        print(f"    - 技术指标数组: {tech_array.shape}")
        print(f"    - 波动度数组: {turbulence_array.shape}")

        # ==================== 异常值处理 ====================
        # 处理技术指标中的NaN值
        # NaN通常出现在指标计算的初期，由于历史数据不足导致
        tech_nan_positions = np.isnan(tech_array)
        nan_count = np.sum(tech_nan_positions)
        if nan_count > 0:
            tech_array[tech_nan_positions] = 0
            print(f"    🔧 处理了 {nan_count} 个NaN值")

        # 处理技术指标中的无穷大值
        # 无穷大值可能由除零运算或极端市场条件产生
        tech_inf_positions = np.isinf(tech_array)
        inf_count = np.sum(tech_inf_positions)
        if inf_count > 0:
            tech_array[tech_inf_positions] = 0
            print(f"    🔧 处理了 {inf_count} 个无穷大值")

        print("  ✅ 数据转换完成，可用于深度学习模型训练")
        return price_array, tech_array, turbulence_array
