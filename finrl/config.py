"""
FinRL 框架配置文件

包含了深度强化学习金融交易系统的所有核心配置参数，包括：
- 目录结构配置
- 训练/测试/交易时间窗口设置
- 技术指标参数
- 各种DRL算法的超参数
- 数据源API配置
"""

from __future__ import annotations

# ==================== 目录配置 ====================
# 定义数据存储和模型保存的目录结构

DATA_SAVE_DIR = "datasets"  # 存储下载的金融数据（股价、技术指标等）
TRAINED_MODEL_DIR = "trained_models"  # 存储训练好的DRL模型文件
TENSORBOARD_LOG_DIR = "tensorboard_log"  # TensorBoard日志目录，用于可视化训练过程
RESULTS_DIR = "results"  # 存储回测结果和性能分析报告

# ==================== 时间窗口配置 ====================
# 金融交易中的关键概念：将数据分为训练期、测试期和实际交易期
# 这种分离避免了"前瞻偏差"（look-ahead bias），确保模型的可靠性

# 日期格式: '%Y-%m-%d' (年-月-日)

# 训练期（Training Period）：用于训练DRL智能体的历史数据时间段
TRAIN_START_DATE = "2014-01-06"  # 训练开始日期（修正：设置为周一，避免维度不匹配错误）
TRAIN_END_DATE = "2020-07-31"  # 训练结束日期

# 测试期（Testing/Validation Period）：用于验证模型泛化能力的数据时间段
TEST_START_DATE = "2020-08-01"  # 测试开始日期
TEST_END_DATE = "2021-10-01"  # 测试结束日期

# 交易期（Trading Period）：用于实际部署模型进行交易的时间段
TRADE_START_DATE = "2021-11-01"  # 实际交易开始日期
TRADE_END_DATE = "2021-12-01"  # 实际交易结束日期

# ==================== 技术指标配置 ====================
# 技术指标（Technical Indicators）是量化分析中的重要工具
# 基于历史价格和成交量数据计算，帮助识别趋势和交易信号
# 使用 stockstats 库，详见: https://pypi.org/project/stockstats/

INDICATORS = [
    # 趋势指标
    "macd",  # 移动平均收敛发散指标（Moving Average Convergence Divergence）
    # 用于识别趋势变化和买卖信号
    # 波动率指标
    "boll_ub",  # 布林带上轨（Bollinger Band Upper）
    # 表示价格的上阻力位，价格触及时可能下跌
    "boll_lb",  # 布林带下轨（Bollinger Band Lower）
    # 表示价格的下支撑位，价格触及时可能反弹
    # 动量指标
    "rsi_30",  # 30日相对强弱指数（Relative Strength Index）
    # 衡量价格变动速度，>70超买，<30超卖
    "cci_30",  # 30日商品通道指数（Commodity Channel Index）
    # 衡量价格偏离统计平均值的程度
    "dx_30",  # 30日方向性指数（Directional Index）
    # 衡量趋势强度的指标
    # 移动平均线（Moving Averages）
    "close_30_sma",  # 30日简单移动平均线（Simple Moving Average）
    # 平滑价格波动，识别中期趋势
    "close_60_sma",  # 60日简单移动平均线
    # 识别长期趋势，常用于确认趋势方向
]

# ==================== DRL算法参数配置 ====================
# 各种深度强化学习算法的超参数设置
# 这些参数直接影响算法的收敛速度和最终性能

# A2C (Advantage Actor-Critic) - 优势演员-评论家算法
A2C_PARAMS = {
    "n_steps": 5,  # 每次更新使用的步数
    "ent_coef": 0.01,  # 熵系数，控制探索程度
    "learning_rate": 0.0007,  # 学习率，控制参数更新幅度
}

# PPO (Proximal Policy Optimization) - 近端策略优化算法
PPO_PARAMS = {
    "n_steps": 2048,  # 收集经验的步数
    "ent_coef": 0.01,  # 熵系数，鼓励探索
    "learning_rate": 0.00025,  # 学习率
    "batch_size": 64,  # 批处理大小
}

# DDPG (Deep Deterministic Policy Gradient) - 深度确定性策略梯度算法
DDPG_PARAMS = {
    "batch_size": 128,  # 训练批次大小
    "buffer_size": 50000,  # 经验回放缓冲区大小
    "learning_rate": 0.001,  # 学习率
}

# TD3 (Twin Delayed Deep Deterministic Policy Gradient) - 双延迟DDPG算法
TD3_PARAMS = {
    "batch_size": 100,  # 训练批次大小
    "buffer_size": 1000000,  # 大容量经验缓冲区
    "learning_rate": 0.001,  # 学习率
}

# SAC (Soft Actor-Critic) - 软演员-评论家算法
SAC_PARAMS = {
    "batch_size": 64,  # 训练批次大小
    "buffer_size": 100000,  # 经验缓冲区大小
    "learning_rate": 0.0001,  # 学习率
    "learning_starts": 100,  # 开始学习前的预热步数
    "ent_coef": "auto_0.1",  # 自动调节的熵系数
}

# ERL (ElegantRL) - AI4Finance自研的DRL算法库参数
ERL_PARAMS = {
    "learning_rate": 3e-5,  # 学习率
    "batch_size": 2048,  # 批处理大小
    "gamma": 0.985,  # 折扣因子，控制对未来奖励的重视程度
    "seed": 312,  # 随机种子，确保实验可重复
    "net_dimension": 512,  # 神经网络维度
    "target_step": 5000,  # 目标步数
    "eval_gap": 30,  # 评估间隔
    "eval_times": 64,  # 评估次数（修正：防止KeyError）
}

# RLlib (Ray RLlib) - Ray生态的强化学习库参数
RLlib_PARAMS = {
    "lr": 5e-5,  # 学习率
    "train_batch_size": 500,  # 训练批次大小
    "gamma": 0.99,  # 折扣因子
}

# ==================== 时区配置 ====================
# 不同金融市场的时区设置，影响交易时间和数据同步

TIME_ZONE_SHANGHAI = (
    "Asia/Shanghai"  # 亚洲市场：恒生指数(HSI)、上证指数(SSE)、中证指数(CSI)
)
TIME_ZONE_USEASTERN = (
    "US/Eastern"  # 美国市场：道琼斯(Dow)、纳斯达克(Nasdaq)、标普500(SP)
)
TIME_ZONE_PARIS = "Europe/Paris"  # 欧洲市场：CAC 40指数
TIME_ZONE_BERLIN = "Europe/Berlin"  # 德国市场：DAX、TECDAX、MDAX、SDAX指数
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # 印尼市场：LQ45指数
TIME_ZONE_SELFDEFINED = "xxx"  # 自定义时区
USE_TIME_ZONE_SELFDEFINED = 0  # 是否使用自定义时区 (0=否, 1=是)

# ==================== 数据源API配置 ====================
# 各种金融数据提供商的API配置信息

# Alpaca - 美股交易和数据服务提供商
ALPACA_API_KEY = "xxx"  # 你的Alpaca API密钥（需要注册获取）
ALPACA_API_SECRET = "xxx"  # 你的Alpaca API密钥（需要注册获取）
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # Alpaca模拟交易API地址

# Binance - 全球最大的加密货币交易所
BINANCE_BASE_URL = "https://data.binance.vision/"  # Binance历史数据API地址
