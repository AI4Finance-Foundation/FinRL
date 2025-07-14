"""
FinRL 框架主入口文件

这是 FinRL（Financial Reinforcement Learning）金融强化学习框架的主要执行入口。
FinRL 是一个专门用于量化金融和算法交易的深度强化学习框架。

核心功能模式：
1. train  - 训练DRL模型：使用历史金融数据训练深度强化学习智能体
2. test   - 模型测试：在测试数据集上评估训练好的模型性能
3. trade  - 实盘交易：部署模型进行模拟交易或实盘交易

金融术语解释：
- DRL (Deep Reinforcement Learning): 深度强化学习，结合神经网络和强化学习
- Quantitative Finance: 量化金融，使用数学模型进行金融决策
- Algorithmic Trading: 算法交易，使用计算机程序自动执行交易
- Backtesting: 回测，使用历史数据验证交易策略的有效性
- Paper Trading: 模拟交易，使用虚拟资金测试交易策略

系统架构：
FinRL 采用三层架构设计：
- Environment Layer (环境层): 模拟金融市场环境
- Agent Layer (智能体层): DRL算法实现
- Application Layer (应用层): 具体的金融应用场景

作者：AI4Finance Foundation
项目地址：https://github.com/AI4Finance-Foundation/FinRL
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from typing import List

from finrl.config import ALPACA_API_BASE_URL  # Alpaca交易平台API地址
from finrl.config import DATA_SAVE_DIR  # 数据存储目录
from finrl.config import ERL_PARAMS  # ElegantRL算法参数
from finrl.config import INDICATORS  # 技术指标列表
from finrl.config import RESULTS_DIR  # 结果存储目录
from finrl.config import TENSORBOARD_LOG_DIR  # TensorBoard日志目录
from finrl.config import TEST_END_DATE  # 测试结束日期
from finrl.config import TEST_START_DATE  # 测试开始日期
from finrl.config import TRADE_END_DATE  # 交易结束日期
from finrl.config import TRADE_START_DATE  # 交易开始日期
from finrl.config import TRAIN_END_DATE  # 训练结束日期
from finrl.config import TRAIN_START_DATE  # 训练开始日期
from finrl.config import TRAINED_MODEL_DIR  # 训练模型存储目录
from finrl.config_tickers import DOW_30_TICKER  # 道琼斯30指数成分股代码列表
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

# 导入配置参数
# 导入股票代码配置
# 导入股票交易环境

# 构建交易环境 - 这是强化学习的核心概念
# 环境(Environment)定义了智能体(Agent)可以观察的状态、可以执行的动作以及奖励机制


def build_parser():
    """
    构建命令行参数解析器

    支持的运行模式：
    - train: 训练模式，使用历史数据训练DRL模型
    - test: 测试模式，在测试集上验证模型性能
    - trade: 交易模式，部署模型进行实际交易

    Returns:
        ArgumentParser: 配置好的命令行参数解析器
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="运行模式选择: train(训练), test(测试), trade(交易)",
        metavar="MODE",
        default="train",  # 默认为训练模式
    )
    return parser


def check_and_make_directories(directories: list[str]):
    """
    检查并创建必要的目录结构

    金融量化系统需要规范的目录结构来存储：
    - 原始和处理后的金融数据
    - 训练好的模型文件
    - 实验日志和可视化结果
    - 回测和交易结果

    Args:
        directories: 需要创建的目录列表
    """
    for directory in directories:
        directory_path = "./" + directory
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"创建目录: {directory_path}")


def main() -> int:
    """
    FinRL框架主函数

    根据命令行参数执行相应的操作模式：

    1. 训练模式 (train):
       - 下载和预处理金融数据
       - 计算技术指标
       - 训练深度强化学习模型
       - 保存训练好的模型

    2. 测试模式 (test):
       - 加载训练好的模型
       - 在测试数据集上进行回测
       - 评估模型的交易性能
       - 生成性能分析报告

    3. 交易模式 (trade):
       - 连接真实的交易平台API
       - 实时获取市场数据
       - 基于模型预测执行交易决策
       - 支持模拟交易和实盘交易

    Returns:
        int: 程序退出状态码 (0=成功, 非0=失败)
    """
    # 解析命令行参数
    parser = build_parser()
    options = parser.parse_args()

    # 创建必要的目录结构
    check_and_make_directories(
        [
            DATA_SAVE_DIR,  # 数据存储目录
            TRAINED_MODEL_DIR,  # 模型存储目录
            TENSORBOARD_LOG_DIR,  # 训练日志目录
            RESULTS_DIR,  # 结果存储目录
        ]
    )

    # ==================== 训练模式 ====================
    if options.mode == "train":
        print("🚀 启动训练模式 - 开始训练深度强化学习交易智能体")

        from finrl import train

        # 设置交易环境类
        env = StockTradingEnv

        # 演示使用 ElegantRL 算法库
        # kwargs: 额外参数字典，对于Yahoo Finance数据源通常为空
        # 对于其他数据源（如聚宽joinquant），可能需要额外的认证参数
        kwargs = {}

        # 开始训练过程
        train(
            start_date=TRAIN_START_DATE,  # 训练数据开始日期
            end_date=TRAIN_END_DATE,  # 训练数据结束日期
            ticker_list=DOW_30_TICKER,  # 股票代码列表（道琼斯30指数）
            data_source="yahoofinance",  # 数据源：Yahoo Finance（免费、可靠）
            time_interval="1D",  # 时间间隔：1天（日线数据）
            technical_indicator_list=INDICATORS,  # 技术指标列表
            drl_lib="elegantrl",  # DRL算法库选择
            env=env,  # 交易环境类
            model_name="ppo",  # 模型名称：PPO算法
            cwd="./test_ppo",  # 模型保存目录
            erl_params=ERL_PARAMS,  # ElegantRL参数配置
            break_step=1e5,  # 训练步数上限
            kwargs=kwargs,  # 额外参数
        )

    # ==================== 测试模式 ====================
    elif options.mode == "test":
        print("📊 启动测试模式 - 在测试集上评估模型性能")

        from finrl import test

        env = StockTradingEnv
        kwargs = {}  # 对于Yahoo Finance数据源，额外参数为空

        # 执行模型测试和回测
        account_value_erl = (
            test(  # noqa: F841 (变量已定义但未使用，这里保留用于后续分析)
                start_date=TEST_START_DATE,  # 测试数据开始日期
                end_date=TEST_END_DATE,  # 测试数据结束日期
                ticker_list=DOW_30_TICKER,  # 股票代码列表
                data_source="yahoofinance",  # 数据源
                time_interval="1D",  # 时间间隔
                technical_indicator_list=INDICATORS,  # 技术指标列表
                drl_lib="elegantrl",  # DRL算法库
                env=env,  # 交易环境
                model_name="ppo",  # 模型名称
                cwd="./test_ppo",  # 模型文件路径
                net_dimension=512,  # 神经网络维度
                kwargs=kwargs,  # 额外参数
            )
        )

    # ==================== 交易模式 ====================
    elif options.mode == "trade":
        print("💰 启动交易模式 - 部署模型进行实际交易")

        from finrl import trade

        # 导入私有API配置（需要用户自行配置）
        try:
            from finrl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
        except ImportError:
            raise FileNotFoundError(
                "❌ 未找到API配置文件！\n"
                "请在 config_private.py 中设置您的 ALPACA_API_KEY 和 ALPACA_API_SECRET\n"
                "这些密钥用于连接Alpaca交易平台进行实际交易"
            )

        env = StockTradingEnv
        kwargs = {}

        # 执行交易
        trade(
            start_date=TRADE_START_DATE,  # 交易开始日期
            end_date=TRADE_END_DATE,  # 交易结束日期
            ticker_list=DOW_30_TICKER,  # 交易股票列表
            data_source="yahoofinance",  # 数据源
            time_interval="1D",  # 数据时间间隔
            technical_indicator_list=INDICATORS,  # 技术指标
            drl_lib="elegantrl",  # DRL算法库
            env=env,  # 交易环境
            model_name="ppo",  # 使用的模型
            API_KEY=ALPACA_API_KEY,  # Alpaca API密钥
            API_SECRET=ALPACA_API_SECRET,  # Alpaca API秘钥
            API_BASE_URL=ALPACA_API_BASE_URL,  # Alpaca API地址
            trade_mode="paper_trading",  # 交易模式：模拟交易（降低风险）
            if_vix=True,  # 是否包含VIX恐慌指数（市场情绪指标）
            kwargs=kwargs,  # 额外参数
            # 状态空间维度计算（PPO算法必需参数）
            # 状态空间包括：股票数量 × (技术指标数量 + OHLC价格 + 持仓) + 账户信息 + VIX指数
            state_dim=len(DOW_30_TICKER) * (len(INDICATORS) + 3) + 3,
            # 动作空间维度：每只股票的买卖决策
            action_dim=len(DOW_30_TICKER),
        )
    else:
        # 无效的运行模式
        raise ValueError(
            f"❌ 无效的运行模式: {options.mode}\n"
            "请使用以下模式之一: train, test, trade"
        )

    print("✅ 程序执行完成！")
    return 0


# 程序入口点
# 用户可以在终端输入以下命令来运行不同模式：
# python main.py --mode=train   # 训练深度强化学习模型
# python main.py --mode=test    # 测试模型性能，进行回测分析
# python main.py --mode=trade   # 部署模型进行实际交易
if __name__ == "__main__":
    raise SystemExit(main())
