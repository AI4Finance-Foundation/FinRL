"""
FinRL 深度强化学习模型训练模块

本模块实现了金融强化学习模型的完整训练流程，包括：
1. 金融数据获取和预处理
2. 技术指标计算
3. 市场环境构建
4. DRL智能体训练
5. 模型保存和评估

支持的DRL算法库：
- ElegantRL: AI4Finance自研的高性能DRL库
- RLlib: Ray生态的分布式强化学习库
- Stable Baselines3: 稳定且易用的DRL算法实现

金融概念说明：
- Technical Indicators: 技术指标，基于价格和成交量计算的数学指标
- VIX Index: 波动率指数，衡量市场恐慌程度的指标
- OHLCV Data: 开盘价(Open)、最高价(High)、最低价(Low)、收盘价(Close)、成交量(Volume)
- Turbulence: 市场波动度，用于风险控制
- Environment: 强化学习环境，模拟真实的交易市场

作者：AI4Finance Foundation
"""

from __future__ import annotations

from finrl.config import ERL_PARAMS  # ElegantRL算法参数
from finrl.config import INDICATORS  # 技术指标列表
from finrl.config import RLlib_PARAMS  # RLlib算法参数
from finrl.config import SAC_PARAMS  # SAC算法参数
from finrl.config import TRAIN_END_DATE  # 训练结束日期
from finrl.config import TRAIN_START_DATE  # 训练开始日期
from finrl.config_tickers import DOW_30_TICKER  # 道琼斯30指数成分股
from finrl.meta.data_processor import DataProcessor  # 数据处理器
from finrl.meta.env_stock_trading.env_stocktrading_np import (
    StockTradingEnv,
)  # 股票交易环境

# 导入配置参数
# 导入股票代码配置
# 导入核心组件

# 构建交易环境


def train(
    start_date,  # 训练开始日期
    end_date,  # 训练结束日期
    ticker_list,  # 股票代码列表
    data_source,  # 数据源（如yahoofinance）
    time_interval,  # 时间间隔（如1D表示日线）
    technical_indicator_list,  # 技术指标列表
    drl_lib,  # DRL算法库选择
    env,  # 交易环境类
    model_name,  # 模型名称
    if_vix=True,  # 是否添加VIX恐慌指数
    **kwargs,  # 其他可选参数
):
    """
    深度强化学习交易模型训练函数

    这个函数实现了完整的DRL训练流水线：

    1. 数据获取阶段：
       - 从指定数据源下载股票的OHLCV数据
       - 数据清洗，处理缺失值和异常值

    2. 特征工程阶段：
       - 计算技术指标（MACD、布林带、RSI等）
       - 添加VIX恐慌指数作为市场情绪指标
       - 计算市场波动度（turbulence）用于风险控制

    3. 环境构建阶段：
       - 将处理后的数据转换为强化学习环境所需的格式
       - 构建状态空间、动作空间和奖励函数

    4. 模型训练阶段：
       - 根据选择的DRL库初始化智能体
       - 设置训练超参数
       - 执行训练过程并保存模型

    Args:
        start_date: 训练数据起始日期，格式为'YYYY-MM-DD'
        end_date: 训练数据结束日期
        ticker_list: 要交易的股票代码列表，如['AAPL', 'MSFT', 'GOOGL']
        data_source: 数据源名称，支持'yahoofinance', 'alpaca', 'joinquant'等
        time_interval: 数据时间间隔，'1D'表示日线，'1H'表示小时线
        technical_indicator_list: 技术指标列表，如['macd', 'rsi_30', 'boll_ub']
        drl_lib: DRL算法库，可选'elegantrl', 'rllib', 'stable_baselines3'
        env: 交易环境类，通常是StockTradingEnv
        model_name: 模型算法名称，如'ppo', 'sac', 'ddpg'
        if_vix: 是否包含VIX指数，有助于捕捉市场恐慌情绪
        **kwargs: 其他参数，如模型保存路径、训练步数等
    """

    # ==================== 数据获取和预处理阶段 ====================
    print("📈 开始数据获取和预处理...")

    # 初始化数据处理器
    # DataProcessor负责从各种数据源获取金融数据并进行标准化处理
    dp = DataProcessor(data_source, **kwargs)

    # 下载原始OHLCV数据
    # OHLCV是金融数据的标准格式：开盘价、最高价、最低价、收盘价、成交量
    print(f"  📊 从 {data_source} 下载 {len(ticker_list)} 只股票的数据...")
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)

    # 数据清洗：处理缺失值、异常值、数据不一致等问题
    print("  🧹 清洗数据，处理缺失值和异常值...")
    data = dp.clean_data(data)

    # ==================== 技术指标计算阶段 ====================
    print("📊 计算技术指标...")

    # 添加技术指标
    # 技术指标是量化分析的核心工具，帮助识别市场趋势和交易信号
    data = dp.add_technical_indicator(data, technical_indicator_list)
    print(f"  ✅ 已添加 {len(technical_indicator_list)} 个技术指标")

    # 添加VIX恐慌指数（可选）
    # VIX是衡量市场恐慌程度的重要指标，高VIX通常表示市场不稳定
    if if_vix:
        print("  😱 添加VIX恐慌指数作为市场情绪指标...")
        data = dp.add_vix(data)

    # ==================== 数据格式转换阶段 ====================
    print("🔄 转换数据格式用于强化学习...")

    # 将DataFrame格式的数据转换为强化学习环境所需的numpy数组格式
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    print(f"  📏 价格数组形状: {price_array.shape}")
    print(f"  📏 技术指标数组形状: {tech_array.shape}")
    print(f"  📏 波动度数组形状: {turbulence_array.shape}")

    # ==================== 环境配置阶段 ====================
    print("🏗️ 构建强化学习交易环境...")

    # 配置环境参数
    env_config = {
        "price_array": price_array,  # 价格数据：包含OHLC价格信息
        "tech_array": tech_array,  # 技术指标数据：MACD、RSI、布林带等
        "turbulence_array": turbulence_array,  # 市场波动度：用于风险控制
        "if_train": True,  # 标记为训练模式
    }

    # 实例化交易环境
    # 环境定义了智能体的观察空间、动作空间和奖励机制
    env_instance = env(config=env_config)
    print("  ✅ 交易环境构建完成")

    # ==================== 模型参数读取 ====================
    # 读取模型保存路径
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print(f"📁 模型将保存到: {cwd}")

    # ==================== DRL算法训练阶段 ====================
    print(f"🤖 开始使用 {drl_lib} 库训练 {model_name} 模型...")

    if drl_lib == "elegantrl":
        # ========== ElegantRL 训练流程 ==========
        print("🎯 使用 ElegantRL 算法库进行训练")

        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        # 获取训练参数
        break_step = kwargs.get("break_step", 1e6)  # 最大训练步数
        erl_params = kwargs.get("erl_params")  # ElegantRL专用参数

        print(f"  📊 最大训练步数: {break_step}")
        print(f"  ⚙️ 算法参数: {erl_params}")

        # 初始化ElegantRL智能体
        agent = DRLAgent_erl(
            env=env,  # 环境类
            price_array=price_array,  # 价格数据
            tech_array=tech_array,  # 技术指标数据
            turbulence_array=turbulence_array,  # 波动度数据
        )

        # 获取指定的模型（PPO、SAC、DDPG等）
        model = agent.get_model(model_name, model_kwargs=erl_params)
        print(f"  🧠 已初始化 {model_name} 模型")

        # 开始训练
        print("  🚀 开始训练过程...")
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
        print("  ✅ ElegantRL 训练完成！")

    elif drl_lib == "rllib":
        # ========== RLlib 训练流程 ==========
        print("⚡ 使用 RLlib 分布式强化学习库进行训练")

        # 获取训练参数
        total_episodes = kwargs.get("total_episodes", 100)  # 训练回合数
        rllib_params = kwargs.get("rllib_params")  # RLlib参数

        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        # 初始化RLlib智能体
        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )

        # 获取模型和配置
        model, model_config = agent_rllib.get_model(model_name)

        # 设置超参数
        model_config["lr"] = rllib_params["lr"]  # 学习率
        model_config["train_batch_size"] = rllib_params[
            "train_batch_size"
        ]  # 训练批次大小
        model_config["gamma"] = rllib_params["gamma"]  # 折扣因子

        print(f"  📊 训练回合数: {total_episodes}")
        print(f"  ⚙️ 模型配置: {model_config}")

        # 开始训练
        print("  🚀 开始分布式训练...")
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )

        # 保存模型
        trained_model.save(cwd)
        print("  ✅ RLlib 训练完成并保存模型！")

    elif drl_lib == "stable_baselines3":
        # ========== Stable Baselines3 训练流程 ==========
        print("🔧 使用 Stable Baselines3 库进行训练")

        # 获取训练参数
        total_timesteps = kwargs.get("total_timesteps", 1e6)  # 总训练步数
        agent_params = kwargs.get("agent_params")  # 算法参数

        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        # 初始化Stable Baselines3智能体
        agent = DRLAgent_sb3(env=env_instance)

        # 获取指定模型
        model = agent.get_model(model_name, model_kwargs=agent_params)
        print(f"  🧠 已初始化 {model_name} 模型")
        print(f"  📊 总训练步数: {total_timesteps}")

        # 开始训练
        print("  🚀 开始训练过程...")
        trained_model = agent.train_model(
            model=model,
            tb_log_name=model_name,  # TensorBoard日志名称
            total_timesteps=total_timesteps,
        )

        print("✅ 训练完成！")

        # 保存模型
        trained_model.save(cwd)
        print(f"📁 训练好的模型已保存到: {cwd}")

    else:
        # 不支持的DRL库
        raise ValueError(
            f"❌ 不支持的DRL库: {drl_lib}\n"
            "请选择以下之一: 'elegantrl', 'rllib', 'stable_baselines3'"
        )

    print("🎉 模型训练流程全部完成！")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    训练模块的独立运行入口

    当直接运行此文件时，会使用默认参数训练一个PPO模型
    这主要用于开发和测试目的
    """
    print("🎯 独立运行训练模块 - 使用默认参数")

    # 使用股票交易环境
    env = StockTradingEnv

    # ElegantRL 训练演示
    print("\n=== ElegantRL 训练演示 ===")
    kwargs = {}  # 对于Yahoo Finance数据源，额外参数为空

    train(
        start_date=TRAIN_START_DATE,  # 训练开始日期
        end_date=TRAIN_END_DATE,  # 训练结束日期
        ticker_list=DOW_30_TICKER,  # 道琼斯30指数成分股
        data_source="yahoofinance",  # 数据源：Yahoo Finance
        time_interval="1D",  # 日线数据
        technical_indicator_list=INDICATORS,  # 技术指标列表
        drl_lib="elegantrl",  # 使用ElegantRL库
        env=env,  # 交易环境
        model_name="ppo",  # PPO算法
        cwd="./test_ppo",  # 模型保存路径
        erl_params=ERL_PARAMS,  # ElegantRL参数
        break_step=1e5,  # 训练步数
        kwargs=kwargs,
    )

    # ==================== 其他算法库演示 ====================
    # 用户可以取消以下注释来尝试其他DRL算法库

    # # RLlib 训练演示
    # print("\n=== RLlib 训练演示 ===")
    # import ray
    # ray.shutdown()  # 关闭之前的Ray会话
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",                          # 使用RLlib库
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,                # RLlib参数
    #     total_episodes=30,                        # 训练回合数
    # )
    #
    # # Stable Baselines3 训练演示
    # print("\n=== Stable Baselines3 训练演示 ===")
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",              # 使用Stable Baselines3库
    #     env=env,
    #     model_name="sac",                         # SAC算法
    #     cwd="./test_sac",
    #     agent_params=SAC_PARAMS,                  # SAC参数
    #     total_timesteps=1e4,                      # 训练步数
    # )
