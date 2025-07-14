"""
FinRL 模型测试和回测模块

本模块实现了训练好的深度强化学习模型的测试和回测功能，包括：
1. 加载已训练的DRL模型
2. 在测试数据集上进行回测
3. 评估模型的交易性能
4. 生成投资组合价值时间序列

回测（Backtesting）是量化金融中的核心概念：
- 使用历史数据验证交易策略的有效性
- 评估策略的收益率、风险和稳定性
- 确保模型在"未见过"的数据上的泛化能力
- 避免过拟合和前瞻偏差

关键金融指标：
- Total Return: 总收益率，衡量投资策略的盈利能力
- Sharpe Ratio: 夏普比率，衡量风险调整后的收益
- Maximum Drawdown: 最大回撤，衡量策略的最大损失
- Volatility: 波动率，衡量投资回报的不确定性
- Portfolio Value: 投资组合价值，反映策略的累计表现

作者：AI4Finance Foundation
"""
from __future__ import annotations

# 导入配置参数
from finrl.config import INDICATORS      # 技术指标列表
from finrl.config import RLlib_PARAMS    # RLlib算法参数
from finrl.config import TEST_END_DATE   # 测试结束日期
from finrl.config import TEST_START_DATE # 测试开始日期

# 导入股票代码配置
from finrl.config_tickers import DOW_30_TICKER # 道琼斯30指数成分股

# 导入交易环境
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def test(
    start_date,               # 测试开始日期
    end_date,                 # 测试结束日期
    ticker_list,              # 股票代码列表
    data_source,              # 数据源
    time_interval,            # 时间间隔
    technical_indicator_list, # 技术指标列表
    drl_lib,                  # DRL算法库
    env,                      # 交易环境类
    model_name,               # 模型名称
    if_vix=True,              # 是否包含VIX指数
    **kwargs,                 # 其他参数
):
    """
    深度强化学习模型测试和回测函数
    
    这个函数实现了完整的模型验证流程，用于评估训练好的DRL模型在"未见过"的
    测试数据上的表现。这是量化金融中至关重要的一步，确保模型具有良好的
    泛化能力，能够在真实市场环境中稳定运行。
    
    回测流程说明：
    1. 数据准备：获取测试期间的市场数据，确保数据质量和一致性
    2. 环境重建：使用与训练时相同的参数重建交易环境
    3. 模型加载：载入训练好的DRL模型参数
    4. 策略执行：让模型在测试数据上执行交易决策
    5. 性能评估：计算各种金融绩效指标
    
    Args:
        start_date: 测试数据开始日期，通常是训练结束日期之后
        end_date: 测试数据结束日期
        ticker_list: 测试的股票代码列表，应与训练时保持一致
        data_source: 数据源，建议与训练时使用相同的数据源
        time_interval: 数据时间间隔，与训练时保持一致
        technical_indicator_list: 技术指标列表，必须与训练时完全一致
        drl_lib: DRL算法库，与训练时使用的库保持一致
        env: 交易环境类，与训练时使用的环境保持一致
        model_name: 模型算法名称，与训练时保持一致
        if_vix: 是否包含VIX指数，与训练时设置保持一致
        **kwargs: 其他参数，如模型路径、网络维度等
    
    Returns:
        list: 回测期间的投资组合价值时间序列，用于后续性能分析
    
    金融术语解释：
    - Out-of-sample testing: 样本外测试，使用未参与训练的数据测试模型
    - Walk-forward analysis: 前向分析，模拟真实交易中的时间推进过程
    - Portfolio rebalancing: 投资组合再平衡，根据模型信号调整持仓
    """
    
    # ==================== 数据处理模块导入 ====================
    # 在函数内部导入，避免循环导入问题
    from finrl.meta.data_processor import DataProcessor

    # ==================== 测试数据获取阶段 ====================
    print("📊 开始获取测试数据进行回测...")
    
    # 初始化数据处理器
    # 使用与训练时相同的数据处理器确保数据格式一致性
    dp = DataProcessor(data_source, **kwargs)
    
    # 获取测试期间的市场数据
    # 重要：测试数据必须是模型"从未见过"的数据，确保评估的公正性
    print(f"  📈 从 {data_source} 获取 {start_date} 到 {end_date} 的测试数据...")
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    
    # 数据清洗：与训练时使用相同的清洗步骤
    print("  🧹 清洗测试数据...")
    data = dp.clean_data(data)
    
    # 计算技术指标：必须与训练时完全一致
    print("  📊 计算技术指标...")
    data = dp.add_technical_indicator(data, technical_indicator_list)
    print(f"  ✅ 已计算 {len(technical_indicator_list)} 个技术指标")

    # 添加VIX恐慌指数（如果在训练时包含）
    if if_vix:
        print("  😱 添加VIX恐慌指数...")
        data = dp.add_vix(data)
    
    # ==================== 数据格式转换阶段 ====================
    print("🔄 转换数据格式...")
    
    # 将数据转换为环境所需的数组格式
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    
    print(f"  📏 测试数据维度 - 价格: {price_array.shape}, "
          f"技术指标: {tech_array.shape}, 波动度: {turbulence_array.shape}")

    # ==================== 测试环境构建阶段 ====================
    print("🏗️ 构建测试环境...")
    
    # 配置测试环境参数
    # 注意：if_train=False 表示这是测试模式，不进行参数更新
    env_config = {
        "price_array": price_array,           # 测试期间的价格数据
        "tech_array": tech_array,             # 测试期间的技术指标数据
        "turbulence_array": turbulence_array, # 测试期间的市场波动度数据
        "if_train": False,                    # 关键：设置为False，表示测试模式
    }
    
    # 实例化测试环境
    env_instance = env(config=env_config)
    print("  ✅ 测试环境构建完成")

    # ==================== 模型参数配置 ====================
    # 加载ElegantRL需要的网络维度参数
    net_dimension = kwargs.get("net_dimension", 2**7)  # 神经网络维度，默认128
    cwd = kwargs.get("cwd", "./" + str(model_name))     # 模型文件路径
    
    print(f"📁 模型路径: {cwd}")
    print(f"🧠 网络维度: {net_dimension}")
    print(f"📏 价格数组长度: {len(price_array)}")

    # ==================== 模型加载和预测阶段 ====================
    print(f"🤖 使用 {drl_lib} 库加载并测试 {model_name} 模型...")
    
    if drl_lib == "elegantrl":
        # ========== ElegantRL 模型测试 ==========
        print("🎯 使用 ElegantRL 进行模型推理...")
        
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        # 执行模型预测和回测
        # DRL_prediction 方法会加载训练好的模型并在测试环境中运行
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,        # 模型算法名称
            cwd=cwd,                      # 模型文件路径
            net_dimension=net_dimension,  # 网络维度
            environment=env_instance,     # 测试环境实例
        )
        
        print("  ✅ ElegantRL 回测完成")
        return episode_total_assets
        
    elif drl_lib == "rllib":
        # ========== RLlib 模型测试 ==========
        print("⚡ 使用 RLlib 进行模型推理...")
        
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        # RLlib的预测方法需要原始数据数组
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,            # 模型名称
            env=env,                          # 环境类
            price_array=price_array,          # 价格数据
            tech_array=tech_array,            # 技术指标数据
            turbulence_array=turbulence_array, # 波动度数据
            agent_path=cwd,                   # 模型路径
        )
        
        print("  ✅ RLlib 回测完成")
        return episode_total_assets
        
    elif drl_lib == "stable_baselines3":
        # ========== Stable Baselines3 模型测试 ==========
        print("🔧 使用 Stable Baselines3 进行模型推理...")
        
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        # 从文件加载模型并执行预测
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name,    # 模型名称
            environment=env_instance, # 测试环境
            cwd=cwd,                  # 模型文件路径
        )
        
        print("  ✅ Stable Baselines3 回测完成")
        return episode_total_assets
        
    else:
        # 不支持的DRL库
        raise ValueError(f"❌ 不支持的DRL库: {drl_lib}\n"
                        "请选择以下之一: 'elegantrl', 'rllib', 'stable_baselines3'")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    """
    测试模块的独立运行入口
    
    当直接运行此文件时，会使用默认参数对训练好的模型进行回测
    这主要用于开发、调试和快速验证模型性能
    """
    print("🎯 独立运行测试模块 - 使用默认参数进行回测")
    
    # 使用股票交易环境
    env = StockTradingEnv

    # ElegantRL 测试演示
    print("\n=== ElegantRL 模型回测演示 ===")
    kwargs = {}  # 对于Yahoo Finance数据源，额外参数为空

    # 执行回测并获取投资组合价值序列
    account_value_erl = test(
        start_date=TEST_START_DATE,               # 测试开始日期
        end_date=TEST_END_DATE,                   # 测试结束日期
        ticker_list=DOW_30_TICKER,                # 道琼斯30指数成分股
        data_source="yahoofinance",               # 数据源：Yahoo Finance
        time_interval="1D",                       # 日线数据
        technical_indicator_list=INDICATORS,      # 技术指标列表
        drl_lib="elegantrl",                      # 使用ElegantRL库
        env=env,                                  # 交易环境
        model_name="ppo",                         # PPO算法
        cwd="./test_ppo",                         # 模型文件路径
        net_dimension=512,                        # 神经网络维度
        kwargs=kwargs,                            # 额外参数
    )
    
    print(f"📈 回测完成！最终投资组合价值: ${account_value_erl[-1]:,.2f}")
    print(f"📊 投资组合价值序列长度: {len(account_value_erl)} 个交易日")

    # ==================== 其他算法库演示 ====================
    # 用户可以取消以下注释来测试其他DRL算法库训练的模型
    
    # # RLlib 模型回测演示
    # print("\n=== RLlib 模型回测演示 ===")
    # import ray
    # ray.shutdown()  # 关闭之前的Ray会话
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",                          # 使用RLlib库
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30", # RLlib的模型路径格式
    #     rllib_params=RLlib_PARAMS,
    # )
    # print(f"📈 RLlib回测完成！最终投资组合价值: ${account_value_rllib[-1]:,.2f}")
    #
    # # Stable Baselines3 模型回测演示  
    # print("\n=== Stable Baselines3 模型回测演示 ===")
    # account_value_sb3 = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",              # 使用Stable Baselines3库
    #     env=env,
    #     model_name="sac",                         # SAC算法
    #     cwd="./test_sac.zip",                     # SB3的模型文件格式
    # )
    # print(f"📈 SB3回测完成！最终投资组合价值: ${account_value_sb3[-1]:,.2f}")
    
    print("\n🎉 所有回测演示完成！")
