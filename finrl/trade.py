"""
FinRL 实盘交易和模拟交易模块

本模块实现了训练好的深度强化学习模型的实际部署，包括：
1. 回测模式：使用历史数据验证策略性能
2. 模拟交易模式：使用虚拟资金进行实时交易测试
3. 实盘交易模式：使用真实资金进行自动化交易

交易模式说明：
- Backtesting（回测）：使用历史数据验证策略，无真实资金风险
- Paper Trading（模拟交易）：使用实时数据但虚拟资金，测试策略的实时表现
- Live Trading（实盘交易）：使用真实资金和实时数据进行自动化交易

风险管理要点：
- 建议先进行充分的回测和模拟交易验证
- 实盘交易前请设置合理的风险控制参数
- 监控市场异常情况和系统故障
- 设置止损和仓位限制

支持的交易平台：
- Alpaca：美股零佣金交易平台，支持模拟和实盘交易
- 其他平台可通过扩展接口支持

作者：AI4Finance Foundation
"""

from __future__ import annotations

from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.test import test

# 导入模拟交易实现
# 导入测试模块用于回测


def trade(
    start_date,  # 交易开始日期
    end_date,  # 交易结束日期
    ticker_list,  # 交易股票列表
    data_source,  # 数据源
    time_interval,  # 数据时间间隔
    technical_indicator_list,  # 技术指标列表
    drl_lib,  # DRL算法库
    env,  # 交易环境类
    model_name,  # 模型名称
    API_KEY,  # 交易平台API密钥
    API_SECRET,  # 交易平台API秘钥
    API_BASE_URL,  # 交易平台API地址
    trade_mode="backtesting",  # 交易模式选择
    if_vix=True,  # 是否包含VIX指数
    **kwargs,  # 其他参数
):
    """
    深度强化学习交易模型部署函数

    这个函数是FinRL框架的最终部署环节，将训练好的DRL模型应用到实际的
    交易场景中。支持多种交易模式，从安全的回测到真实的资金交易。

    交易模式详解：

    1. Backtesting（回测模式）：
       - 使用历史数据验证策略
       - 无资金风险，适合策略验证
       - 快速评估模型在历史市场中的表现

    2. Paper Trading（模拟交易模式）：
       - 使用实时市场数据
       - 虚拟资金交易，无真实损失
       - 测试模型在实时市场中的表现
       - 验证交易执行的延迟和滑点影响

    3. Live Trading（实盘交易模式）：
       - 使用真实资金和实时数据
       - 需要谨慎使用，建议充分测试后部署
       - 需要严格的风险管理和监控

    Args:
        start_date: 交易开始日期，格式'YYYY-MM-DD'
        end_date: 交易结束日期
        ticker_list: 要交易的股票代码列表
        data_source: 数据源，建议与训练时保持一致
        time_interval: 数据时间间隔，如'1D'表示日线
        technical_indicator_list: 技术指标列表，必须与训练时一致
        drl_lib: DRL算法库，与训练时保持一致
        env: 交易环境类
        model_name: 模型算法名称
        API_KEY: 交易平台API密钥（Alpaca等）
        API_SECRET: 交易平台API秘钥
        API_BASE_URL: 交易平台API地址
        trade_mode: 交易模式，'backtesting'或'paper_trading'
        if_vix: 是否包含VIX指数，与训练时设置保持一致
        **kwargs: 其他参数，如模型路径、网络维度、风险参数等

    风险警告：
        实盘交易涉及真实资金损失风险，请在充分验证策略有效性后谨慎使用。
        建议设置合理的止损、仓位限制和监控机制。
    """

    print(f"🚀 启动交易模式: {trade_mode}")
    print(f"📅 交易时间范围: {start_date} 到 {end_date}")
    print(f"📈 交易标的: {len(ticker_list)} 只股票")

    if trade_mode == "backtesting":
        # ==================== 回测模式 ====================
        print("📊 执行回测模式 - 使用历史数据验证策略性能")
        print("  ✅ 无资金风险，适合策略验证和性能评估")

        # 直接调用测试函数进行回测
        # 回测模式使用历史数据，不涉及真实交易API
        test(
            start_date,  # 回测开始日期
            end_date,  # 回测结束日期
            ticker_list,  # 股票列表
            data_source,  # 数据源
            time_interval,  # 时间间隔
            technical_indicator_list,  # 技术指标
            drl_lib,  # DRL库
            env,  # 环境
            model_name,  # 模型名称
            if_vix=if_vix,  # VIX设置
            **kwargs,  # 其他参数
        )

        print("✅ 回测完成！请查看results目录中的性能分析报告")

    elif trade_mode == "paper_trading":
        # ==================== 模拟交易模式 ====================
        print("💰 执行模拟交易模式 - 使用实时数据和虚拟资金")
        print("  ⚠️ 虚拟资金交易，无真实损失，但需要交易平台API")

        # ==================== 参数验证和读取 ====================
        print("⚙️ 读取交易参数...")

        try:
            # 神经网络维度，影响模型推理性能
            net_dim = kwargs.get("net_dimension", 2**7)  # 默认128维

            # 模型文件路径，指向训练好的模型
            cwd = kwargs.get("cwd", "./" + str(model_name))

            # 状态空间维度，必须与训练时完全一致
            # 状态空间包括：股票价格、技术指标、持仓信息、账户状态等
            state_dim = kwargs.get("state_dim")

            # 动作空间维度，通常等于股票数量
            # 每只股票对应一个动作（买入/卖出/持有的比例）
            action_dim = kwargs.get("action_dim")

            print(f"  🧠 神经网络维度: {net_dim}")
            print(f"  📁 模型路径: {cwd}")
            print(f"  📊 状态空间维度: {state_dim}")
            print(f"  🎯 动作空间维度: {action_dim}")

        except Exception as e:
            raise ValueError(
                f"❌ 参数读取失败: {str(e)}\n"
                "请检查以下参数是否正确设置:\n"
                "- net_dimension: 神经网络维度\n"
                "- cwd: 模型文件路径\n"
                "- state_dim: 状态空间维度\n"
                "- action_dim: 动作空间维度"
            )

        # ==================== 模拟交易环境初始化 ====================
        print("🏗️ 初始化Alpaca模拟交易环境...")

        # 创建Alpaca模拟交易实例
        # AlpacaPaperTrading 封装了与Alpaca API的交互逻辑
        paper_trading = AlpacaPaperTrading(
            ticker_list,  # 交易股票列表
            time_interval,  # 数据时间间隔
            drl_lib,  # DRL算法库
            model_name,  # 模型名称
            cwd,  # 模型文件路径
            net_dim,  # 神经网络维度
            state_dim,  # 状态空间维度
            action_dim,  # 动作空间维度
            API_KEY,  # Alpaca API密钥
            API_SECRET,  # Alpaca API秘钥
            API_BASE_URL,  # Alpaca API地址
            technical_indicator_list,  # 技术指标列表
            # ==================== 风险控制参数 ====================
            turbulence_thresh=30,  # 市场波动度阈值，超过时停止交易
            # 这是重要的风险控制机制，避免在极端市场条件下交易
            max_stock=1e2,  # 单只股票最大持仓数量限制
            # 防止过度集中投资于单一标的
            latency=None,  # 交易延迟设置，None表示使用默认值
            # 模拟真实交易中的网络和执行延迟
        )

        print("  ✅ Alpaca模拟交易环境初始化完成")
        print("  📊 风险控制参数:")
        print(f"    - 波动度阈值: {30} (超过时暂停交易)")
        print(f"    - 单股最大持仓: {1e2} 股")

        # ==================== 开始模拟交易 ====================
        print("🚀 开始执行模拟交易...")
        print("  📡 连接Alpaca API获取实时市场数据...")
        print("  🤖 DRL模型将根据实时数据做出交易决策...")
        print("  💼 所有交易将使用虚拟资金执行...")

        # 运行模拟交易
        # 这会启动一个持续运行的交易循环，直到交易时间结束
        paper_trading.run()

        print("✅ 模拟交易完成！")
        print("📊 请查看Alpaca账户页面查看交易记录和性能统计")

    else:
        # ==================== 无效交易模式 ====================
        raise ValueError(
            f"❌ 无效的交易模式: '{trade_mode}'\n"
            "请选择以下模式之一:\n"
            "- 'backtesting': 回测模式，使用历史数据验证策略\n"
            "- 'paper_trading': 模拟交易模式，使用实时数据和虚拟资金\n"
            "\n注意：实盘交易功能需要额外的风险管理和监控机制"
        )

    print("🎉 交易任务执行完成！")
