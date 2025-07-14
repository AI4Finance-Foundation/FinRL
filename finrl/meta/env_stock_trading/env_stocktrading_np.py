"""
FinRL 股票交易强化学习环境 (NumPy优化版)

这是FinRL框架的核心股票交易环境，基于OpenAI Gymnasium标准实现。
该环境模拟真实的股票交易市场，为深度强化学习智能体提供交互接口。

环境特点：
1. 高性能：使用NumPy优化，支持向量化计算
2. 真实性：模拟交易成本、市场波动和风险控制
3. 灵活性：支持多种交易策略和风险管理机制
4. 标准化：遵循Gymnasium接口，兼容所有DRL算法

强化学习要素：
- 状态(State)：包含账户信息、股票价格、技术指标和市场风险指标
- 动作(Action)：每只股票的买卖数量（连续动作空间）
- 奖励(Reward)：基于投资组合价值变化的即时回报
- 环境(Environment)：模拟的股票交易市场

金融概念：
- Portfolio：投资组合，包含现金和多只股票的组合
- Turbulence：市场波动度，用于风险控制
- Transaction Cost：交易成本，包括买入和卖出费用
- Asset Allocation：资产配置，在不同资产间分配资金

作者：AI4Finance Foundation
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy import random as rd


class StockTradingEnv(gym.Env):
    """
    股票交易强化学习环境
    
    这个环境类实现了一个完整的股票交易模拟器，智能体可以在其中
    学习如何进行多股票投资组合管理。环境支持连续动作空间，允许
    智能体对每只股票进行精确的买卖决策。
    
    环境设计原理：
    1. 状态空间：包含投资组合的完整信息
       - 账户现金
       - 市场风险指标（波动度）
       - 股票价格和持仓
       - 技术指标
    
    2. 动作空间：每只股票的交易数量
       - 正值：买入股票
       - 负值：卖出股票
       - 零值：不交易
    
    3. 奖励设计：基于投资组合价值变化
       - 短期奖励：每步的资产价值变化
       - 长期奖励：累计折扣奖励
    
    4. 风险控制：集成多种风险管理机制
       - 市场波动度监控
       - 强制平仓机制
       - 交易冷却期
    
    Attributes:
        price_ary: 股票价格数组 (时间, 股票数, 价格特征)
        tech_ary: 技术指标数组 (时间, 股票数, 指标数)
        turbulence_ary: 市场波动度数组 (时间,)
        state_dim: 状态空间维度
        action_dim: 动作空间维度（等于股票数量）
    """
    
    def __init__(
        self,
        config,                    # 环境配置字典
        initial_account=1e6,       # 初始账户资金
        gamma=0.99,                # 折扣因子
        turbulence_thresh=99,      # 市场波动度阈值
        min_stock_rate=0.1,        # 最小交易比例
        max_stock=1e2,             # 单次最大交易股数
        initial_capital=1e6,       # 初始资本（与initial_account相同）
        buy_cost_pct=1e-3,         # 买入交易成本比例（0.1%）
        sell_cost_pct=1e-3,        # 卖出交易成本比例（0.1%）
        reward_scaling=2**-11,     # 奖励缩放因子
        initial_stocks=None,       # 初始股票持仓
    ):
        """
        初始化股票交易环境
        
        Args:
            config (dict): 环境配置，包含:
                - price_array: 价格数据数组
                - tech_array: 技术指标数组
                - turbulence_array: 市场波动度数组
                - if_train: 是否为训练模式
            initial_account (float): 初始账户资金，默认100万
            gamma (float): 折扣因子，用于计算长期累计奖励
            turbulence_thresh (float): 市场波动度阈值，超过时强制平仓
            min_stock_rate (float): 最小交易比例，避免过小的交易
            max_stock (float): 单次最大交易股数限制
            initial_capital (float): 初始资本总额
            buy_cost_pct (float): 买入交易费用百分比
            sell_cost_pct (float): 卖出交易费用百分比
            reward_scaling (float): 奖励缩放系数，调节奖励大小
            initial_stocks (np.array): 初始股票持仓数量
        """
        print("🏗️ 初始化股票交易强化学习环境...")
        
        # ==================== 数据加载和预处理 ====================
        # 从配置中提取核心数据
        price_ary = config["price_array"]           # 股票价格时间序列
        tech_ary = config["tech_array"]             # 技术指标时间序列
        turbulence_ary = config["turbulence_array"] # 市场波动度时间序列
        if_train = config["if_train"]               # 训练/测试模式标志
        
        # 数据类型转换和标准化
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        # ==================== 数据标准化和预处理 ====================
        # 技术指标标准化：缩放到合适的数值范围
        # 2**-7 ≈ 0.0078，将技术指标值缩放到较小范围
        self.tech_ary = self.tech_ary * 2**-7
        
        # 市场波动度处理
        # 生成波动度布尔标志：True表示高波动（危险），False表示正常
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        
        # 使用sigmoid函数平滑处理波动度，避免数值过大
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        # ==================== 环境参数设置 ====================
        stock_dim = self.price_ary.shape[1]  # 股票数量
        
        # 强化学习参数
        self.gamma = gamma                    # 折扣因子，控制对未来奖励的重视程度
        
        # 交易限制参数
        self.max_stock = max_stock            # 单次最大交易股数
        self.min_stock_rate = min_stock_rate  # 最小交易比例
        
        # 交易成本参数（模拟真实交易费用）
        self.buy_cost_pct = buy_cost_pct      # 买入手续费
        self.sell_cost_pct = sell_cost_pct    # 卖出手续费
        
        # 奖励和资金参数
        self.reward_scaling = reward_scaling  # 奖励缩放因子
        self.initial_capital = initial_capital # 初始资本
        
        # 初始股票持仓设置
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)  # 默认空仓开始
            if initial_stocks is None
            else initial_stocks
        )

        # ==================== 环境状态变量初始化 ====================
        # 这些变量将在reset()方法中正式初始化
        self.day = None                    # 当前交易日
        self.amount = None                 # 账户现金
        self.stocks = None                 # 股票持仓数量
        self.total_asset = None            # 总资产价值
        self.gamma_reward = None           # 累计折扣奖励
        self.initial_total_asset = None    # 初始总资产

        # ==================== 强化学习环境信息 ====================
        self.env_name = "StockEnv"
        
        # 状态空间维度计算
        # 状态包含：现金(1) + 波动度指标(2) + 价格*股票数*3 + 技术指标维度
        # 3倍是因为包含：当前价格、持仓数量、冷却时间
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        
        # 交易冷却机制：防止过度频繁交易
        self.stocks_cd = None
        
        # 动作空间维度：每只股票一个动作
        self.action_dim = stock_dim
        
        # 环境参数
        self.max_step = self.price_ary.shape[0] - 1  # 最大步数
        self.if_train = if_train                     # 训练模式标志
        self.if_discrete = False                     # 连续动作空间
        self.target_return = 10.0                    # 目标收益率
        self.episode_return = 0.0                    # 回合收益率

        # ==================== Gymnasium标准接口定义 ====================
        # 观察空间：状态的数值范围
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000,              # 状态值范围
            shape=(self.state_dim,),           # 状态维度
            dtype=np.float32                   # 数据类型
        )
        
        # 动作空间：每只股票的交易数量（标准化后）
        self.action_space = gym.spaces.Box(
            low=-1, high=1,                    # 动作值范围[-1,1]
            shape=(self.action_dim,),          # 动作维度
            dtype=np.float32                   # 数据类型
        )
        
        print(f"  📊 环境参数:")
        print(f"    - 股票数量: {stock_dim}")
        print(f"    - 交易天数: {self.max_step + 1}")
        print(f"    - 状态维度: {self.state_dim}")
        print(f"    - 动作维度: {self.action_dim}")
        print(f"    - 初始资金: ${initial_capital:,.0f}")
        print(f"    - 交易成本: 买入{buy_cost_pct*100:.1f}%, 卖出{sell_cost_pct*100:.1f}%")
        print("  ✅ 股票交易环境初始化完成")

    def reset(
        self,
        *,
        seed=None,      # 随机种子
        options=None,   # 额外选项
    ):
        """
        重置环境到初始状态
        
        这个方法在每个回合开始时调用，将环境重置为初始状态。
        在训练模式下会添加随机性，在测试模式下保持确定性。
        
        Args:
            seed: 随机种子，用于可重复实验
            options: 额外的重置选项
        
        Returns:
            tuple: (初始状态, 信息字典)
        
        重置策略：
        - 训练模式：添加随机性，提高探索能力
        - 测试模式：使用固定初始值，确保结果可比较
        """
        print("🔄 重置交易环境...")
        
        # 重置交易日到第0天
        self.day = 0
        price = self.price_ary[self.day]  # 获取第一天的股票价格

        if self.if_train:
            # ========== 训练模式：添加随机性 ==========
            print("  🎲 训练模式：添加随机初始化")
            
            # 随机初始化股票持仓（0-64股随机数量）
            # 这种随机性有助于智能体学习不同的初始状态
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            
            # 初始化股票冷却时间（防止过度频繁交易）
            self.stocks_cool_down = np.zeros_like(self.stocks)
            
            # 随机调整初始现金（95%-105%范围内波动）
            # 减去股票价值，确保总资产在合理范围内
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            # ========== 测试模式：确定性初始化 ==========
            print("  📊 测试模式：确定性初始化")
            
            # 使用固定的初始股票持仓
            self.stocks = self.initial_stocks.astype(np.float32)
            
            # 初始化股票冷却时间
            self.stocks_cool_down = np.zeros_like(self.stocks)
            
            # 使用固定的初始现金
            self.amount = self.initial_capital

        # ==================== 计算初始资产状态 ====================
        # 计算总资产：现金 + 股票市值
        self.total_asset = self.amount + (self.stocks * price).sum()
        
        # 记录初始总资产，用于计算最终收益率
        self.initial_total_asset = self.total_asset
        
        # 初始化累计奖励
        self.gamma_reward = 0.0
        
        print(f"  💰 初始状态:")
        print(f"    - 现金: ${self.amount:,.0f}")
        print(f"    - 股票市值: ${(self.stocks * price).sum():,.0f}")
        print(f"    - 总资产: ${self.total_asset:,.0f}")
        print("  ✅ 环境重置完成")
        
        # 返回初始状态和空的信息字典（Gymnasium标准）
        return self.get_state(price), {}

    def step(self, actions):
        """
        执行一步交易动作
        
        这是强化学习环境的核心方法，接收智能体的动作，
        更新环境状态，并返回新的观察、奖励和完成标志。
        
        Args:
            actions (np.array): 标准化的动作数组，范围[-1,1]
        
        Returns:
            tuple: (新状态, 奖励, 是否结束, 是否截断, 信息字典)
        
        交易逻辑：
        1. 动作预处理：将标准化动作转换为实际交易数量
        2. 风险检查：检查市场波动度，决定是否允许交易
        3. 执行交易：按照动作执行买卖操作
        4. 更新状态：计算新的投资组合状态
        5. 计算奖励：基于资产变化计算奖励
        """
        # ==================== 动作预处理 ====================
        # 将标准化动作[-1,1]转换为实际交易股数
        # 正数表示买入，负数表示卖出
        actions = (actions * self.max_stock).astype(int)

        # ==================== 时间推进 ====================
        self.day += 1
        price = self.price_ary[self.day]    # 获取当天股票价格
        self.stocks_cool_down += 1          # 更新股票冷却时间

        # ==================== 风险控制检查 ====================
        if self.turbulence_bool[self.day] == 0:
            # ========== 正常市场条件：执行交易 ==========
            print(f"📈 第{self.day}天：正常交易模式")
            
            # 计算最小交易数量阈值
            min_action = int(self.max_stock * self.min_stock_rate)
            
            # ========== 卖出操作处理 ==========
            # 处理所有卖出动作（actions < -min_action）
            for index in np.where(actions < -min_action)[0]:
                if price[index] > 0:  # 确保价格有效
                    # 计算实际卖出数量：不能超过当前持仓
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    
                    # 更新持仓和现金
                    self.stocks[index] -= sell_num_shares
                    
                    # 卖出收入 = 股价 × 数量 × (1 - 手续费)
                    self.amount += (
                        price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    
                    # 重置该股票的冷却时间
                    self.stocks_cool_down[index] = 0
            
            # ========== 买入操作处理 ==========
            # 处理所有买入动作（actions > min_action）
            for index in np.where(actions > min_action)[0]:
                if price[index] > 0:  # 确保价格有效
                    # 计算实际买入数量：受限于可用现金
                    available_shares = self.amount // price[index]  # 可买入股数
                    buy_num_shares = min(available_shares, actions[index])
                    
                    # 更新持仓和现金
                    self.stocks[index] += buy_num_shares
                    
                    # 买入支出 = 股价 × 数量 × (1 + 手续费)
                    self.amount -= (
                        price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    
                    # 重置该股票的冷却时间
                    self.stocks_cool_down[index] = 0

        else:
            # ========== 高波动市场：强制平仓 ==========
            print(f"⚠️ 第{self.day}天：市场高波动，执行强制平仓")
            
            # 卖出所有股票，转换为现金
            # 这是重要的风险控制机制，在市场极度不稳定时保护资产
            sell_amount = (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.amount += sell_amount
            
            # 清空所有持仓
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0
            
            print(f"  💰 强制平仓收入: ${sell_amount:,.0f}")

        # ==================== 状态更新 ====================
        # 获取新的环境状态
        state = self.get_state(price)
        
        # 计算当前总资产价值
        total_asset = self.amount + (self.stocks * price).sum()
        
        # ==================== 奖励计算 ====================
        # 计算即时奖励：基于资产价值变化
        reward = (total_asset - self.total_asset) * self.reward_scaling
        
        # 更新总资产记录
        self.total_asset = total_asset

        # ==================== 累计奖励计算 ====================
        # 使用折扣因子计算长期累计奖励
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        
        # ==================== 回合结束检查 ====================
        done = self.day == self.max_step  # 检查是否到达最后一天
        
        if done:
            # 回合结束时使用累计奖励作为最终奖励
            reward = self.gamma_reward
            
            # 计算整个回合的收益率
            self.episode_return = total_asset / self.initial_total_asset
            
            print(f"🏁 交易回合结束")
            print(f"  📊 最终收益率: {(self.episode_return - 1) * 100:.2f}%")
            print(f"  💰 最终资产: ${total_asset:,.0f}")
        
        # 返回新状态、奖励、完成标志、截断标志（False）、信息字典
        return state, reward, done, False, dict()

    def get_state(self, price):
        """
        构建当前环境状态
        
        将所有相关信息组合成一个状态向量，供智能体决策使用。
        状态包含了智能体做出交易决策所需的所有信息。
        
        Args:
            price (np.array): 当前股票价格数组
        
        Returns:
            np.array: 标准化的状态向量
        
        状态组成：
        1. 账户现金（标准化）
        2. 市场波动度指标
        3. 波动度布尔标志
        4. 股票价格（标准化）
        5. 股票持仓（标准化）
        6. 股票冷却时间
        7. 技术指标
        """
        # 现金数量标准化：除以2^12=4096进行缩放
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        
        # 价格和持仓标准化比例因子
        scale = np.array(2**-6, dtype=np.float32)  # 1/64
        
        # 拼接所有状态信息
        state = np.hstack((
            amount,                           # 标准化现金
            self.turbulence_ary[self.day],   # 市场波动度
            self.turbulence_bool[self.day],  # 波动度布尔标志
            price * scale,                   # 标准化股票价格
            self.stocks * scale,             # 标准化持仓数量
            self.stocks_cool_down,           # 股票冷却时间
            self.tech_ary[self.day],         # 技术指标
        ))
        
        return state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        """
        Sigmoid函数变换
        
        对数组进行sigmoid变换，用于平滑处理波动度数据。
        这种变换可以将可能很大的波动度值压缩到合理范围内。
        
        Args:
            ary (np.array): 输入数组
            thresh (float): 阈值参数
        
        Returns:
            np.array: 变换后的数组
        
        数学原理：
        使用修正的sigmoid函数：1/(1+exp(-x*e)) - 0.5
        这确保了输出值在合理范围内，同时保持原始数据的相对关系
        """
        def sigmoid(x):
            # 修正的sigmoid函数，输出范围约为[-0.5, 0.5]
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        # 先归一化再应用sigmoid，最后还原尺度
        return sigmoid(ary / thresh) * thresh
