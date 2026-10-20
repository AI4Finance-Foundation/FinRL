## FinRL Stock Trading 2026 Tutorial

### Step 1: Clone the Repository

```bash
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install FinRL

```bash
pip install -e .
```

### Step 4: Run the Scripts

**1. Data Download & Preprocessing**

```bash
python examples/FinRL_StockTrading_2026_1_data.py
```

This script downloads the union of Dow constituents that were members during 2014–2025 plus the roster snapshot used for the 2026-01-01 to 2026-03-20 trading period. It adds technical indicators (MACD, RSI, etc.), VIX, and turbulence, then saves `train_data.csv` and `trade_data.csv` with a date-specific `is_member` field.

The training data records historical membership as of each trading date. The training and backtest environments keep a fixed action dimension by using the union of constituents, expose the membership mask to the policy, reject actions for inactive stocks, and liquidate holdings when a constituent leaves the index. Missing Yahoo Finance prices for active members stop data preparation with an error instead of silently dropping the constituent.

**2. Train DRL Agents**

```bash
python examples/FinRL_StockTrading_2026_2_train.py
```

This script trains 5 DRL agents (A2C, DDPG, PPO, TD3, SAC) using Stable Baselines 3 on the training data. Trained models are saved to the `trained_models/` directory.

**Key Hyperparameters:**

| Parameter | Description | Default in Script |
|-----------|-------------|-------------------|
| `total_timesteps` | Total number of environment interactions for training. **This is the most important parameter** — higher values give the agent more experience to learn from, leading to better trading performance. Start with a small value (e.g., 1,000) for a quick test, then increase (e.g., 20,000–200,000) for serious training. | 20,000 |
| `learning_rate` | Controls how much the model weights are updated at each step. Too high causes instability; too low causes slow learning. | 0.001 |
| `batch_size` | Number of samples used per gradient update. Larger batches give more stable updates but require more memory. | 100 |
| `buffer_size` | Size of the replay buffer (for off-policy algorithms like DDPG, TD3, SAC). Stores past experiences for the agent to learn from. Larger buffers retain more diverse experiences. | 1,000,000 |

**3. Backtest**

```bash
python examples/FinRL_StockTrading_2026_3_Backtest.py
```

This script loads the trained agents and compares their performance against a point-in-time equal-weight basket of the active Dow members and the DJIA index. Results are printed to the console and a plot is saved as `backtest_result.png`.
