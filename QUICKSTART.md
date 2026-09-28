# Quick Start: Running the Tests

## TL;DR - Run This

```bash
# 1. Setup environment (one time)
./setup_test_env.sh

# 2. Run tests
source finrl_test_env/bin/activate
pytest unit_tests/environments/test_stocktrading.py -v
```

---

## Known Issue: Dependency Conflicts

⚠️ **The current repository has a dependency conflict** with `alpaca-trade-api` that prevents tests from running.

**The problem:** The repo imports `alpaca-trade-api` at the top level (`finrl/__init__.py` → `finrl/trade.py` → `env_stock_papertrading.py`), but this library has incompatible dependencies with Python 3.12.

**Temporary workaround to run tests:**

### Option 1: Comment out problematic import (Quick fix)

```bash
# Temporarily disable the import
sed -i.bak 's/from finrl.trade import trade/# from finrl.trade import trade/' finrl/__init__.py

# Now run tests
source finrl_test_env/bin/activate
pytest unit_tests/environments/test_stocktrading.py -v

# Restore the file when done
mv finrl/__init__.py.bak finrl/__init__.py
```

### Option 2: Fix the import structure (Better solution)

The root cause is that `finrl/__init__.py` eagerly imports everything, including paper trading code that most users don't need.

**Recommended fix for the maintainers:**

```python
# finrl/__init__.py - Use lazy imports
from finrl.train import train
from finrl.test import test

# Don't import trade at top level - let users import explicitly if needed
# from finrl.trade import trade  # Remove this line
```

This way, users who need paper trading can do:
```python
from finrl.trade import trade  # Explicit import only when needed
```

---

## What the Tests Cover

The test suite in `unit_tests/environments/test_stocktrading.py` includes:

- **31 test functions** across 9 test classes
- **Environment initialization** - validates setup, state spaces, action spaces
- **Reset functionality** - ensures clean episode starts
- **Step function** - core trading logic, buy/sell actions
- **Trading costs** - commission and fee calculations
- **Turbulence handling** - risk management features
- **Reward calculation** - profit/loss tracking
- **Memory tracking** - episode history
- **Edge cases** - extreme values, boundary conditions
- **Determinism** - reproducible results with seeding

---

## Running Specific Tests

```bash
# Activate environment
source finrl_test_env/bin/activate

# Run all tests
pytest unit_tests/environments/test_stocktrading.py -v

# Run a specific test class
pytest unit_tests/environments/test_stocktrading.py::TestEnvironmentInitialization -v

# Run a single test
pytest unit_tests/environments/test_stocktrading.py::TestEnvironmentStep::test_buy_action_decreases_cash -v

# Run with coverage
pytest unit_tests/environments/test_stocktrading.py --cov=finrl.meta.env_stock_trading.env_stocktrading --cov-report=html
```

---

## Manual Verification (No Dependencies Required)

If you just want to verify the bug fixes without running tests:

### 1. Exception Handling Fix

```bash
# Should show NO bare "except:" - all should specify exception types
grep -n "except:" finrl/meta/paper_trading/alpaca.py
grep -n "except:" finrl/trade.py
```

### 2. Import Fix

```bash
# Should compile without errors
python3 -m py_compile finrl/agents/stablebaselines3/hyperparams_opt.py
echo "✓ No ImportError"
```

### 3. Pandas API Fix

```bash
# Should show modern syntax: .ffill().bfill()
grep "fillna" finrl/plot.py
```

### 4. Gymnasium Migration

```bash
# Should find NO old gym imports
grep -rn "^import gym$" finrl/meta/env_stock_trading/
echo "✓ All using gymnasium"
```

---

## For Maintainers: Fixing the Dependency Issue

To permanently fix this issue, consider:

1. **Lazy imports** - Don't import everything in `__init__.py`
2. **Optional dependencies** - Make `alpaca-trade-api` optional:
   ```python
   # requirements.txt → requirements-trading.txt
   alpaca-trade-api>=2.0  # Move to separate file
   ```
3. **Conditional imports** - Check if dependencies exist before importing
4. **Update alpaca library** - Use `alpaca-py` instead of old `alpaca-trade-api`

---

## Help & Support

- **Full documentation**: See `TESTING_README.md`
- **Test run notes**: See `TEST_RUN_NOTES.md`
- **Issues**: Report at https://github.com/AI4Finance-Foundation/FinRL/issues
