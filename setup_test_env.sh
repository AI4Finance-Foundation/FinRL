#!/bin/bash
# Setup script for running FinRL tests
# This script creates a clean test environment and installs all dependencies

set -e  # Exit on error

echo "================================================"
echo "FinRL Test Environment Setup"
echo "================================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    echo "  Found Python 3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    echo "  Found Python 3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
    echo "  Found Python 3.10"
else
    echo "  ✗ ERROR: Python 3.10, 3.11, or 3.12 required"
    echo "  Please install Python 3.12: brew install python@3.12"
    exit 1
fi

# Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
rm -rf finrl_test_env  # Clean up any existing environment
$PYTHON_CMD -m venv finrl_test_env
echo "  Virtual environment created"

# Activate environment
source finrl_test_env/bin/activate

# Upgrade pip
echo ""
echo "✓ Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

# Install core dependencies first
echo ""
echo "✓ Installing core dependencies..."
pip install pytest numpy pandas --quiet

# Install specific versions to avoid conflicts
echo ""
echo "✓ Installing compatible dependency versions..."
pip install 'urllib3==1.24.3' --quiet  # Exact version compatible with alpaca-trade-api
pip install 'six' --quiet  # Required by urllib3 packages
pip install alpha-vantage --quiet  # Required by alpaca-trade-api
pip install alpaca-trade-api==0.48 --quiet  # Use older stable version

# Install remaining dependencies
echo ""
echo "✓ Installing remaining dependencies..."
pip install gymnasium yfinance matplotlib stockstats stable-baselines3 pandas-market-calendars --quiet

echo ""
echo "================================================"
echo "✓ Setup complete!"
echo "================================================"
echo ""
echo "To run the tests:"
echo ""
echo "  source finrl_test_env/bin/activate"
echo "  pytest unit_tests/environments/test_stocktrading.py -v"
echo ""
echo "To run a specific test class:"
echo ""
echo "  pytest unit_tests/environments/test_stocktrading.py::TestEnvironmentInitialization -v"
echo ""
echo "To deactivate the environment when done:"
echo ""
echo "  deactivate"
echo ""
