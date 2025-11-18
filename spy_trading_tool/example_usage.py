"""Example usage of the SPY Trading Tool.

This script demonstrates the basic usage of the SPY options trading system.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spy_trading_tool import SPYTradingTool


def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)

    # Initialize the tool
    tool = SPYTradingTool(
        ticker='SPY',
        initial_capital=10000,
        max_options=5,
        auto_save=True,
        save_dir='./example_models'
    )

    # Perform a single update
    print("\nPerforming single update...")
    result = tool.update(timeframe='1m')

    if result['status'] == 'success':
        signal = result['signal']
        print(f"\n✅ Update successful!")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.2%}")

        if signal.get('price_target'):
            targets = signal['price_target']
            print(f"\nPrice Targets:")
            print(f"  Current: ${targets.get('current', 0):.2f}")
            print(f"  Upside:  ${targets.get('upside', 0):.2f}")
            print(f"  Downside: ${targets.get('downside', 0):.2f}")
    else:
        print(f"❌ Update failed: {result.get('status')}")


def example_training():
    """Example 2: Train initial model."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Initial Model Training")
    print("="*70)

    tool = SPYTradingTool(
        initial_capital=10000,
        save_dir='./example_models'
    )

    # Train initial model
    print("\nTraining initial model on historical data...")
    print("This may take a few minutes...")

    tool.train_initial_model(
        training_days=7,  # Use 7 days for quick demo
        timesteps=1000    # Reduced for demo
    )

    print("\n✅ Training complete!")
    print("Model saved to ./example_models/")


def example_timeframe_optimization():
    """Example 3: Timeframe optimization."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Timeframe Optimization")
    print("="*70)

    tool = SPYTradingTool(
        timeframes=['1m', '5m', '15m'],  # Reduced set for demo
        initial_capital=10000
    )

    print("\nOptimizing across timeframes...")
    print("This will analyze multiple timeframes and select the best one.\n")

    results = tool.optimize_timeframes()

    if not results.empty:
        print("\n✅ Optimization complete!")
        print("\nResults:")
        print(results.to_string())
    else:
        print("❌ Optimization failed - no data available")


def example_continuous_trading():
    """Example 4: Run continuous trading (limited updates)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Continuous Trading (5 updates)")
    print("="*70)

    tool = SPYTradingTool(
        initial_capital=10000,
        auto_save=False,  # Don't save for demo
    )

    print("\nRunning 5 consecutive updates with 10-second intervals...")
    print("Press Ctrl+C to stop early.\n")

    try:
        tool.run_continuous(
            update_interval=10,  # 10 seconds for demo
            max_updates=5        # Only 5 updates
        )

        print("\n✅ Continuous trading demo complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Stopped by user")


def example_performance_tracking():
    """Example 5: Performance tracking and visualization."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Performance Tracking")
    print("="*70)

    from spy_trading_tool.performance_tracker import PerformanceTracker
    import numpy as np

    # Create tracker
    tracker = PerformanceTracker(
        initial_capital=10000,
        log_file='./example_trades.log'
    )

    # Simulate some trades
    print("\nSimulating trades...")

    tracker.log_trade({'type': 'BUY_CALL', 'pnl': 150, 'contracts': 2})
    tracker.update_equity(10150)

    tracker.log_trade({'type': 'BUY_PUT', 'pnl': -50, 'contracts': 1})
    tracker.update_equity(10100)

    tracker.log_trade({'type': 'CLOSE_CALL', 'pnl': 200, 'contracts': 2})
    tracker.update_equity(10300)

    tracker.log_trade({'type': 'CLOSE_PUT', 'pnl': 100, 'contracts': 1})
    tracker.update_equity(10400)

    # Calculate metrics
    metrics = tracker.calculate_metrics()

    print("\n✅ Performance Metrics:")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

    # Generate report
    report = tracker.generate_report()
    print(report)

    # Note: Plotting requires display - skip in headless environments
    print("\n📊 Equity curve plot saved to ./example_equity.png")
    try:
        tracker.plot_equity_curve(save_path='./example_equity.png', show=False)
    except Exception as e:
        print(f"   (Plotting skipped: {e})")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("SPY TRADING TOOL - USAGE EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates various usage scenarios.")
    print("Choose an example to run:\n")

    examples = {
        '1': ('Basic Usage (Single Update)', example_basic_usage),
        '2': ('Initial Model Training', example_training),
        '3': ('Timeframe Optimization', example_timeframe_optimization),
        '4': ('Continuous Trading (Demo)', example_continuous_trading),
        '5': ('Performance Tracking', example_performance_tracking),
        'all': ('Run All Examples', None),
    }

    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    print("\nEnter choice (1-5, or 'all'): ", end='')
    choice = input().strip().lower()

    if choice == 'all':
        # Run all examples in sequence
        for key in ['1', '5']:  # Run safe examples only
            if key in examples and examples[key][1]:
                examples[key][1]()
                print("\n" + "-"*70 + "\n")
    elif choice in examples and examples[choice][1]:
        examples[choice][1]()
    else:
        print("Invalid choice. Please run again and choose 1-5 or 'all'.")

    print("\n" + "="*70)
    print("Examples complete! Check the README.md for more information.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
