import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimization.backtest import BacktestEngine

if __name__ == "__main__":
    engine = BacktestEngine()
    engine.run()
