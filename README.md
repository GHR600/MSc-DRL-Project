# DDQN Trading System - Goldman Roll Strategy

A Deep Reinforcement Learning system for trading calendar spreads using Double Deep Q-Networks (DDQN) with CNN feature extraction.

## Overview

This system is designed to trade calendar spreads (price difference between front month and second month futures contracts) by learning from historical patterns, particularly around the Goldman roll period (5th-9th business day of each month).

## Features

- ??????**CNN-DDQN Architecture**: Combines 1D CNN for temporal feature extraction with Double DQN for decision making
- ?????**Flexible Position Management**: Trade in 25% increments, build positions over multiple days
- **Risk Management**: Configurable stop-loss, daily loss limits, and position constraints
- **Goldman Roll Detection**: Automatic calculation of business days and roll periods
- **Comprehensive Backtesting**: Detailed performance metrics and visualizations
- **Easily Configurable**: All parameters centralized in config.py

## File Structure

```
trading_ddqn/
├── main.py              # Main execution script
├── config.py            # Configuration settings (MODIFY THIS FOR YOUR NEEDS)
├── data_handler.py      # Data loading and preprocessing
├── environment.py       # Trading environment simulation
├── ddqn_agent.py       # DDQN agent implementation
├── networks.py         # Neural network architectures
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Quick Start

### 1. Dependencies

```bash
pip install -r requirements.txt
```

??????? - also needs a fix ### 2. Data

CSV files have 26 columns:
- A: front_open
- B: front_high  
- C: front_low
- D: front_close
- E: front_volume
- F: front_oi
- G: second_open
- H: second_high
- I: second_low
- J: second_close
- K: second_volume
- L: second_oi
- M: date (YYYY-MM-DD format)
- N: front_vol_change
- O: front_vol_change_pct
- P: front_oi_change
- Q: front_oi_change_pct
- R: calendar_spread (THIS IS WHAT THE AGENT TRADES)
- S: volume_ratio
- T: volume_ratio_change
- U: oi_ratio_pct
- V: oi_ratio_change_pct
- W: second_vol_change
- X: second_vol_change_pct
- Y: second_oi_change
- Z: second_oi_change_pct

### 3. Configure System

Edit `config.py` to match your commodity:

```python
# For Crude Oil
TICK_VALUE = 10.0
TRANSACTION_COST_PER_CONTRACT = 2.50
MAX_CONTRACTS = 8
MAX_DAILY_LOSS = 1000.0

# For Wheat/Corn  
TICK_VALUE = 12.50
# ... etc
```

### 4. Run the System

```bash
python main.py your_data.csv
```

## Configuration Parameters

### Market Parameters (config.py)
- `TICK_VALUE`: Dollar value per tick movement
- `MAX_CONTRACTS`: Maximum position size
- `TRANSACTION_COST_PER_CONTRACT`: Trading costs
- `BID_ASK_SLIPPAGE`: Slippage in ticks

### Risk Management
- `MAX_DAILY_LOSS`: Daily loss limit in dollars
- `STOP_LOSS_PERCENTAGE`: Portfolio stop-loss threshold
- `LOOKBACK_WINDOW`: Days of historical data for CNN

### Training Parameters
- `EPISODES`: Number of training episodes
- `LEARNING_RATE`: Neural network learning rate
- `BATCH_SIZE`: Training batch size
- `EPSILON_DECAY`: Exploration decay rate

### Quick Testing
- `QUICK_TEST_MODE = True`: Use smaller dataset for initial testing
- `QUICK_TEST_YEARS = 2`: Years of data for quick test
- `QUICK_TEST_EPISODES = 100`: Fewer episodes for testing

## Understanding the Output

### Training Progress
The system will print training progress showing:
- Episode rewards (higher is better)
- Validation performance 
- Training loss (should decrease)
- Epsilon decay (exploration reduction)

### Backtesting Results
Final results include:
- **Total Return**: Overall percentage return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Action Distribution
Shows how often the agent chose each action:
- Action 0-8 correspond to position changes from -100% to +100%
- Negative actions reduce long positions or increase short positions
- Positive actions increase long positions or reduce short positions

## File Outputs

The system creates several directories and saves:
- `models/`: Trained neural network checkpoints
- `results/`: Performance plots and CSV results
- `logs/`: Training logs and metrics

Key files generated:
- `best_model.pth`: Best performing model
- `backtest_results.csv`: Detailed performance metrics
- `training_curves.png`: Training progress visualization
- `backtest_analysis.png`: Comprehensive backtest plots
- `summary_report.txt`: Complete text summary

## Customization

### For Different Commodities
1. Update `TICK_VALUE` in config.py
2. Adjust `MAX_CONTRACTS` based on account size
3. Modify `TRANSACTION_COST_PER_CONTRACT` for your broker
4. Set appropriate `MAX_DAILY_LOSS` limits

### For Different Strategies
1. Modify reward function in `environment.py`
2. Adjust CNN architecture in `networks.py`
3. Change lookback window in config.py
4. Add new features in `data_handler.py`

## Performance Tips

### For Better Results
- Use more historical data (4+ years recommended)
- Increase `LOOKBACK_WINDOW` to capture more patterns
- Train for more episodes (`EPISODES = 2000+`)
- Experiment with different reward function weights

### For Faster Training
- Enable `QUICK_TEST_MODE` for initial experiments
- Use GPU if available (automatically detected)
- Reduce `BATCH_SIZE` if memory limited
- Decrease `LOOKBACK_WINDOW` for faster processing

### Common Issues

1. **"CUDA out of memory"**
   - Reduce `BATCH_SIZE` in config.py
   - Decrease `LOOKBACK_WINDOW`
   - Use CPU instead: set `DEVICE = "cpu"` in config.py

2. **Poor performance**
   - Increase training episodes
   - Check data quality and date formats
   - Verify calendar spread calculation
   - Adjust reward function weights

3. **Data loading errors**
   - Ensure CSV has exact column structure
   - Check date format (YYYY-MM-DD)
   - Remove any headers from CSV
   - Verify no missing values in critical columns

### Getting Help

1. Check the summary report for training diagnostics
2. Review training curves for convergence issues
3. Examine action distribution for reasonable behavior
4. Verify backtesting metrics align with expectations

## Advanced Usage

### Loading Pretrained Models
```python
from ddqn_agent import DDQNTradingAgent

agent = DDQNTradingAgent(...)
agent.load_agent("models/best_model.pth")
```

### Custom Evaluation
```python
# Run additional backtests
results = agent.backtest(custom_test_env)
```

### Hyperparameter Tuning
Modify config.py and run multiple experiments with different:
- Learning rates
- Network architectures  
- Reward function weights
- Risk parameters

## Next Steps

1. Start with `QUICK_TEST_MODE = True` to verify everything works
2. Run full training with your complete dataset
3. Analyze results and adjust parameters
4. Implement live trading integration (not included)
5. Add additional features or modify strategy logic

