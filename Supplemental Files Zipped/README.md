# DDQN Trading System - Goldman Roll Strategy

A deep reinforcement learning implementation using Double Deep Q-Networks (DDQN) for automated commodity futures trading based on Goldman Roll patterns.

## Overview

This project implements a sophisticated DDQN trading agent that identifies and exploits roll patterns in commodity futures markets. The system combines LSTM networks for temporal pattern recognition with fully connected layers for trading decision-making, incorporating realistic transaction costs and risk management.

## Project Structure

```
Supplemental Files/
├── source_code.ipynb          # Complete Jupyter notebook implementation
├── StandardGoldmanRollStrat.py # Benchmark Goldman Roll strategy
├── Detailed DDQN Test Run Results # Results gathered from training and backtest runs 
|
├── CSVfiles/                  # Data directory
│   ├── CL12 - Sheet1.csv     # Brent Oil futures data
│   ├── CO12 - Sheet1.csv     # Crude Oil futures data
│   ├── GC12 - Sheet1.csv     # Gold futures data
│   ├── LC12 - Sheet1.csv     # Live Cattle futures data
│   ├── SB12 - Sheet1.csv     # Sugar #11 futures data
│   └── SI12 - Sheet1.csv     # Silver futures data
```


## Data Format

The CSV files contain commodity futures data with the following structure:
- **Dates**: Trading dates (DD/MM/YYYY format)
- **Front Contract**: Price, volume, and open interest for front month
- **Back Contract**: Price, volume, and open interest for back month
- **Calendar Spread**: Price differential between contracts
- **Expected format**: 25 features total with 3 header rows to skip

###  Commodities
- **CL12**: Brent Oil futures
- **CO12**: Crude Oil futures  
- **GC12**: Gold futures
- **LC12**: Live Cattle futures
- **SB12**: Sugar #11 futures
- **SI12**: Silver futures

## Requirements

```bash
# Core dependencies
numpy
pandas
torch
matplotlib
seaborn

scikit-learn  # For data preprocessing
pytorch       # Deep learning framework
```

## Installation

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Ensure your CSV data files are in the `CSVfiles/` directory
4. Update file paths in the main scripts if necessary

## Usage

### Single File Execution

```bash
# Run on specific commodity
python main.py path/to/CL12\ -\ Sheet1.csv

# Or modify the csv_file_path variable in main.py
python main.py
```

### Batch Processing (Jupyter Notebook)

The `source_code.ipynb` notebook provides comprehensive batch processing capabilities:

```python
# Run 5 statistical validation experiments across all commodities
csv_files = [
    "/notebooks/CL12 - Sheet1.csv",    # Brent Oil
    "/notebooks/CO12 - Sheet1.csv",    # Crude Oil
    "/notebooks/GC12 - Sheet1.csv",    # Gold
    "/notebooks/LC12 - Sheet1.csv",    # Live Cattle
    "/notebooks/SB12 - Sheet1.csv",    # Sugar #11
    "/notebooks/SI12 - Sheet1.csv",    # Silver
]

all_results, all_summaries = run_multiple_csvs_and_seeds(csv_files)
```

### Benchmark Comparison

```bash
# Run traditional Goldman Roll strategy for comparison
python StandardGoldmanRollStrat.py
```

## Configuration

Key parameters can be modified in the configuration module

- **Trading Parameters**: Initial capital, max contracts, transaction costs
- **DDQN Parameters**: Learning rate, epsilon decay, replay buffer size
- **Network Architecture**: LSTM layers, hidden dimensions, dropout rates
- **Data Processing**: Lookback window, train/validation/test splits
- **Risk Management**: Stop loss, maximum daily loss limits

**Note**: If no separate config file exists, parameters may be defined as constants within the main implementation files.

## Key Features

### Deep Reinforcement Learning
- **DDQN Architecture**: Combines LSTM for temporal pattern recognition with fully connected layers for decision-making
- **Action Space**: Discrete position changes (-50%, -25%, 0%, +25%, +50%)
- **State Space**: Market features (prices, volumes, spreads) + trading state (positions, P&L, drawdown)

### Risk Management
- Realistic transaction costs and slippage modeling
- Position size limits and stop-loss mechanisms
- Maximum daily loss protection
- Comprehensive drawdown monitoring

### Statistical Validation
- Multiple random seed testing (5 runs per commodity)
- Confidence interval calculation
- Performance comparison against benchmark strategies
- Comprehensive performance metrics (Sharpe ratio, win rate, profit factor)

## Expected Results

Based on the research findings, you should expect:

### DDQN Performance
- **Win Rates**: 42-60% across commodities (significantly better than benchmark)
- **Returns**: Generally negative due to transaction costs (-5% to -30%)
- **Best Performers**: Gold (60.2% win rate) and Live Cattle (58.9% win rate)
- **Pattern Recognition**: Strong ability to identify profitable trades pre-cost

### Benchmark Strategy
- **Traditional Goldman Roll**: 0% win rate in energy/metals markets (2022-2025 period)
- **Market Evolution**: Historical arbitrage opportunities largely eliminated

## Important Notes

### Data Requirements
- CSV files must contain exactly 25 features with 3 header rows
- Date format should be DD/MM/YYYY
- Missing files will be automatically skipped with error messages

### Performance Expectations
- The project demonstrates that transaction costs eliminate profitability despite pattern recognition capability
- The system is designed for academic research rather than live trading
- Results validate the obsolescence of traditional roll strategies in modern markets

### Computational Requirements
- GPU recommended for faster training (automatically detected)
- Each full experiment (5 seeds × 6 commodities = 30 runs) may take several hours
- Memory requirements scale with lookback window and batch size

## Troubleshooting

### Common Issues
1. **File Not Found**: Ensure CSV files are in correct directory and paths match
2. **Memory Errors**: Reduce batch size or lookback window in config
3. **Training Instability**: Check learning rate and network architecture parameters
4. **Poor Performance**: Verify data quality and feature preprocessing

### Debug Mode
Enable quick testing mode in config for faster debugging:
- Reduced episodes for training
- Smaller data subsets
- Simplified plotting

## Research Context

This implementation is based on academic research comparing modern DRL approaches to traditional Goldman Roll strategies. The key finding is that while AI can identify profitable patterns, transaction costs in realistic trading scenarios eliminate net profitability - demonstrating the evolution of market efficiency since the original Goldman Roll strategies were profitable (pre-2012).

## License

This project is intended for academic and research purposes. Please ensure compliance with relevant financial regulations if adapting for commercial use.