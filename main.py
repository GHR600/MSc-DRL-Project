"""
Main execution script for DDQN Trading System
Coordinates data loading, training, and backtesting
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import config
from data_handler import TradingDataHandler
from environment import TradingEnvironmentWrapper
from ddqn_agent import DDQNTradingAgent, create_agent_from_env
from networks import save_model, load_model

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_directories():
    """Create necessary directories for saving models and results"""
    directories = [config.MODEL_SAVE_PATH, config.LOG_SAVE_PATH, config.RESULTS_SAVE_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully")

def load_and_prepare_data(csv_path: str):
    """
    Load and prepare data for training
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        data_splits: Dictionary containing train/val/test splits
        data_handler: Data handler instance
    """
    print("Loading and preparing data...")
    
    # Initialize data handler
    data_handler = TradingDataHandler(csv_path)
    
    # Load and preprocess data
    raw_data = data_handler.load_data()
    processed_data = data_handler.preprocess_data()
    
    # Create sequences for CNN
    sequences, targets, dates = data_handler.create_sequences(
        processed_data, 
        lookback=config.LOOKBACK_WINDOW
    )
    
    # Split data
    data_splits = data_handler.split_data(
        sequences, targets, dates, 
        quick_test=config.QUICK_TEST_MODE
    )
    
    # Print data information
    feature_info = data_handler.get_feature_info()
    print(f"\nData Information:")
    print(f"  Features: {feature_info['n_features']}")
    print(f"  Samples: {feature_info['n_samples']}")
    print(f"  Date range: {feature_info['date_range'][0]} to {feature_info['date_range'][1]}")
    print(f"  Lookback window: {config.LOOKBACK_WINDOW} days")
    
    return data_splits, data_handler

def create_environments(data_splits):
    """
    Create trading environments for train/validation/test
    
    Args:
        data_splits: Data splits dictionary
        
    Returns:
        Environment wrapper and individual environments
    """
    print("Creating trading environments...")
    
    # Create environment wrapper
    env_wrapper = TradingEnvironmentWrapper(data_splits)
    
    # Create individual environments
    train_env = env_wrapper.create_env('train')
    val_env = env_wrapper.create_env('val')
    test_env = env_wrapper.create_env('test')
    
    # Print environment information
    env_info = env_wrapper.get_env_info()
    for split, info in env_info.items():
        print(f"\n{split.upper()} Environment:")
        print(f"  Samples: {info['n_samples']}")
        print(f"  State size: {info['state_size']}")
        print(f"  Action space: {info['action_space_size']}")
        print(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
    
    return env_wrapper, train_env, val_env, test_env

def train_agent(train_env, val_env):
    """
    Train the DDQN agent
    
    Args:
        train_env: Training environment
        val_env: Validation environment
        
    Returns:
        Trained agent and training history
    """
    print("\nInitializing DDQN agent...")
    
    # Create agent
    agent = create_agent_from_env(train_env)
    
    # Determine number of episodes
    num_episodes = config.QUICK_TEST_EPISODES if config.QUICK_TEST_MODE else config.EPISODES
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Quick test mode: {config.QUICK_TEST_MODE}")
    
    # Train agent
    training_history = agent.train(train_env, val_env, num_episodes)
    
    # Plot training curves
    if config.PLOT_TRAINING_CURVES:
        save_path = f"{config.RESULTS_SAVE_PATH}training_curves.png" if config.SAVE_PLOTS else None
        agent.plot_training_curves(training_history, save_path)
    
    # Save final agent
    agent.save_agent(
        f"{config.MODEL_SAVE_PATH}final_agent.pth",
        {'training_complete': True, 'timestamp': datetime.now().isoformat()}
    )
    
    return agent, training_history

def backtest_agent(agent, test_env):
    """
    Backtest the trained agent
    
    Args:
        agent: Trained DDQN agent
        test_env: Test environment
        
    Returns:
        Backtesting results
    """
    print("\nStarting backtesting...")
    
    # Load best model for backtesting
    try:
        agent.load_agent(f"{config.MODEL_SAVE_PATH}best_model.pth")
        print("Loaded best model for backtesting")
    except FileNotFoundError:
        print("Best model not found, using current model")
    
    # Run backtest
    backtest_results = agent.backtest(test_env)
    
    # Save results
    results_df = pd.DataFrame([backtest_results['performance_metrics']])
    results_df.to_csv(f"{config.RESULTS_SAVE_PATH}backtest_results.csv", index=False)
    
    return backtest_results

def plot_backtest_results(test_env, backtest_results):
    """
    Plot comprehensive backtesting results
    
    Args:
        test_env: Test environment used for backtesting
        backtest_results: Results from backtesting
    """
    if not config.PLOT_TRADING_RESULTS:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Equity curve
    equity_curve = test_env.equity_curve
    dates = test_env.data_dates[:len(equity_curve)]
    
    axes[0, 0].plot(dates, equity_curve, linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Equity Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Equity ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    
    axes[0, 1].fill_between(dates, drawdown, 0, alpha=0.7, color='red')
    axes[0, 1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Action distribution
    action_dist = backtest_results['action_distribution']
    actions = list(action_dist.keys())
    counts = list(action_dist.values())
    action_labels = [f"Action {a}\n({config.POSITION_ACTIONS[a]:+.0%})" for a in actions]
    
    axes[1, 0].bar(action_labels, counts, alpha=0.8)
    axes[1, 0].set_title('Action Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics
    performance = backtest_results['performance_metrics']
    metrics_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
    metrics_values = [
        performance.get('total_return', 0),
        performance.get('sharpe_ratio', 0),
        performance.get('max_drawdown', 0),
        performance.get('win_rate', 0),
        performance.get('profit_factor', 0)
    ]
    
    # Format values for display
    formatted_values = [
        f"{metrics_values[0]:.1%}",
        f"{metrics_values[1]:.2f}",
        f"{metrics_values[2]:.1%}",
        f"{metrics_values[3]:.1%}",
        f"{metrics_values[4]:.2f}"
    ]
    
    # Create bar chart
    bars = axes[1, 1].bar(metrics_names, metrics_values, alpha=0.8)
    axes[1, 1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, formatted_val in zip(bars, formatted_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(metrics_values)*0.01,
                       formatted_val, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if config.SAVE_PLOTS:
        plt.savefig(f"{config.RESULTS_SAVE_PATH}backtest_analysis.png", 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def save_summary_report(training_history, backtest_results, data_handler):
    """
    Save a comprehensive summary report
    
    Args:
        training_history: Training metrics
        backtest_results: Backtesting results
        data_handler: Data handler with configuration info
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
DDQN TRADING SYSTEM - SUMMARY REPORT
Generated: {timestamp}

=== CONFIGURATION ===
Commodity: {config.TICK_VALUE} per tick
Max Contracts: {config.MAX_CONTRACTS}
Lookback Window: {config.LOOKBACK_WINDOW} days
Initial Capital: ${config.INITIAL_CAPITAL:,.2f}
Quick Test Mode: {config.QUICK_TEST_MODE}

=== DATA INFORMATION ===
Total Samples: {len(data_handler.processed_data)}
Features: {len(config.FEATURE_COLUMNS)}
Date Range: {data_handler.processed_data['date'].min()} to {data_handler.processed_data['date'].max()}

=== TRAINING RESULTS ===
Episodes Trained: {len(training_history['train_rewards'])}
Final Training Reward: {training_history['train_rewards'][-1]:.4f}
Final Validation Reward: {training_history['val_rewards'][-1]:.4f}
Best Validation Performance: {max([p.get('total_return', 0) for p in training_history['val_performance']]):.2%}

=== BACKTESTING RESULTS ===
Total Return: {backtest_results['performance_metrics'].get('total_return', 0):.2%}
Sharpe Ratio: {backtest_results['performance_metrics'].get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {backtest_results['performance_metrics'].get('max_drawdown', 0):.2%}
Win Rate: {backtest_results['performance_metrics'].get('win_rate', 0):.1%}
Profit Factor: {backtest_results['performance_metrics'].get('profit_factor', 0):.2f}
Total Trades: {backtest_results['performance_metrics'].get('total_trades', 0)}
Final Equity: ${backtest_results['performance_metrics'].get('final_equity', 0):,.2f}

=== RISK METRICS ===
Transaction Costs: ${config.TRANSACTION_COST_PER_CONTRACT} per contract
Slippage: {config.BID_ASK_SLIPPAGE} ticks per contract
Max Daily Loss Limit: ${config.MAX_DAILY_LOSS:,.2f}
Stop Loss: {config.STOP_LOSS_PERCENTAGE:.1%}

=== ACTION ANALYSIS ===
Most Used Action: {max(backtest_results['action_distribution'], key=backtest_results['action_distribution'].get)}
Action Distribution:
"""
    
    for action, count in backtest_results['action_distribution'].items():
        percentage = count / sum(backtest_results['action_distribution'].values()) * 100
        action_desc = config.POSITION_ACTIONS[action]
        report += f"  Action {action} ({action_desc:+.0%}): {count} times ({percentage:.1f}%)\n"
    
    # Save report
    with open(f"{config.RESULTS_SAVE_PATH}summary_report.txt", 'w') as f:
        f.write(report)
    
    print("\nSummary report saved to results/summary_report.txt")
    print(report)

def main(csv_path: str = None):
    """
    Main execution function
    
    Args:
        csv_path: Path to CSV data file
    """
    print("="*60)
    print("DDQN TRADING SYSTEM - GOLDMAN ROLL STRATEGY")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load and prepare data
    if csv_path is None:
        print("Please provide the path to your CSV file")
        csv_path = input("Enter CSV file path: ")
    
    try:
        data_splits, data_handler = load_and_prepare_data(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create environments
    env_wrapper, train_env, val_env, test_env = create_environments(data_splits)
    
    # Train agent
    try:
        agent, training_history = train_agent(train_env, val_env)
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Backtest agent
    try:
        backtest_results = backtest_agent(agent, test_env)
    except Exception as e:
        print(f"Error during backtesting: {e}")
        return
    
    # Plot results
    try:
        plot_backtest_results(test_env, backtest_results)
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    # Save summary report
    try:
        save_summary_report(training_history, backtest_results, data_handler)
    except Exception as e:
        print(f"Error saving report: {e}")
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return agent, backtest_results, training_history

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        main(csv_file_path)
    else:
        # For testing - you would replace this with your actual file path
        
        csv_file_path  = "/home/k24104079/MSc code/MSc-DRL-Project/CSVfiles/SB12 - Sheet1.csv"
        
        
        
        print(f"No CSV file provided. Please run with: python main.py your_data.csv")
        print(f"Or modify the csv_file_path variable in main() and uncomment the line below.")
        main(csv_file_path)  # Uncomment this line and provide your CSV path


