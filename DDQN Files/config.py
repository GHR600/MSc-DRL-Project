"""
Configuration file for DDQN Trading System
All adjustable parameters are centralized here
"""

import torch

# =============================================================================
# MARKET PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

# Commodity-specific settings
TICK_VALUE = 10.0  # $10 for crude, $12.50 for wheat/corn, $25 for LS Gasoil
MAX_CONTRACTS = 8  # Maximum position size
TRANSACTION_COST_PER_CONTRACT = 2.50  # Transaction cost per contract

# Slippage settings
BID_ASK_SLIPPAGE = 0.5  # Slippage in ticks per contract
SLIPPAGE_ENABLED = True

# =============================================================================
# RISK MANAGEMENT PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

MAX_DAILY_LOSS = 500.0  # Maximum daily loss in dollars
STOP_LOSS_ENABLED = True
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss on total capital

# Position limits
MIN_POSITION_SIZE = 1  # Minimum position size in contracts
POSITION_SIZE_INCREMENT = 1  # Position size increments

# =============================================================================
# DATA PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

# Lookback window for CNN
LOOKBACK_WINDOW = 90  # Days to look back (adjustable: 20-252)
MIN_LOOKBACK_FOR_TRAINING = 120  # Minimum data needed before training starts

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature columns (based on your CSV structure)
FEATURE_COLUMNS = [
    'front_open', 'front_high', 'front_low', 'front_close', 'front_volume', 'front_oi',
    'second_open', 'second_high', 'second_low', 'second_close', 'second_volume', 'second_oi',
    'front_vol_change', 'front_vol_change_pct', 'front_oi_change', 'front_oi_change_pct',
    'calendar_spread', 'volume_ratio', 'volume_ratio_change', 'oi_ratio_pct', 
    'oi_ratio_change_pct', 'second_vol_change', 'second_vol_change_pct', 
    'second_oi_change', 'second_oi_change_pct'
]

TARGET_COLUMN = 'calendar_spread'  # Primary trading target
DATE_COLUMN = 'date'

# =============================================================================
# GOLDMAN ROLL PATTERN PARAMETERS
# =============================================================================

# Goldman roll typically occurs 5th-9th business day of month
GOLDMAN_ROLL_START_DAY = 5  # 5th business day
GOLDMAN_ROLL_END_DAY = 9    # 9th business day
GOLDMAN_ROLL_WINDOW = 15    # Total window around roll (5 before, 5 during, 5 after)

# =============================================================================
# NEURAL NETWORK PARAMETERS
# =============================================================================

# CNN Architecture
CNN_CHANNELS = [32, 64, 128]  # Channel progression
CNN_KERNEL_SIZES = [3, 3, 3]  # Kernel sizes for each layer
CNN_DROPOUT = 0.2

# DDQN Architecture
HIDDEN_DIMS = [512, 256, 128]  # Hidden layer dimensions
LEARNING_RATE = 0.001
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
TAU = 0.005   # Soft update parameter

# =============================================================================
# ACTION SPACE PARAMETERS
# =============================================================================

# Position changes (as percentages of max position)
POSITION_ACTIONS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
# Negative = reduce long/increase short, Positive = increase long/reduce short

# Action mapping
ACTION_SPACE_SIZE = len(POSITION_ACTIONS)

# =============================================================================
# REWARD FUNCTION PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

# Reward function weights
REWARD_PNL_WEIGHT = 0.6        # Weight for P&L component
REWARD_WINRATE_WEIGHT = 0.3    # Weight for win rate component  
REWARD_RISK_WEIGHT = 0.1       # Weight for risk penalty component

# Risk penalty parameters
DRAWDOWN_PENALTY_THRESHOLD = 0.05  # 5% drawdown threshold
POSITION_SIZE_PENALTY = 0.01       # Penalty for large positions

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Training schedule
EPISODES = 1000
MEMORY_SIZE = 50000
UPDATE_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 1000

# Exploration parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Early stopping
PATIENCE = 50  # Episodes without improvement before stopping
MIN_IMPROVEMENT = 0.001

# =============================================================================
# DEVICE AND PERFORMANCE
# =============================================================================

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # For data loading

# =============================================================================
# LOGGING AND VISUALIZATION
# =============================================================================

# Logging frequency
LOG_FREQUENCY = 10  # Episodes
SAVE_FREQUENCY = 100  # Episodes

# Paths
MODEL_SAVE_PATH = "models/"
LOG_SAVE_PATH = "logs/"
RESULTS_SAVE_PATH = "results/"

# Visualization parameters
PLOT_TRAINING_CURVES = True
PLOT_TRADING_RESULTS = True
SAVE_PLOTS = True

# =============================================================================
# BACKTESTING PARAMETERS
# =============================================================================

# Initial capital
INITIAL_CAPITAL = 100000.0  # $100,000 starting capital

# Performance metrics to track
TRACK_METRICS = [
    'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
    'profit_factor', 'total_trades', 'avg_trade_duration'
]

# =============================================================================
# QUICK TEST PARAMETERS (FOR INITIAL FAMILIARIZATION)
# =============================================================================

# Use smaller dataset for initial testing
QUICK_TEST_MODE = True
QUICK_TEST_YEARS = 2  # Use 2 years of data (1 train, 1 test)
QUICK_TEST_EPISODES = 100  # Fewer episodes for quick testing

print(f"Configuration loaded successfully!")
print(f"Device: {DEVICE}")
print(f"Lookback window: {LOOKBACK_WINDOW} days")
print(f"Max contracts: {MAX_CONTRACTS}")
print(f"Tick value: ${TICK_VALUE}")
print(f"Quick test mode: {QUICK_TEST_MODE}")
