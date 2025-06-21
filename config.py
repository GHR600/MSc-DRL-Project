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
TRANSACTION_COST_PER_CONTRACT = 1.50  # Transaction cost per contract

# Slippage settings
BID_ASK_SLIPPAGE = 1  # Slippage in ticks per contract
SLIPPAGE_ENABLED = True

# =============================================================================
# RISK MANAGEMENT PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

MAX_DAILY_LOSS = 500.0  # Maximum daily loss in dollars
STOP_LOSS_ENABLED = True
STOP_LOSS_PERCENTAGE = 0.1  # 2% stop loss on total capital

# Position limits
MIN_POSITION_SIZE = 0  # Minimum position size in contracts
#POSITION_SIZE_INCREMENT = 1  # Position size increments

# =============================================================================
# DATA PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

# Lookback window for LSTM
LOOKBACK_WINDOW = 40  # Days to look back (adjustable: 20-252)
MIN_LOOKBACK_FOR_TRAINING = 40  # Minimum data needed before training starts

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature columns (based on your CSV structure)
FEATURE_COLUMNS = [
    'PX_OPEN1', 'PX_HIGH1', 'PX_LOW1', 'PX_LAST1', 'PX_VOLUME1', 'OPEN_INT1',
    'PX_OPEN2', 'PX_HIGH2', 'PX_LOW2', 'PX_LAST2', 'PX_VOLUME2', 'OPEN_INT2',
    'VOL Change1', 'Vol Change %1', 'OI Change1', 'OI Change %1',
    'CALENDAR', 'Vol Ratio', 'Vol Ratio Change', 'OI Ratio', 'OI Ratio Change',
    'VOL Change2', 'Vol Change %2', 'OI Change2', 'OI Change %2'
]

TARGET_COLUMN = 'CALENDAR'  # What the agent trades
DATE_COLUMN = 'Dates'       # Your date column name

# =============================================================================
# GOLDMAN ROLL PATTERN PARAMETERS
# =============================================================================

# Goldman roll typically occurs 5th-9th business day of month
GOLDMAN_ROLL_START_DAY = 5  # 5th business day
GOLDMAN_ROLL_END_DAY = 9    # 9th business day
GOLDMAN_ROLL_WINDOW = 15    # Total window around roll (5 before, 5 during, 5 after)

# =============================================================================
# LSTM PARAMETERS
# =============================================================================

# LSTM Architecture
LSTM_HIDDEN_SIZE = 200        # Hidden state size
LSTM_NUM_LAYERS = 3           # Number of LSTM layers (2-3 typical)
LSTM_DROPOUT = 0.3            # Dropout between LSTM layers
LSTM_BIDIRECTIONAL = False    # Whether to use bidirectional LSTM
LSTM_PROCESSING_DIM = 256     # Size of post-LSTM processing layers

HIDDEN_DIMS = [512, 256, 128]  # Sizes of hidden layers in DQN head

# =============================================================================
# DDQN TRAINING PARAMETERS
# =============================================================================

LEARNING_RATE = 0.0001        # Adam learning rate
GAMMA = 0.99                  # Discount factor for future rewards
TAU = 0.005                   # Soft update rate for target network
BATCH_SIZE = 16               # Training batch size

# =============================================================================
# ACTION SPACE PARAMETERS
# =============================================================================

# Position changes (as percentages of max position)
POSITION_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]
# Negative = reduce long/increase short, Positive = increase long/reduce short

# Action mapping
ACTION_SPACE_SIZE = len(POSITION_ACTIONS)

# =============================================================================
# REWARD FUNCTION PARAMETERS (EASILY ADJUSTABLE)
# =============================================================================

# Reward function weights
REWARD_PNL_WEIGHT = 0.8        # Weight for P&L component
REWARD_WINRATE_WEIGHT = 0.1    # Weight for win rate component  
REWARD_RISK_WEIGHT = 0.1       # Weight for risk penalty component

# Risk penalty parameters
DRAWDOWN_PENALTY_THRESHOLD = 0.1  # 5% drawdown threshold
POSITION_SIZE_PENALTY = 0.01       # Penalty for large positions

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Training schedule
EPISODES = 10000  # Number of complete runs through data
MEMORY_SIZE = 4000  # How many past experiences the agent remembers.
UPDATE_FREQUENCY = 3  # How often the network trains
TARGET_UPDATE_FREQUENCY = 200

# Exploration parameters
EPSILON_START = 1.0     # Starting exploration (100%)
EPSILON_END = 0.01      # Final exploration (1%)
EPSILON_DECAY = 0.999   #  How fast to reduce exploration

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
SAVE_FREQUENCY = 250  # Episodes

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
INITIAL_CAPITAL = 10000.0 

# Performance metrics to track
TRACK_METRICS = [
    'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
    'profit_factor', 'total_trades', 'avg_trade_duration'
]

# =============================================================================
# QUICK TEST PARAMETERS (FOR INITIAL FAMILIARIZATION)
# =============================================================================

# Use smaller dataset for initial testing
QUICK_TEST_MODE = False
QUICK_TEST_YEARS = 3  # Use 2 years of data (1 train, 1 test)
QUICK_TEST_EPISODES = 1000  # Fewer episodes for quick testing

print(f"Configuration loaded successfully!")
print(f"Device: {DEVICE}")
print(f"Lookback window: {LOOKBACK_WINDOW} days")
print(f"Max contracts: {MAX_CONTRACTS}")
print(f"Tick value: ${TICK_VALUE}")
print(f"Quick test mode: {QUICK_TEST_MODE}")
