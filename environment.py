"""
Trading Environment for DDQN Agent
Simulates the trading of calendar spreads with realistic constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import config

# Define action space for position changes
class TradingEnvironment:
    def __init__(self, data_sequences: np.ndarray, data_targets: np.ndarray, 
                 data_dates: np.ndarray, initial_capital: float = None):
        """
        Initialize trading environment
        
        Args:
            data_sequences: Feature sequences (n_samples, n_features, lookback)
            data_targets: Calendar spread values (n_samples,)
            data_dates: Corresponding dates (n_samples,)
            initial_capital: Starting capital
        """
        self.data_sequences = data_sequences
        self.data_targets = data_targets
        self.data_dates = data_dates
        self.n_samples = len(data_sequences)
        
        # Trading parameters
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.tick_value = config.TICK_VALUE
        self.max_contracts = config.MAX_CONTRACTS
        self.transaction_cost = config.TRANSACTION_COST_PER_CONTRACT
        self.slippage = config.BID_ASK_SLIPPAGE if config.SLIPPAGE_ENABLED else 0.0
        
        # Risk management
        self.max_daily_loss = config.MAX_DAILY_LOSS
        self.stop_loss_enabled = config.STOP_LOSS_ENABLED
        self.stop_loss_pct = config.STOP_LOSS_PERCENTAGE
        
        # State tracking
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0  # Current position in contracts (-8 to +8)
        self.cash = self.initial_capital
        self.total_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        
        # Performance tracking
        self.trade_history = []
        self.equity_curve = [self.initial_capital]
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_trade_entry_price = None
        self.current_trade_entry_step = None
        
        # Risk tracking
        self.daily_loss_tracker = 0.0
        self.stop_loss_triggered = False
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= self.n_samples:
            # Return zeros if we've exceeded data
            market_features = np.zeros((self.data_sequences.shape[1], self.data_sequences.shape[2]))
        else:
            market_features = self.data_sequences[self.current_step]
        
        # Flatten market features
        market_state = market_features.flatten()
        
        # Add trading state features
        trading_state = np.array([
            self.position / self.max_contracts,  # Normalized position
            self.unrealized_pnl / self.initial_capital,  # Normalized unrealized PnL
            self.total_pnl / self.initial_capital,  # Normalized total PnL
            self.daily_pnl / self.initial_capital,  # Normalized daily PnL
            (self.cash + self.unrealized_pnl) / self.initial_capital,  # Normalized equity
            self.max_drawdown,  # Current drawdown
            float(self.stop_loss_triggered),  # Stop loss status
            self.current_step / self.n_samples,  # Progress through data
        ])
        
        return np.concatenate([market_state, trading_state])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one trading step
        
        Args:
            action: Action index from POSITION_ACTIONS
            
        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= self.n_samples - 1:
            return self._get_state(), 0.0, True, {'reason': 'end_of_data'}
        
        # Get current and next prices
        current_price = self.data_targets[self.current_step]
        next_price = self.data_targets[self.current_step + 1]
        
        # Calculate position change from action
        position_change_pct = config.POSITION_ACTIONS[action]
        position_change = int(position_change_pct * self.max_contracts)
        
        # Apply position change with constraints
        new_position = np.clip(
            self.position + position_change, 
            -self.max_contracts, 
            self.max_contracts
        )
        actual_position_change = new_position - self.position
        
        # Calculate transaction costs and slippage
        transaction_cost = abs(actual_position_change) * self.transaction_cost
        slippage_cost = abs(actual_position_change) * self.slippage * self.tick_value
        total_cost = transaction_cost + slippage_cost
        
        # Update position and cash
        self.position = new_position
        self.cash -= total_cost
        
        # Move to next step
        self.current_step += 1
        
        # Calculate P&L from price movement
        if self.position != 0:
            price_change = next_price - current_price
            position_pnl = self.position * price_change * self.tick_value
            self.unrealized_pnl += position_pnl
            self.daily_pnl += position_pnl
        
        # Track trade if position closed
        if actual_position_change != 0 and self.current_trade_entry_price is not None:
            if self.position == 0:  # Position fully closed
                self._record_trade()
        
        # Track new trade entry
        if self.position != 0 and self.current_trade_entry_price is None:
            self.current_trade_entry_price = current_price
            self.current_trade_entry_step = self.current_step
        
        # Update equity and performance metrics
        current_equity = self.cash + self.unrealized_pnl
        self.equity_curve.append(current_equity)
        
        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Check risk limits
        done = False
        info = {}
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            done = True
            info['reason'] = 'daily_loss_limit'
            self.stop_loss_triggered = True
        
        # Check stop loss
        if self.stop_loss_enabled and current_drawdown > self.stop_loss_pct:
            done = True
            info['reason'] = 'stop_loss'
            self.stop_loss_triggered = True
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Reset daily P&L at end of day (simplified - you might want date-based logic)
        if self.current_step % 1 == 0:  # Reset every step for now
            self.daily_pnl = 0.0
        
        next_state = self._get_state()
        
        # Add performance info
        info.update({
            'position': self.position,
            'cash': self.cash,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'equity': current_equity,
            'drawdown': current_drawdown,
            'transaction_cost': total_cost,
            'current_price': next_price
        })
        
        return next_state, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on multiple factors"""
        current_equity = self.cash + self.unrealized_pnl
        
        # P&L component (normalized by initial capital)
        pnl_reward = self.daily_pnl / self.initial_capital
        
        # Win rate component
        win_rate = self.winning_trades / max(1, self.total_trades)
        win_rate_reward = (win_rate - 0.5) * 2  # Scale to -1 to 1
        
        # Risk penalty component
        risk_penalty = 0.0
        
        # Drawdown penalty
        if self.max_drawdown > config.DRAWDOWN_PENALTY_THRESHOLD:
            risk_penalty += (self.max_drawdown - config.DRAWDOWN_PENALTY_THRESHOLD) * 10
        
        # Position size penalty (encourage reasonable position sizing)
        position_ratio = abs(self.position) / self.max_contracts
        if position_ratio > 0.8:  # Penalty for using >80% of max position
            risk_penalty += (position_ratio - 0.8) * config.POSITION_SIZE_PENALTY
        
        # Combine components
        reward = (
            config.REWARD_PNL_WEIGHT * pnl_reward +
            config.REWARD_WINRATE_WEIGHT * win_rate_reward -
            config.REWARD_RISK_WEIGHT * risk_penalty
        )

        #print(f"Daily PnL: {self.daily_pnl:.2f}")
        #print(f"PnL Reward Component: {pnl_reward:.4f}")
        #print(f"Final Reward: {reward:.4f}")
        #print(f"Win Rate Reward Component: {win_rate_reward:.4f}")

        return reward
    
    def _record_trade(self):
        """Record completed trade statistics"""
        if self.current_trade_entry_price is None:
            return
            
        current_price = self.data_targets[self.current_step]
        trade_pnl = (current_price - self.current_trade_entry_price) * self.position * self.tick_value
        
        self.total_trades += 1
        
        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        trade_info = {
            'entry_price': self.current_trade_entry_price,
            'exit_price': current_price,
            'entry_step': self.current_trade_entry_step,
            'exit_step': self.current_step,
            'position': self.position,
            'pnl': trade_pnl,
            'duration': self.current_step - self.current_trade_entry_step
        }
        
        self.trade_history.append(trade_info)
        self.total_pnl += trade_pnl
        
        # Reset trade tracking
        self.current_trade_entry_price = None
        self.current_trade_entry_step = None
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) < 2:
            return {}
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (assuming daily data, annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        # Profit factor
        if self.losing_trades > 0:
            winning_pnl = sum([trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0])
            losing_pnl = abs(sum([trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        else:
            profit_factor = float('inf') if self.winning_trades > 0 else 0.0
        
        # Average trade duration
        if self.trade_history:
            avg_trade_duration = np.mean([trade['duration'] for trade in self.trade_history])
        else:
            avg_trade_duration = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_trade_duration': avg_trade_duration,
            'final_equity': self.equity_curve[-1],
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        return len(config.POSITION_ACTIONS)
    
    def get_state_size(self) -> int:
        """Get the size of the state space"""
        if self.current_step < self.n_samples:
            market_features = self.data_sequences[self.current_step].flatten()
        else:
            market_features = np.zeros((self.data_sequences.shape[1] * self.data_sequences.shape[2]))
        
        trading_features = 8  # Number of trading state features
        return len(market_features) + trading_features
    
    def render(self, mode='human'):
        """Render current environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Position: {self.position} contracts")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Unrealized P&L: ${self.unrealized_pnl:,.2f}")
            print(f"Total P&L: ${self.total_pnl:,.2f}")
            print(f"Equity: ${self.cash + self.unrealized_pnl:,.2f}")
            print(f"Drawdown: {self.max_drawdown:.2%}")
            print(f"Trades: {self.total_trades} (Win Rate: {self.winning_trades/max(1,self.total_trades):.1%})")
            print("-" * 50)

# Example environment wrapper for easier use with different data splits
class TradingEnvironmentWrapper:
    def __init__(self, data_splits: Dict):
        """
        Wrapper to easily create environments for train/val/test splits
        
        Args:
            data_splits: Dictionary containing train/val/test data
        """
        self.data_splits = data_splits
        
    def create_env(self, split: str = 'train') -> TradingEnvironment:
        """Create environment for specified data split"""
        if split not in self.data_splits:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.data_splits.keys())}")
        
        data = self.data_splits[split]
        return TradingEnvironment(
            data_sequences=data['sequences'],
            data_targets=data['targets'],
            data_dates=data['dates']
        )
    
    def get_env_info(self) -> Dict:
        """Get information about environments"""
        info = {}
        for split in self.data_splits:
            env = self.create_env(split)
            info[split] = {
                'n_samples': env.n_samples,
                'state_size': env.get_state_size(),
                'action_space_size': env.get_action_space_size(),
                'date_range': (env.data_dates[0], env.data_dates[-1])
            }
        return info

# Testing function
if __name__ == "__main__":
    print("Testing TradingEnvironment...")
    
    # Create dummy data for testing
    n_samples = 100
    n_features = 25
    lookback = 30
    
    dummy_sequences = np.random.randn(n_samples, n_features, lookback)
    dummy_targets = np.random.randn(n_samples) * 10  # Calendar spread values
    dummy_dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Test environment
    env = TradingEnvironment(dummy_sequences, dummy_targets, dummy_dates)
    
    print(f"State size: {env.get_state_size()}")
    print(f"Action space size: {env.get_action_space_size()}")
    
    # Test a few random actions
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for i in range(5):
        action = np.random.randint(0, env.get_action_space_size())
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Done={done}")
        env.render()
        
        if done:
            break
    
    # Print final performance
    metrics = env.get_performance_metrics()
    print("\nFinal Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("Environment testing complete!")