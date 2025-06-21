"""
DDQN Agent for Trading Calendar Spreads
Main agent class that implements the Double Deep Q-Network algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import config
from networks import DoubleDQN, ReplayBuffer, save_model, load_model

class DDQNTradingAgent:
    """
    Double Deep Q-Network agent for trading calendar spreads
    """
    
    def __init__(self, market_feature_dim: int, trading_state_dim: int, 
                 action_space_size: int, lookback_window: int):
        """
        Initialize DDQN agent
        
        Args:
            market_feature_dim: Number of market features
            trading_state_dim: Number of trading state features
            action_space_size: Number of possible actions
            lookback_window: Number of time steps to look back
        """
        self.market_feature_dim = market_feature_dim
        self.trading_state_dim = trading_state_dim
        self.action_space_size = action_space_size
        self.lookback_window = lookback_window
        
        # Initialize networks
        self.ddqn = DoubleDQN(
            market_feature_dim, trading_state_dim, action_space_size, lookback_window
        ).to(config.DEVICE)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.ddqn.online_net.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.MEMORY_SIZE)
        
        # Training parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.update_frequency = config.UPDATE_FREQUENCY
        self.target_update_frequency = config.TARGET_UPDATE_FREQUENCY
        
        # Training tracking
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_epsilons = []
        self.training_metrics = {
            'rewards': [],
            'losses': [],
            'epsilons': [],
            'q_values': [],
            'actions_taken': []
        }
        
        print(f"DDQN Agent initialized with {self._count_parameters()} parameters")
        print(f"Device: {config.DEVICE}")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.ddqn.online_net.parameters() if p.requires_grad)
    
    def _state_to_tensors(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert environment state to network input tensors
        
        Args:
            state: Combined state from environment
            
        Returns:
            market_features, trading_state tensors
        """
        # Split state into market features and trading state
        market_size = self.market_feature_dim * self.lookback_window
        market_features = state[:market_size].reshape(self.market_feature_dim, self.lookback_window)
        trading_state = state[market_size:]
        
        # Convert to tensors and add batch dimension
        market_tensor = torch.FloatTensor(market_features).permute(1, 0).unsqueeze(0).to(config.DEVICE)
        trading_tensor = torch.FloatTensor(trading_state).unsqueeze(0).to(config.DEVICE)
        
        return market_tensor, trading_tensor
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current environment state
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action = random.randint(0, self.action_space_size - 1)
            self.training_metrics['actions_taken'].append(('random', action))
        else:
            # Greedy action (exploitation)
            market_features, trading_state = self._state_to_tensors(state)
            
            with torch.no_grad():
                q_values = self.ddqn(market_features, trading_state, use_target=False)
                action = q_values.argmax().item()
                
                # Store Q-values for analysis
                if training:
                    self.training_metrics['q_values'].append(q_values.cpu().numpy()[0])
                    self.training_metrics['actions_taken'].append(('greedy', action))
        

        #print(f"Action taken: {action} ({config.POSITION_ACTIONS[action]:+.0%})")
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Convert states to market features and trading states
        market_features, trading_state = self._state_to_tensors(state)
        next_market_features, next_trading_state = self._state_to_tensors(next_state)
        
        # Remove batch dimension for storage
        self.replay_buffer.push(
            market_features.squeeze(0).cpu().numpy(),
            trading_state.squeeze(0).cpu().numpy(),
            action,
            reward,
            next_market_features.squeeze(0).cpu().numpy(),
            next_trading_state.squeeze(0).cpu().numpy(),
            done
        )
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        market_features, trading_states, actions, rewards, next_market_features, next_trading_states, dones = batch
        
        # Move to device
        market_features = market_features.to(config.DEVICE)
        trading_states = trading_states.to(config.DEVICE)
        actions = actions.to(config.DEVICE)
        rewards = rewards.to(config.DEVICE)
        next_market_features = next_market_features.to(config.DEVICE)
        next_trading_states = next_trading_states.to(config.DEVICE)
        dones = dones.to(config.DEVICE)
        
        # Current Q-values
        current_q_values = self.ddqn(market_features, trading_states, use_target=False)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_values_online = self.ddqn(next_market_features, next_trading_states, use_target=False)
            next_actions = next_q_values_online.argmax(1)
            
            next_q_values_target = self.ddqn(next_market_features, next_trading_states, use_target=True)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target Q-values
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ddqn.online_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.training_step % self.target_update_frequency == 0:
            self.ddqn.update_target_network(self.tau)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        return loss.item()
    
    def train_episode(self, env, episode: int) -> Dict:
        """
        Train for one episode
        
        Args:
            env: Trading environment
            episode: Episode number
            
        Returns:
            Episode metrics
        """
        state = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        episode_actions = []
        
        while True:
            # Get action
            action = self.get_action(state, training=True)
            episode_actions.append(action)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, next_state, done)
            
            # Train if enough samples available
            if len(self.replay_buffer) >= self.batch_size and steps % self.update_frequency == 0:
                loss = self.train_step()
                total_loss += loss
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        episode_metrics = {
            'episode': episode,
            'total_reward': total_reward,
            'average_loss': total_loss / max(1, steps // self.update_frequency),
            'steps': steps,
            'epsilon': self.epsilon,
            'actions': episode_actions,
            'final_performance': env.get_performance_metrics()
        }
        
        self.episode_rewards.append(total_reward)
        self.episode_losses.append(total_loss / max(1, steps // self.update_frequency))
        self.episode_epsilons.append(self.epsilon)
        
        return episode_metrics
    
    def evaluate_episode(self, env) -> Dict:
        """
        Evaluate agent performance for one episode (no training)
        
        Args:
            env: Trading environment
            
        Returns:
            Evaluation metrics
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        episode_actions = []
        
        while True:
            # Get action (no exploration)
            action = self.get_action(state, training=False)
            episode_actions.append(action)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Get comprehensive performance metrics
        performance_metrics = env.get_performance_metrics()
        
        eval_metrics = {
            'total_reward': total_reward,
            'steps': steps,
            'actions': episode_actions,
            'performance': performance_metrics
        }
        
        return eval_metrics
    
    def train(self, train_env, val_env, num_episodes: int = None) -> Dict:
        """
        Main training loop
        
        Args:
            train_env: Training environment
            val_env: Validation environment
            num_episodes: Number of episodes to train
            
        Returns:
            Training history
        """
        if num_episodes is None:
            num_episodes = config.EPISODES
        
        training_history = {
            'train_rewards': [],
            'val_rewards': [],
            'train_performance': [],
            'val_performance': [],
            'losses': [],
            'epsilons': []
        }
        
        best_val_reward = float('-inf')
        patience_counter = 0
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Replay buffer size: {config.MEMORY_SIZE}")
        print(f"Batch size: {self.batch_size}")
        print(f"Update frequency: {self.update_frequency}")
        
        for episode in range(num_episodes):
            # Training episode
            train_metrics = self.train_episode(train_env, episode)
            
            # Validation episode (every N episodes)
            if episode % config.LOG_FREQUENCY == 0:
                val_metrics = self.evaluate_episode(val_env)
                
                # Store metrics
                training_history['train_rewards'].append(train_metrics['total_reward'])
                training_history['val_rewards'].append(val_metrics['total_reward'])
                training_history['train_performance'].append(train_metrics['final_performance'])
                training_history['val_performance'].append(val_metrics['performance'])
                training_history['losses'].append(train_metrics['average_loss'])
                training_history['epsilons'].append(train_metrics['epsilon'])
                
                # Print progress
                print(f"Episode {episode}/{num_episodes}")
                print(f"  Train Reward: {train_metrics['total_reward']:.4f}")
                print(f"  Val Reward: {val_metrics['total_reward']:.4f}")
                print(f"  Loss: {train_metrics['average_loss']:.6f}")
                print(f"  Epsilon: {train_metrics['epsilon']:.4f}")
                print(f"  Val Performance: {val_metrics['performance'].get('total_return', 0):.2%}")
                print("-" * 50)
                
                # Early stopping check
                if val_metrics['total_reward'] > best_val_reward + config.MIN_IMPROVEMENT:
                    best_val_reward = val_metrics['total_reward']
                    patience_counter = 0
                    
                    # Save best model
                    save_model(
                        self.ddqn, 
                        f"{config.MODEL_SAVE_PATH}best_model.pth",
                        self.optimizer.state_dict(),
                        {'episode': episode, 'val_reward': best_val_reward}
                    )
                else:
                    patience_counter += 1
                
                if patience_counter >= config.PATIENCE:
                    print(f"Early stopping at episode {episode}")
                    break
            
            # Save checkpoint periodically
            if episode % config.SAVE_FREQUENCY == 0 and episode > 0:
                save_model(
                    self.ddqn,
                    f"{config.MODEL_SAVE_PATH}checkpoint_episode_{episode}.pth",
                    self.optimizer.state_dict(),
                    {'episode': episode, 'training_step': self.training_step}
                )
        
        print("Training completed!")
        return training_history
    
    def backtest(self, test_env) -> Dict:
        """
        Perform backtesting on test environment
        
        Args:
            test_env: Test environment
            
        Returns:
            Backtesting results
        """
        print("Starting backtesting...")
        
        # Set to evaluation mode
        self.ddqn.online_net.eval()
        
        # Run evaluation
        results = self.evaluate_episode(test_env)
        
        # Add detailed analysis
        performance = results['performance']
        actions = results['actions']
        
        # Action analysis
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        backtest_results = {
            'performance_metrics': performance,
            'total_reward': results['total_reward'],
            'total_steps': results['steps'],
            'action_distribution': action_counts,
            'action_sequence': actions
        }
        
        # Print results
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        
        print(f"Total Return: {performance.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {performance.get('win_rate', 0):.1%}")
        print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        print(f"Total Trades: {performance.get('total_trades', 0)}")
        print(f"Final Equity: ${performance.get('final_equity', 0):,.2f}")
        
        print(f"\nAction Distribution:")
        for action, count in action_counts.items():
            action_pct = config.POSITION_ACTIONS[action]
            print(f"  Action {action} ({action_pct:+.0%}): {count} times ({count/len(actions):.1%})")
        
        return backtest_results
    
    def plot_training_curves(self, training_history: Dict, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(training_history['train_rewards'], label='Train', alpha=0.7)
        axes[0, 0].plot(training_history['val_rewards'], label='Validation', alpha=0.7)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Losses
        axes[0, 1].plot(training_history['losses'], color='red', alpha=0.7)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Loss')
        axes[0, 1].grid(True)
        
        # Epsilon decay
        axes[1, 0].plot(training_history['epsilons'], color='green', alpha=0.7)
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Performance metrics
        if training_history['val_performance']:
            returns = [p.get('total_return', 0) for p in training_history['val_performance']]
            sharpes = [p.get('sharpe_ratio', 0) for p in training_history['val_performance']]
            
            ax2 = axes[1, 1]
            ax3 = ax2.twinx()
            
            line1 = ax2.plot(returns, 'b-', label='Total Return', alpha=0.7)
            line2 = ax3.plot(sharpes, 'r-', label='Sharpe Ratio', alpha=0.7)
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Total Return', color='b')
            ax3.set_ylabel('Sharpe Ratio', color='r')
            ax2.set_title('Validation Performance')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if config.PLOT_TRAINING_CURVES:
            plt.show()
    
    def save_agent(self, filepath: str, metadata: Dict = None):
        """Save complete agent state"""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'market_feature_dim': self.market_feature_dim,
            'trading_state_dim': self.trading_state_dim,
            'action_space_size': self.action_space_size,
            'lookback_window': self.lookback_window,
            'training_step': self.training_step,
            'epsilon': self.epsilon
        })
        
        save_model(self.ddqn, filepath, self.optimizer.state_dict(), metadata)
    
    def load_agent(self, filepath: str):
        """Load complete agent state"""
        optimizer_state, metadata = load_model(self.ddqn, filepath, load_optimizer=True)
        
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        
        if metadata:
            self.training_step = metadata.get('training_step', 0)
            self.epsilon = metadata.get('epsilon', self.epsilon)
        
        print(f"Agent loaded from {filepath}")
        print(f"Training step: {self.training_step}")
        print(f"Current epsilon: {self.epsilon}")

# Utility functions for agent management
def create_agent_from_env(env) -> DDQNTradingAgent:
    """Create agent with correct dimensions from environment"""
    state = env.reset()
    state_size = len(state)
    
    # Calculate dimensions
    market_size = env.data_sequences.shape[1] * env.data_sequences.shape[2]
    trading_state_size = state_size - market_size
    
    agent = DDQNTradingAgent(
        market_feature_dim=env.data_sequences.shape[1],
        trading_state_dim=trading_state_size,
        action_space_size=env.get_action_space_size(),
        lookback_window=env.data_sequences.shape[2]
    )
    
    return agent

# Testing function
if __name__ == "__main__":
    print("Testing DDQN Agent...")
    
    # This would be used with real environment
    # env = TradingEnvironment(...)
    # agent = create_agent_from_env(env)
    # training_history = agent.train(train_env, val_env, num_episodes=10)
    # backtest_results = agent.backtest(test_env)
    
    print("DDQN Agent ready for use!")