"""
Neural Network Architectures for DDQN Trading System
Combines LSTM for feature extraction with DDQN for trading decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class LSTMFeatureExtractor(nn.Module):
    """
    LSTM for extracting features from time series data
    Processes (batch_size, sequence_length, n_features) sequences
    """
    
    def __init__(self, n_features: int, sequence_length: int):
        super(LSTMFeatureExtractor, self).__init__()
        
        self.n_features = n_features
        self.sequence_length = sequence_length
        
        # LSTM parameters
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.num_layers = config.LSTM_NUM_LAYERS
        self.dropout = config.LSTM_DROPOUT
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.LSTM_BIDIRECTIONAL
        )
        
        # Calculate LSTM output size
        self.lstm_output_size = self.hidden_size * (2 if config.LSTM_BIDIRECTIONAL else 1)
        
        # Additional processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(self.lstm_output_size, config.LSTM_PROCESSING_DIM),
            nn.ReLU(),
            nn.Dropout(config.LSTM_DROPOUT),
            nn.Linear(config.LSTM_PROCESSING_DIM, config.LSTM_PROCESSING_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.LSTM_DROPOUT)
        )
        
        # Output dimension
        self.output_dim = config.LSTM_PROCESSING_DIM // 2
        
    def forward(self, x):
        """
        Forward pass through LSTM
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)
        
        Returns:
            Extracted features of shape (batch_size, output_dim)
        """
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_layers * (2 if config.LSTM_BIDIRECTIONAL else 1),
            batch_size,
            self.hidden_size,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers * (2 if config.LSTM_BIDIRECTIONAL else 1),
            batch_size,
            self.hidden_size,
            device=x.device
        )
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last output from the sequence
        # If bidirectional, lstm_out[:, -1, :] contains both forward and backward final states
        final_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_size)
        
        # Process through additional layers
        features = self.feature_processor(final_output)
        
        return features

class DQNNetwork(nn.Module):
    """
    Deep Q-Network that combines LSTM features with trading state features
    """
    
    def __init__(self, market_feature_dim: int, trading_state_dim: int, 
                 action_space_size: int, sequence_length: int):
        super(DQNNetwork, self).__init__()
        
        self.market_feature_dim = market_feature_dim
        self.trading_state_dim = trading_state_dim
        self.action_space_size = action_space_size
        
        # LSTM for processing market features
        self.lstm = LSTMFeatureExtractor(market_feature_dim, sequence_length)
        
        # Calculate total input dimension for fully connected layers
        total_input_dim = self.lstm.output_dim + trading_state_dim
        
        # Fully connected layers for Q-value estimation
        self.fc_layers = nn.ModuleList()
        
        # First hidden layer
        self.fc_layers.append(nn.Linear(total_input_dim, config.HIDDEN_DIMS[0]))
        
        # Additional hidden layers
        for i in range(1, len(config.HIDDEN_DIMS)):
            self.fc_layers.append(
                nn.Linear(config.HIDDEN_DIMS[i-1], config.HIDDEN_DIMS[i])
            )
        
        # Output layer for Q-values
        self.q_values = nn.Linear(config.HIDDEN_DIMS[-1], action_space_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
    
    def forward(self, market_features, trading_state):
        """
        Forward pass through the network
        
        Args:
            market_features: Market data tensor (batch_size, sequence_length, n_features)
            trading_state: Trading state tensor (batch_size, trading_state_dim)
        
        Returns:
            Q-values for each action (batch_size, action_space_size)
        """
        # Extract features from market data using LSTM
        lstm_features = self.lstm(market_features)
        
        # Combine LSTM features with trading state
        combined_features = torch.cat([lstm_features, trading_state], dim=1)
        
        # Pass through fully connected layers
        x = combined_features
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output Q-values
        q_values = self.q_values(x)
        
        return q_values

class DoubleDQN(nn.Module):
    """
    Double Deep Q-Network implementation with LSTM
    Uses two networks: online and target
    """
    
    def __init__(self, market_feature_dim: int, trading_state_dim: int,
                 action_space_size: int, sequence_length: int):
        super(DoubleDQN, self).__init__()
        
        # Online network (updated frequently)
        self.online_net = DQNNetwork(
            market_feature_dim, trading_state_dim, action_space_size, sequence_length
        )
        
        # Target network (updated less frequently)
        self.target_net = DQNNetwork(
            market_feature_dim, trading_state_dim, action_space_size, sequence_length
        )
        
        # Initialize target network with same weights as online network
        self.update_target_network()
        
        # Freeze target network parameters (will be updated via soft updates)
        for param in self.target_net.parameters():
            param.requires_grad = False
    
    def forward(self, market_features, trading_state, use_target=False):
        """
        Forward pass through either online or target network
        
        Args:
            market_features: Market data tensor (batch_size, sequence_length, n_features)
            trading_state: Trading state tensor (batch_size, trading_state_dim)
            use_target: Whether to use target network
        
        Returns:
            Q-values
        """
        if use_target:
            return self.target_net(market_features, trading_state)
        else:
            return self.online_net(market_features, trading_state)
    
    def update_target_network(self, tau: float = None):
        """
        Update target network parameters
        
        Args:
            tau: Soft update parameter (None for hard update)
        """
        if tau is None:
            # Hard update: copy weights completely
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            # Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )
    
    def get_action(self, market_features, trading_state, epsilon=0.0):
        """
        Get action using epsilon-greedy policy
        
        Args:
            market_features: Market data tensor (batch_size, sequence_length, n_features)
            trading_state: Trading state tensor (batch_size, trading_state_dim)
            epsilon: Exploration probability
        
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.online_net.action_space_size)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.online_net(market_features, trading_state)
                return q_values.argmax().item()

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, market_features, trading_state, action, reward, 
             next_market_features, next_trading_state, done):
        """Store a transition in the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (
            market_features, trading_state, action, reward,
            next_market_features, next_trading_state, done
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        market_features = []
        trading_states = []
        actions = []
        rewards = []
        next_market_features = []
        next_trading_states = []
        dones = []
        
        for idx in batch:
            mf, ts, a, r, nmf, nts, d = self.buffer[idx]
            market_features.append(mf)
            trading_states.append(ts)
            actions.append(a)
            rewards.append(r)
            next_market_features.append(nmf)
            next_trading_states.append(nts)
            dones.append(d)
        
        return (
            torch.FloatTensor(market_features),
            torch.FloatTensor(trading_states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_market_features),
            torch.FloatTensor(next_trading_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# Utility functions for network management
def save_model(model: DoubleDQN, filepath: str, optimizer_state: dict = None, 
               metadata: dict = None):
    """Save model checkpoint"""
    checkpoint = {
        'online_net_state_dict': model.online_net.state_dict(),
        'target_net_state_dict': model.target_net.state_dict(),
    }
    
    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(model: DoubleDQN, filepath: str, load_optimizer: bool = False):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    
    model.online_net.load_state_dict(checkpoint['online_net_state_dict'])
    model.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    optimizer_state = None
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
    
    metadata = checkpoint.get('metadata', {})
    
    print(f"Model loaded from {filepath}")
    return optimizer_state, metadata

# Testing function
if __name__ == "__main__":
    print("Testing LSTM neural network architectures...")
    
    # Test parameters
    batch_size = 32
    market_feature_dim = 25
    trading_state_dim = 8
    action_space_size = 9
    sequence_length = 30
    
    # Create test data - NOTE: Different shape for LSTM
    market_features = torch.randn(batch_size, sequence_length, market_feature_dim)
    trading_state = torch.randn(batch_size, trading_state_dim)
    
    # Test LSTM feature extractor
    lstm = LSTMFeatureExtractor(market_feature_dim, sequence_length)
    lstm_output = lstm(market_features)
    print(f"LSTM output shape: {lstm_output.shape}")
    
    # Test DQN network
    dqn = DQNNetwork(market_feature_dim, trading_state_dim, action_space_size, sequence_length)
    q_values = dqn(market_features, trading_state)
    print(f"DQN Q-values shape: {q_values.shape}")
    
    # Test Double DQN
    ddqn = DoubleDQN(market_feature_dim, trading_state_dim, action_space_size, sequence_length)
    
    # Test online network
    online_q_values = ddqn(market_features, trading_state, use_target=False)
    print(f"Online network Q-values shape: {online_q_values.shape}")
    
    # Test target network
    target_q_values = ddqn(market_features, trading_state, use_target=True)
    print(f"Target network Q-values shape: {target_q_values.shape}")
    
    # Test action selection
    action = ddqn.get_action(market_features[0:1], trading_state[0:1], epsilon=0.1)
    print(f"Selected action: {action}")
    
    print("LSTM neural network testing complete!")
