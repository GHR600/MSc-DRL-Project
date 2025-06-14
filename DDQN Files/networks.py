"""
Neural Network Architectures for DDQN Trading System
Combines CNN for feature extraction with DDQN for trading decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class CNNFeatureExtractor(nn.Module):
    """
    1D CNN for extracting features from time series data
    Processes (n_features, lookback_window) sequences
    """
    
    def __init__(self, n_features: int, lookback_window: int):
        super(CNNFeatureExtractor, self).__init__()
        
        self.n_features = n_features
        self.lookback_window = lookback_window
        
        # CNN layers for temporal feature extraction
        self.conv_layers = nn.ModuleList()
        
        # First convolution layer
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=config.CNN_CHANNELS[0],
                kernel_size=config.CNN_KERNEL_SIZES[0],
                padding=config.CNN_KERNEL_SIZES[0]//2
            )
        )
        
        # Additional convolution layers
        for i in range(1, len(config.CNN_CHANNELS)):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=config.CNN_CHANNELS[i-1],
                    out_channels=config.CNN_CHANNELS[i],
                    kernel_size=config.CNN_KERNEL_SIZES[i],
                    padding=config.CNN_KERNEL_SIZES[i]//2
                )
            )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(channels) for channels in config.CNN_CHANNELS
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.CNN_DROPOUT)
        
        # Global average pooling to reduce dimensionality
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate output dimension
        self.output_dim = config.CNN_CHANNELS[-1]
        
    def forward(self, x):
        """
        Forward pass through CNN
        
        Args:
            x: Input tensor of shape (batch_size, n_features, lookback_window)
        
        Returns:
            Extracted features of shape (batch_size, output_dim)
        """
        # Apply convolution layers with ReLU and batch norm
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        return x

class DQNNetwork(nn.Module):
    """
    Deep Q-Network that combines CNN features with trading state features
    """
    
    def __init__(self, market_feature_dim: int, trading_state_dim: int, 
                 action_space_size: int, lookback_window: int):
        super(DQNNetwork, self).__init__()
        
        self.market_feature_dim = market_feature_dim
        self.trading_state_dim = trading_state_dim
        self.action_space_size = action_space_size
        
        # CNN for processing market features
        self.cnn = CNNFeatureExtractor(market_feature_dim, lookback_window)
        
        # Calculate total input dimension for fully connected layers
        total_input_dim = self.cnn.output_dim + trading_state_dim
        
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
        self.dropout = nn.Dropout(config.CNN_DROPOUT)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, market_features, trading_state):
        """
        Forward pass through the network
        
        Args:
            market_features: Market data tensor (batch_size, n_features, lookback)
            trading_state: Trading state tensor (batch_size, trading_state_dim)
        
        Returns:
            Q-values for each action (batch_size, action_space_size)
        """
        # Extract features from market data using CNN
        cnn_features = self.cnn(market_features)
        
        # Combine CNN features with trading state
        combined_features = torch.cat([cnn_features, trading_state], dim=1)
        
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
    Double Deep Q-Network implementation
    Uses two networks: online and target
    """
    
    def __init__(self, market_feature_dim: int, trading_state_dim: int,
                 action_space_size: int, lookback_window: int):
        super(DoubleDQN, self).__init__()
        
        # Online network (updated frequently)
        self.online_net = DQNNetwork(
            market_feature_dim, trading_state_dim, action_space_size, lookback_window
        )
        
        # Target network (updated less frequently)
        self.target_net = DQNNetwork(
            market_feature_dim, trading_state_dim, action_space_size, lookback_window
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
            market_features: Market data tensor
            trading_state: Trading state tensor  
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
            market_features: Market data tensor
            trading_state: Trading state tensor
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
    print("Testing neural network architectures...")
    
    # Test parameters
    batch_size = 32
    market_feature_dim = 25
    trading_state_dim = 8
    action_space_size = 9
    lookback_window = 30
    
    # Create test data
    market_features = torch.randn(batch_size, market_feature_dim, lookback_window)
    trading_state = torch.randn(batch_size, trading_state_dim)
    
    # Test CNN feature extractor
    cnn = CNNFeatureExtractor(market_feature_dim, lookback_window)
    cnn_output = cnn(market_features)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Test DQN network
    dqn = DQNNetwork(market_feature_dim, trading_state_dim, action_space_size, lookback_window)
    q_values = dqn(market_features, trading_state)
    print(f"DQN Q-values shape: {q_values.shape}")
    
    # Test Double DQN
    ddqn = DoubleDQN(market_feature_dim, trading_state_dim, action_space_size, lookback_window)
    
    # Test online network
    online_q_values = ddqn(market_features, trading_state, use_target=False)
    print(f"Online network Q-values shape: {online_q_values.shape}")
    
    # Test target network
    target_q_values = ddqn(market_features, trading_state, use_target=True)
    print(f"Target network Q-values shape: {target_q_values.shape}")
    
    # Test action selection
    action = ddqn.get_action(market_features[0:1], trading_state[0:1], epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test replay buffer
    replay_buffer = ReplayBuffer(1000)
    
    # Add some transitions
    for i in range(100):
        replay_buffer.push(
            market_features[0].numpy(),
            trading_state[0].numpy(),
            np.random.randint(0, action_space_size),
            np.random.randn(),
            market_features[0].numpy(),
            trading_state[0].numpy(),
            False
        )
    
    # Sample batch
    batch = replay_buffer.sample(16)
    print(f"Replay buffer batch size: {len(batch)}")
    print(f"Batch market features shape: {batch[0].shape}")
    
    print("Neural network testing complete!")
