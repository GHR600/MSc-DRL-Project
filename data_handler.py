"""
Data Handler for DDQN Trading System
Handles loading, preprocessing, and feature engineering of trading data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import config


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TradingDataHandler:
    def __init__(self, csv_path: str):
        """
        Initialize the data handler
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.feature_columns = config.FEATURE_COLUMNS
        self.date_column = config.DATE_COLUMN
        self.target_column = config.TARGET_COLUMN
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        print(f"Loading data from {self.csv_path}...")
        
        # Load CSV with proper column names
        column_names = [
            'PX_OPEN1', 'PX_HIGH1', 'PX_LOW1', 'PX_LAST1', 'PX_VOLUME1', 'OPEN_INT1',
            'PX_OPEN2', 'PX_HIGH2', 'PX_LOW2', 'PX_LAST2', 'PX_VOLUME2', 'OPEN_INT2',
            'Dates', 'VOL Change1', 'Vol Change %1', 'OI Change1', 'OI Change %1',
            'CALENDAR', 'Vol Ratio', 'Vol Ratio Change', 'OI Ratio', 'OI Ratio Change',
            'VOL Change2', 'Vol Change %2', 'OI Change2', 'OI Change %2'
            ]
        
        self.raw_data = pd.read_csv(self.csv_path, names=column_names, header=None, skiprows=2)
        
        # Convert date column 
        self.raw_data['Dates'] = pd.to_datetime(self.raw_data['Dates'], dayfirst=True)
        self.raw_data = self.raw_data.sort_values('Dates').reset_index(drop=True)
        self.raw_data['date'] = self.raw_data['Dates']

        print(f"Loaded {len(self.raw_data)} rows of data")
        print(f"Date range: {self.raw_data['Dates'].min()} to {self.raw_data['Dates'].max()}")

        print(f"CALENDAR spread stats:")
        print(f"  Min: {self.raw_data['CALENDAR'].min()}")
        print(f"  Max: {self.raw_data['CALENDAR'].max()}")
        print(f"  Mean: {self.raw_data['CALENDAR'].mean()}")
        print(f"  Std: {self.raw_data['CALENDAR'].std()}")

        
        return self.raw_data
    
    def calculate_business_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Goldman roll related features"""
        df = df.copy()
        
        # Calculate business day of month
        df['business_day_of_month'] = df['Dates'].apply(self._get_business_day_of_month)
        
        # Calculate days until/since Goldman roll period
        df['days_to_roll_start'] = df['business_day_of_month'].apply(
            lambda x: max(0, config.GOLDMAN_ROLL_START_DAY - x) if x < config.GOLDMAN_ROLL_START_DAY 
            else 0
        )
        
        df['days_since_roll_end'] = df['business_day_of_month'].apply(
            lambda x: max(0, x - config.GOLDMAN_ROLL_END_DAY) if x > config.GOLDMAN_ROLL_END_DAY 
            else 0
        )
        
        # Binary feature for Goldman roll period
        df['in_goldman_roll'] = (
            (df['business_day_of_month'] >= config.GOLDMAN_ROLL_START_DAY) & 
            (df['business_day_of_month'] <= config.GOLDMAN_ROLL_END_DAY)
        ).astype(int)
        
        # Extended window feature (15 days around roll)
        df['in_extended_roll_window'] = (
            (df['business_day_of_month'] >= config.GOLDMAN_ROLL_START_DAY - 5) & 
            (df['business_day_of_month'] <= config.GOLDMAN_ROLL_END_DAY + 5)
        ).astype(int)
        
        return df
    
    def _get_business_day_of_month(self, date: datetime) -> int:
        """Calculate which business day of the month this is"""
        # Get first day of month
        first_day = date.replace(day=1)
        
        # Count business days from first day to current date
        business_days = 0
        current_date = first_day
        
        while current_date <= date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                business_days += 1
            current_date += timedelta(days=1)
            
        return business_days
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        df = df.copy()
    
        # Price-based features (update column names)
        df['front_hl_ratio'] = df['PX_HIGH1'] / df['PX_LOW1']
        df['second_hl_ratio'] = df['PX_HIGH2'] / df['PX_LOW2']
        df['front_close_position'] = (df['PX_LAST1'] - df['PX_LOW1']) / (df['PX_HIGH1'] - df['PX_LOW1'])
        df['second_close_position'] = (df['PX_LAST2'] - df['PX_LOW2']) / (df['PX_HIGH2'] - df['PX_LOW2'])

        # Volume and OI momentum
        df['volume_momentum'] = df['Vol Ratio'].rolling(5).mean()
        df['oi_momentum'] = df['OI Ratio'].rolling(5).mean()

        # Calendar spread momentum
        df['spread_momentum_5'] = df['CALENDAR'].rolling(5).mean()
        df['spread_momentum_10'] = df['CALENDAR'].rolling(10).mean()
        df['spread_volatility'] = df['CALENDAR'].rolling(10).std()

        # Relative strength
        df['front_vs_second_strength'] = df['PX_LAST1'] / df['PX_LAST2']
    
        return df
    
    def preprocess_data(self) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        if self.raw_data is None:
            self.load_data()
            
        df = self.raw_data.copy()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Add business day features
        df = self.calculate_business_day_features(df)
        
        # Engineer additional features
        df = self.engineer_features(df)
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        self.processed_data = df
        print(f"Preprocessing complete. Final dataset: {len(df)} rows")
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, lookback: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for CNN input
    
        Returns:
        features: (n_samples, n_features, lookback_window)
        targets: (n_samples,) - calendar spread values
        dates: (n_samples,) - corresponding dates
        """
        if lookback is None:
            lookback = config.LOOKBACK_WINDOW
        
        # Get feature columns (exclude date columns and any engineered date features)
        feature_cols = [col for col in df.columns if col not in ['date', 'Dates']]
    
        # Also exclude any datetime-related engineered features
        feature_cols = [col for col in feature_cols if not col.startswith('business_day')]
        feature_cols = [col for col in feature_cols if not col.startswith('days_')]
        feature_cols = [col for col in feature_cols if not col.startswith('in_goldman')]
        feature_cols = [col for col in feature_cols if not col.startswith('in_extended')]
    
        print(f"Using {len(feature_cols)} features for training")
        print(f"Features: {feature_cols}...")  # Print first 5 feature names
    
        # Normalize features
        feature_data = self.scaler.fit_transform(df[feature_cols])
    
        sequences = []
        targets = []
        dates = []
    
        for i in range(lookback, len(df)):
            # Get sequence of features
            sequence = feature_data[i-lookback:i].T  # Transpose to (n_features, lookback)
            sequences.append(sequence)
        
            # Target is the calendar spread at current time
            targets.append(df.iloc[i][self.target_column])
            dates.append(df.iloc[i]['Dates'])  # Use 'Dates' instead of 'date'
    
        return np.array(sequences), np.array(targets), np.array(dates)
    
    def split_data(self, sequences: np.ndarray, targets: np.ndarray, dates: np.ndarray, 
                   quick_test: bool = False) -> Dict:
        """Split data into train/validation/test sets"""
        
        if quick_test:
            # Use only recent data for quick testing
            n_years = config.QUICK_TEST_YEARS
            cutoff_date = dates[-1] - pd.DateOffset(years=n_years)
            mask = pd.to_datetime(dates) >= cutoff_date
            
            sequences = sequences[mask]
            targets = targets[mask]
            dates = dates[mask]
            
        n_samples = len(sequences)
        
        # Calculate split indices
        train_end = int(n_samples * config.TRAIN_RATIO)
        val_end = int(n_samples * (config.TRAIN_RATIO + config.VAL_RATIO))
        
        data_splits = {
            'train': {
                'sequences': sequences[:train_end],
                'targets': targets[:train_end],
                'dates': dates[:train_end]
            },
            'val': {
                'sequences': sequences[train_end:val_end],
                'targets': targets[train_end:val_end],
                'dates': dates[train_end:val_end]
            },
            'test': {
                'sequences': sequences[val_end:],
                'targets': targets[val_end:],
                'dates': dates[val_end:]
            }
        }
        
        print(f"Data splits:")
        print(f"  Train: {len(data_splits['train']['sequences'])} samples")
        print(f"  Validation: {len(data_splits['val']['sequences'])} samples") 
        print(f"  Test: {len(data_splits['test']['sequences'])} samples")
        
        return data_splits
    
    def get_feature_info(self) -> Dict:
        """Get information about features"""
        if self.processed_data is None:
            self.preprocess_data()
            
        feature_cols = [col for col in self.processed_data.columns if col not in ['date']]
        
        return {
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'n_samples': len(self.processed_data),
            'date_range': (self.processed_data['date'].min(), self.processed_data['date'].max())
        }

# Example usage and testing
if __name__ == "__main__":
    # This will be useful for testing the data handler
    print("Testing TradingDataHandler...")
    
    # You would initialize like this:
    # handler = TradingDataHandler("your_data.csv")
    # data = handler.preprocess_data()
    # sequences, targets, dates = handler.create_sequences(data)
    # splits = handler.split_data(sequences, targets, dates, quick_test=True)
    
    print("Data handler ready for use!")
