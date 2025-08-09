import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GoldmanRollStrategy:
    """
    Implementation of Mou's (2011) Goldman Roll front-running strategy.
    
    Strategy Logic:
    - Front-run the GSCI rolling activity by creating calendar spread positions
    - GSCI rolls from 5th to 9th business day of each month
    - Strategy: Enter positions 10-6 business days before GSCI roll (front-run by 10 days)
    - Calendar spread: Short front month, Long back month
    - Unwind positions during GSCI rolling period
    """
    
    def __init__(self, data, transaction_cost=0.75, slippage=0.5, tick_value=10.0):
        
        self.data = data.copy()
        # Handle DD/MM/YYYY format (dayfirst=True)
        self.data['Dates'] = pd.to_datetime(self.data['Dates'], dayfirst=True)
        self.data = self.data.sort_values('Dates').reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.tick_value = tick_value
        
        # Add business day calculations
        self._add_business_day_features()
        
    def _get_business_day_of_month(self, date):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        first_day = date.replace(day=1)
        business_days = 0
        current_date = first_day
        
        while current_date <= date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                business_days += 1
            current_date += timedelta(days=1)
        
        return business_days
    
    def _add_business_day_features(self):
        self.data['business_day_of_month'] = self.data['Dates'].apply(self._get_business_day_of_month)
        
        # GSCI rolling period: 5th to 9th business day
        self.data['gsci_roll_period'] = (
            (self.data['business_day_of_month'] >= 5) & 
            (self.data['business_day_of_month'] <= 9)
        ).astype(int)
        
        #
        self.data['strategy1_entry'] = (
            (self.data['business_day_of_month'] >= 1) & 
            (self.data['business_day_of_month'] <= 5)
        ).astype(int)
        
        
        self.data['strategy1_entry'] = (
            (self.data['business_day_of_month'] >= 1) & 
            (self.data['business_day_of_month'] <= 4)
        ).astype(int)
        
    def calculate_calendar_spread(self):
        self.data['calendar_spread'] = self.data['PX_LAST1'] - self.data['PX_LAST2']
        self.data['spread_return'] = self.data['calendar_spread'].pct_change()
        
    def backtest_strategy(self, initial_capital=10000, data_subset='full'):
        
        self.calculate_calendar_spread()
        
        # Select data subset based on parameter
        if data_subset == 'train':
            # First 70% of data
            end_idx = int(len(self.data) * 0.7)
            data_to_use = self.data.iloc[:end_idx].copy()
            print(f"Using TRAINING data: {data_to_use['Dates'].min().strftime('%Y-%m-%d')} to {data_to_use['Dates'].max().strftime('%Y-%m-%d')}")
        elif data_subset == 'test':
            # Last 20% of data (skipping the 15% validation period to match ML setup)
            start_idx = int(len(self.data) * 0.8)  # Start from 80% to get last 20%
            data_to_use = self.data.iloc[start_idx:].copy()
            print(f"Using TEST data: {data_to_use['Dates'].min().strftime('%Y-%m-%d')} to {data_to_use['Dates'].max().strftime('%Y-%m-%d')}")
        else:
            # Full dataset
            data_to_use = self.data.copy()
            print(f"Using FULL data: {data_to_use['Dates'].min().strftime('%Y-%m-%d')} to {data_to_use['Dates'].max().strftime('%Y-%m-%d')}")
        
        # Reset index for the subset
        data_to_use = data_to_use.reset_index(drop=True)
        
        
        #  tracking variables
        position = 0  # Calendar spread position (positive = long spread)
        cash = initial_capital
        total_pnl = 0
        trade_history = []
        equity_curve = [initial_capital]
        
        entry_col = 'strategy1_entry'  
        
        for i in range(1, len(data_to_use)):
            current_date = data_to_use.iloc[i]['Dates']
            current_spread = data_to_use.iloc[i]['calendar_spread']
            prev_spread = data_to_use.iloc[i-1]['calendar_spread']
            
            # Entry logic: Enter during strategy entry period
            if data_to_use.iloc[i][entry_col] == 1 and position == 0:
                # Enter calendar spread position (short front, long back)
                #   spread to decrease during GSCI roll, so  short the spread
                position = -1  # Short the spread
                entry_price = current_spread
                entry_date = current_date
                
                # Transaction costs (matching ML setup exactly)
                transaction_cost = abs(position) * self.transaction_cost  # $0.75 per contract
                slippage_cost = abs(position) * self.slippage * self.tick_value  # 0.5 ticks * $10 = $5.00
                total_entry_cost = transaction_cost + slippage_cost  # $5.75 total
                cash -= total_entry_cost
                
                trade_history.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'position': position,
                    'entry_cost': total_entry_cost,
                    'action': 'ENTER'
                })
            
            # Exit logic: Exit during GSCI rolling period
            elif data_to_use.iloc[i]['gsci_roll_period'] == 1 and position != 0:
                # Exit the position
                exit_price = current_spread
                exit_date = current_date
                
                # Calculate P&L ( short the spread, so profit when spread decreases)
                trade_pnl = -position * (exit_price - entry_price) * self.tick_value
                
                # Transaction costs for exit 
                transaction_cost = abs(position) * self.transaction_cost  # $0.75 per contract
                slippage_cost = abs(position) * self.slippage * self.tick_value  # 0.5 ticks * $10 = $5.00
                total_exit_cost = transaction_cost + slippage_cost  # $5.75 total
                
                # Total round-trip cost = entry + exit = $11.50
                total_round_trip_cost = trade_history[-1]['entry_cost'] + total_exit_cost
                trade_pnl -= total_exit_cost
                
                total_pnl += trade_pnl
                cash += trade_pnl
                
                trade_history.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'trade_pnl': trade_pnl,
                    'total_cost': total_round_trip_cost,
                    'action': 'EXIT'
                })
                
                position = 0
            
            # Update equity curve
            if position != 0:
                # Mark to market
                unrealized_pnl = -position * (current_spread - entry_price) * self.tick_value
                current_equity = cash + unrealized_pnl
            else:
                current_equity = cash
                
            equity_curve.append(current_equity)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        completed_trades = [t for t in trade_history if 'trade_pnl' in t]
        winning_trades = [t for t in completed_trades if t['trade_pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['trade_pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['trade_pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['trade_pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0
        
        results = {
            'strategy': 'Goldman Roll Strategy 1 (10-day front-run)',
            'data_period': data_subset.upper(),
            'total_return': total_return,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1],
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trade_history': completed_trades,
            'equity_curve': equity_curve,
            'dates': data_to_use['Dates'].tolist()
        }
        
        return results
    
    def print_results(self, results):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print(f"{results['strategy']} BACKTEST RESULTS - {results['data_period']} DATA")
        print(f"{'='*60}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"")
        print(f"Trading Statistics:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Winning trades: {results['winning_trades']}")
        print(f"  Losing trades: {results['losing_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Average Win: ${results['avg_win']:.2f}")
        print(f"  Average Loss: ${results['avg_loss']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")

# Example usage:
def run_goldman_roll_backtest(data_path):
    
    
    # Load data 
    column_names = [
        'PX_OPEN1', 'PX_HIGH1', 'PX_LOW1', 'PX_LAST1', 'PX_VOLUME1', 'OPEN_INT1',
        'PX_OPEN2', 'PX_HIGH2', 'PX_LOW2', 'PX_LAST2', 'PX_VOLUME2', 'OPEN_INT2',
        'Dates', 'VOL Change1', 'Vol Change %1', 'OI Change1', 'OI Change %1',
        'CALENDAR', 'Vol Ratio', 'Vol Ratio Change', 'OI Ratio', 'OI Ratio Change',
        'VOL Change2', 'Vol Change %2', 'OI Change2', 'OI Change %2'
    ]
    
    data = pd.read_csv(data_path, names=column_names, header=None, skiprows=3)
    
    # Initialize strategy
    strategy = GoldmanRollStrategy(data)
    
    print(f"Total data points: {len(data)}")
    print(f"Date range: {pd.to_datetime(data['Dates'], dayfirst=True).min()} to {pd.to_datetime(data['Dates'], dayfirst=True).max()}")
    
    # Store all results
    all_results = {}
    
    # Run on training data (first 70%)
    print(f"\n{'='*80}")
    print("TRAINING DATA RESULTS")
    print(f"{'='*80}")
    
    train_results = strategy.backtest_strategy(data_subset='train')
    strategy.print_results(train_results)
    all_results['train'] = train_results
    
    # Run on test data (last 20%)
    print(f"\n{'='*80}")
    print("TEST DATA RESULTS")
    print(f"{'='*80}")
    
    test_results = strategy.backtest_strategy(data_subset='test')
    strategy.print_results(test_results)
    all_results['test'] = test_results
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Training':<15} {'Test':<15}")
    print(f"{'-'*50}")
    print(f"{'Total Return':<20} {train_results['total_return']:>14.2%} {test_results['total_return']:>14.2%}")
    print(f"{'Sharpe Ratio':<20} {train_results['sharpe_ratio']:>14.2f} {test_results['sharpe_ratio']:>14.2f}")
    print(f"{'Max Drawdown':<20} {train_results['max_drawdown']:>14.2%} {test_results['max_drawdown']:>14.2%}")
    print(f"{'Win Rate':<20} {train_results['win_rate']:>14.1%} {test_results['win_rate']:>14.1%}")
    print(f"{'Total Trades':<20} {train_results['total_trades']:>14d} {test_results['total_trades']:>14d}")
    print(f"{'Final Equity':<20} ${train_results['final_equity']:>13,.0f} ${test_results['final_equity']:>13,.0f}")
    
    return all_results

if __name__ == "__main__":

    csv_file_path = "C:\MSc-DRL-Project\CSVfiles\SB12 - Sheet1.csv"
    
    if os.path.exists(csv_file_path):
        all_results = run_goldman_roll_backtest(csv_file_path)
    else:
        print(f"File not found: {csv_file_path}")