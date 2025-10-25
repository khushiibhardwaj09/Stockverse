"""
Backtesting Framework
Validates trading strategies using historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class Backtester:
    """Backtesting framework for trading strategies"""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital for simulation
            commission: Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio_values = []
        
    def run_strategy(self, df, predictions, strategy_type='long_only'):
        """
        Run backtest on a trading strategy
        
        Args:
            df: DataFrame with historical data (must have 'Date' and 'Close' columns)
            predictions: Array of predictions (same length as df)
            strategy_type: 'long_only', 'short_only', or 'long_short'
            
        Returns:
            Dictionary with backtest results
        """
        df = df.copy()
        df['Prediction'] = predictions
        df['Returns'] = df['Close'].pct_change()
        
        if strategy_type == 'long_only':
            df['Signal'] = (df['Prediction'] > df['Close'].shift(1)).astype(int)
        elif strategy_type == 'short_only':
            df['Signal'] = (df['Prediction'] < df['Close'].shift(1)).astype(int) * -1
        elif strategy_type == 'long_short':
            df['Signal'] = np.where(df['Prediction'] > df['Close'].shift(1), 1, -1)
        
        df['Position'] = df['Signal'].diff()
        
        capital = self.initial_capital
        position = 0
        portfolio_values = [capital]
        trades = []
        
        for i in range(1, len(df)):
            if df['Position'].iloc[i] == 1:
                shares = capital / df['Close'].iloc[i]
                cost = shares * df['Close'].iloc[i] * (1 + self.commission)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    trades.append({
                        'Date': df['Date'].iloc[i],
                        'Action': 'BUY',
                        'Price': df['Close'].iloc[i],
                        'Shares': shares,
                        'Capital': capital
                    })
            
            elif df['Position'].iloc[i] == -1 and position > 0:
                proceeds = position * df['Close'].iloc[i] * (1 - self.commission)
                capital += proceeds
                trades.append({
                    'Date': df['Date'].iloc[i],
                    'Action': 'SELL',
                    'Price': df['Close'].iloc[i],
                    'Shares': position,
                    'Capital': capital
                })
                position = 0
            
            portfolio_value = capital + (position * df['Close'].iloc[i] if position > 0 else 0)
            portfolio_values.append(portfolio_value)
        
        if position > 0:
            final_proceeds = position * df['Close'].iloc[-1] * (1 - self.commission)
            capital += final_proceeds
            portfolio_value = capital
        else:
            portfolio_value = capital
        
        self.trades = trades
        self.portfolio_values = portfolio_values
        
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
        num_trades = len(trades)
        
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        buy_hold_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        
        return {
            'final_value': portfolio_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'trades': trades
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        """
        Calculate maximum drawdown
        
        Args:
            portfolio_values: List of portfolio values over time
            
        Returns:
            Maximum drawdown percentage
        """
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_win_rate(self):
        """
        Calculate win rate from trades
        
        Returns:
            Win rate percentage
        """
        if len(self.trades) < 2:
            return 0
        
        profitable_trades = 0
        total_trades = 0
        
        for i in range(1, len(self.trades), 2):
            if i < len(self.trades):
                buy_price = self.trades[i-1]['Price']
                sell_price = self.trades[i]['Price']
                if sell_price > buy_price:
                    profitable_trades += 1
                total_trades += 1
        
        return (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    def get_trade_summary(self):
        """
        Get summary of all trades
        
        Returns:
            DataFrame with trade information
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self):
        """Initialize portfolio optimizer"""
        pass
    
    def calculate_returns(self, prices_df):
        """
        Calculate returns from price data
        
        Args:
            prices_df: DataFrame with stock prices (columns are tickers)
            
        Returns:
            DataFrame with returns
        """
        return prices_df.pct_change().dropna()
    
    def calculate_portfolio_stats(self, weights, returns):
        """
        Calculate portfolio statistics
        
        Args:
            weights: Array of portfolio weights
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary with portfolio statistics
        """
        portfolio_return = np.dot(weights, returns.mean()) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_sharpe_ratio(self, returns_df, risk_free_rate=0.02):
        """
        Find portfolio weights that maximize Sharpe ratio
        
        Args:
            returns_df: DataFrame with asset returns
            risk_free_rate: Risk-free rate (default 2%)
            
        Returns:
            Optimal weights and portfolio statistics
        """
        num_assets = len(returns_df.columns)
        
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return = np.dot(weights, returns_df.mean()) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
            
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std
            results[2,i] = sharpe
        
        max_sharpe_idx = np.argmax(results[2])
        optimal_weights = weights_record[max_sharpe_idx]
        
        return {
            'weights': optimal_weights,
            'expected_return': results[0, max_sharpe_idx],
            'volatility': results[1, max_sharpe_idx],
            'sharpe_ratio': results[2, max_sharpe_idx],
            'all_results': results,
            'all_weights': weights_record
        }
    
    def optimize_min_volatility(self, returns_df):
        """
        Find portfolio weights that minimize volatility
        
        Args:
            returns_df: DataFrame with asset returns
            
        Returns:
            Optimal weights and portfolio statistics
        """
        num_assets = len(returns_df.columns)
        
        num_portfolios = 10000
        min_vol = float('inf')
        min_vol_weights = None
        min_vol_return = 0
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            portfolio_return = np.dot(weights, returns_df.mean()) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
            
            if portfolio_std < min_vol:
                min_vol = portfolio_std
                min_vol_weights = weights
                min_vol_return = portfolio_return
        
        return {
            'weights': min_vol_weights,
            'expected_return': min_vol_return,
            'volatility': min_vol,
            'sharpe_ratio': min_vol_return / min_vol if min_vol > 0 else 0
        }
