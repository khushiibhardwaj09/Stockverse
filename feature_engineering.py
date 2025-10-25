"""
Feature Engineering Module
Implements technical indicators and feature creation for stock prediction
"""

import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


class FeatureEngineer:
    """Creates technical indicators and features for stock data"""
    
    def __init__(self):
        pass
    
    def add_moving_averages(self, df, windows=[5, 10, 20, 50, 200]):
        """
        Add Simple Moving Averages (SMA) for different windows
        
        Args:
            df: DataFrame with OHLC data
            windows: List of window sizes for moving averages
            
        Returns:
            DataFrame with SMA columns added
        """
        df = df.copy()
        
        for window in windows:
            if len(df) >= window:
                sma = SMAIndicator(close=df['Close'], window=window)
                df[f'SMA_{window}'] = sma.sma_indicator()
        
        return df
    
    def add_exponential_moving_averages(self, df, windows=[12, 26]):
        """
        Add Exponential Moving Averages (EMA)
        
        Args:
            df: DataFrame with OHLC data
            windows: List of window sizes for EMAs
            
        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        
        for window in windows:
            if len(df) >= window:
                ema = EMAIndicator(close=df['Close'], window=window)
                df[f'EMA_{window}'] = ema.ema_indicator()
        
        return df
    
    def add_rsi(self, df, window=14):
        """
        Add Relative Strength Index (RSI)
        RSI measures momentum, ranging from 0-100
        
        Args:
            df: DataFrame with OHLC data
            window: Period for RSI calculation
            
        Returns:
            DataFrame with RSI column
        """
        df = df.copy()
        
        if len(df) >= window:
            rsi = RSIIndicator(close=df['Close'], window=window)
            df['RSI'] = rsi.rsi()
        
        return df
    
    def add_macd(self, df, window_slow=26, window_fast=12, window_sign=9):
        """
        Add MACD (Moving Average Convergence Divergence) indicator
        MACD shows relationship between two moving averages
        
        Args:
            df: DataFrame with OHLC data
            window_slow: Slow EMA period
            window_fast: Fast EMA period
            window_sign: Signal line period
            
        Returns:
            DataFrame with MACD columns
        """
        df = df.copy()
        
        if len(df) >= window_slow:
            macd = MACD(close=df['Close'], 
                       window_slow=window_slow, 
                       window_fast=window_fast, 
                       window_sign=window_sign)
            
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()
        
        return df
    
    def add_bollinger_bands(self, df, window=20, window_dev=2):
        """
        Add Bollinger Bands
        Shows volatility and potential overbought/oversold conditions
        
        Args:
            df: DataFrame with OHLC data
            window: Period for moving average
            window_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Bands columns
        """
        df = df.copy()
        
        if len(df) >= window:
            bb = BollingerBands(close=df['Close'], 
                               window=window, 
                               window_dev=window_dev)
            
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Mid'] = bb.bollinger_mavg()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Width'] = bb.bollinger_wband()
        
        return df
    
    def add_stochastic_oscillator(self, df, window=14, smooth_window=3):
        """
        Add Stochastic Oscillator
        Compares closing price to price range over time
        
        Args:
            df: DataFrame with OHLC data
            window: Period for calculation
            smooth_window: Smoothing period
            
        Returns:
            DataFrame with Stochastic columns
        """
        df = df.copy()
        
        if len(df) >= window:
            stoch = StochasticOscillator(high=df['High'], 
                                        low=df['Low'], 
                                        close=df['Close'], 
                                        window=window, 
                                        smooth_window=smooth_window)
            
            df['Stoch'] = stoch.stoch()
            df['Stoch_Signal'] = stoch.stoch_signal()
        
        return df
    
    def add_atr(self, df, window=14):
        """
        Add Average True Range (ATR)
        Measures market volatility
        
        Args:
            df: DataFrame with OHLC data
            window: Period for ATR calculation
            
        Returns:
            DataFrame with ATR column
        """
        df = df.copy()
        
        if len(df) >= window:
            atr = AverageTrueRange(high=df['High'], 
                                  low=df['Low'], 
                                  close=df['Close'], 
                                  window=window)
            df['ATR'] = atr.average_true_range()
        
        return df
    
    def add_volume_features(self, df):
        """
        Add volume-based features
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        df = df.copy()
        
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            
            df['Volume_Change'] = df['Volume'].pct_change()
        
        return df
    
    def add_price_features(self, df):
        """
        Add price-based features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with price features
        """
        df = df.copy()
        
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = df['Close'].pct_change()
        
        df['Daily_Return'] = df['Close'].pct_change()
        
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        return df
    
    def add_lag_features(self, df, columns=['Close'], lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features for time series
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_statistics(self, df, column='Close', windows=[5, 10, 20]):
        """
        Add rolling statistical features
        
        Args:
            df: DataFrame
            column: Column to calculate statistics on
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()
        
        if column in df.columns:
            for window in windows:
                if len(df) >= window:
                    df[f'{column}_Rolling_Mean_{window}'] = df[column].rolling(window=window).mean()
                    df[f'{column}_Rolling_Std_{window}'] = df[column].rolling(window=window).std()
                    df[f'{column}_Rolling_Min_{window}'] = df[column].rolling(window=window).min()
                    df[f'{column}_Rolling_Max_{window}'] = df[column].rolling(window=window).max()
        
        return df
    
    def create_all_features(self, df):
        """
        Create all technical indicators and features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        df = self.add_moving_averages(df)
        df = self.add_exponential_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_stochastic_oscillator(df)
        df = self.add_atr(df)
        df = self.add_volume_features(df)
        df = self.add_price_features(df)
        
        df = df.dropna()
        
        return df
    
    def get_feature_list(self, df):
        """
        Get list of engineered features (excluding OHLCV and Date)
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        base_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        feature_columns = [col for col in df.columns if col not in base_columns]
        
        return feature_columns
