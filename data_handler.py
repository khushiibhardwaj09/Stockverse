"""
Data Fetching and Preprocessing Module
Handles stock data retrieval, cleaning, and preparation for ML models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockDataHandler:
    """Handles stock data fetching and preprocessing"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.raw_data = None
        self.processed_data = None
        
    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
                
            df = df.reset_index()
            self.raw_data = df
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_company_info(self, ticker):
        """
        Get company information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')
            }
        except Exception as e:
            return {'name': ticker, 'error': str(e)}
    
    def handle_missing_data(self, df, method='ffill'):
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing data ('ffill', 'bfill', 'interpolate', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if method == 'ffill':
            df = df.fillna(method='ffill')
        elif method == 'bfill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        
        df = df.fillna(method='bfill')
        
        return df
    
    def handle_outliers(self, df, columns, method='clip', std_threshold=3):
        """
        Handle outliers in the data
        
        Args:
            df: DataFrame
            columns: List of columns to check for outliers
            method: 'clip' or 'remove'
            std_threshold: Number of standard deviations to consider as outlier
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - (std_threshold * std)
            upper_bound = mean + (std_threshold * std)
            
            if method == 'clip':
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def normalize_data(self, df, columns):
        """
        Normalize specified columns using MinMaxScaler
        
        Args:
            df: DataFrame
            columns: List of columns to normalize
            
        Returns:
            DataFrame with normalized columns and the scaler
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[col] = self.scaler.fit_transform(df[[col]])
        
        return df
    
    def create_sequences(self, data, sequence_length=60):
        """
        Create sequences for time series models (LSTM)
        
        Args:
            data: Array of values
            sequence_length: Length of each sequence
            
        Returns:
            X (sequences), y (targets)
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_lstm(self, df, target_column='Close', sequence_length=60, train_split=0.8):
        """
        Prepare data specifically for LSTM model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to predict
            sequence_length: Length of input sequences
            train_split: Proportion of data for training
            
        Returns:
            Dictionary with train and test data
        """
        df = df.copy()
        
        if target_column not in df.columns:
            raise ValueError(f"{target_column} not found in dataframe")
        
        data = df[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        train_size = int(len(scaled_data) * train_split)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - sequence_length:]
        
        X_train, y_train = self.create_sequences(train_data, sequence_length)
        X_test, y_test = self.create_sequences(test_data, sequence_length)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'train_size': train_size,
            'dates': df['Date'].iloc[train_size:].values
        }
    
    def prepare_data_for_classification(self, df, feature_columns, train_split=0.8):
        """
        Prepare data for classification models (XGBoost)
        Creates target as 1 (price goes up) or 0 (price goes down)
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature columns
            train_split: Proportion of data for training
            
        Returns:
            Dictionary with train and test data
        """
        df = df.copy()
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        X = df[feature_columns].values
        y = df['Target'].values
        
        train_size = int(len(X) * train_split)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_size': train_size,
            'dates': df['Date'].iloc[train_size:].values
        }
    
    def prepare_data_for_prophet(self, df, target_column='Close'):
        """
        Prepare data for Prophet model
        Prophet requires specific column names: 'ds' for date and 'y' for target
        
        Args:
            df: DataFrame with stock data
            target_column: Column to predict
            
        Returns:
            DataFrame formatted for Prophet
        """
        df = df.copy()
        
        prophet_df = pd.DataFrame({
            'ds': df['Date'],
            'y': df[target_column]
        })
        
        return prophet_df
    
    def get_train_test_split(self, df, train_split=0.8):
        """
        Split data into train and test sets based on time
        
        Args:
            df: DataFrame
            train_split: Proportion for training
            
        Returns:
            train_df, test_df
        """
        split_idx = int(len(df) * train_split)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df


def get_popular_stocks():
    """Returns a list of popular stock tickers"""
    return [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
        'META', 'NVDA', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'MA', 'DIS', 'NFLX',
        'PYPL', 'INTC', 'CSCO', 'PFE', 'KO'
    ]
