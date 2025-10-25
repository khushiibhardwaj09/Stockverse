"""
Machine Learning Models Module
Implements LSTM, XGBoost, and Prophet models for stock prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import xgboost as xgb

from prophet import Prophet


class LSTMModel:
    """LSTM Neural Network for time series prediction"""
    
    def __init__(self, sequence_length=60, units=50, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of time steps to look back
            units: Number of LSTM units
            dropout: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """
        Build LSTM architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            
            LSTM(units=self.units, return_sequences=True),
            Dropout(self.dropout),
            
            LSTM(units=self.units, return_sequences=False),
            Dropout(self.dropout),
            
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data proportion
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test, scaler=None):
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            scaler: Scaler used for normalization (to inverse transform)
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        else:
            y_test_actual = y_test.reshape(-1, 1)
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)
        
        mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'predictions': predictions.flatten(),
            'actual': y_test_actual.flatten()
        }


class XGBoostModel:
    """XGBoost for stock price direction classification (up/down)"""
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self):
        """Build XGBoost classifier"""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        return {
            'Accuracy': accuracy,
            'F1_Score': f1,
            'predictions': predictions,
            'probabilities': probabilities,
            'actual': y_test,
            'classification_report': classification_report(y_test, predictions, output_dict=True)
        }
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            Feature importance array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.feature_importances_


class ProphetModel:
    """Prophet model for time series forecasting"""
    
    def __init__(self, changepoint_prior_scale=0.05, seasonality_mode='multiplicative'):
        """
        Initialize Prophet model
        
        Args:
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            seasonality_mode: 'additive' or 'multiplicative'
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model = None
        
    def build_model(self):
        """Build Prophet model"""
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        return self.model
    
    def train(self, df):
        """
        Train the Prophet model
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(df)
        
        return self.model
    
    def predict(self, periods=30):
        """
        Make future predictions
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast
    
    def evaluate(self, train_df, test_df):
        """
        Evaluate model performance
        
        Args:
            train_df: Training data (already used to fit model)
            test_df: Test data with 'ds' and 'y' columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        future = pd.DataFrame({'ds': test_df['ds']})
        forecast = self.model.predict(future)
        
        predictions = forecast['yhat'].values
        actual = test_df['y'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'predictions': predictions,
            'actual': actual,
            'forecast': forecast
        }
    
    def get_components(self):
        """
        Get forecast components (trend, seasonality)
        
        Returns:
            Components plot data
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model


class ModelEvaluator:
    """Utility class for comparing and evaluating multiple models"""
    
    @staticmethod
    def calculate_metrics(actual, predicted):
        """
        Calculate regression metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    @staticmethod
    def calculate_classification_metrics(actual, predicted):
        """
        Calculate classification metrics
        
        Args:
            actual: Actual labels
            predicted: Predicted labels
            
        Returns:
            Dictionary with metrics
        """
        accuracy = accuracy_score(actual, predicted)
        f1 = f1_score(actual, predicted)
        
        return {
            'Accuracy': accuracy,
            'F1_Score': f1
        }
    
    @staticmethod
    def compare_models(results_dict):
        """
        Compare multiple model results
        
        Args:
            results_dict: Dictionary with model names as keys and results as values
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            row.update(results)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


class EnsembleModel:
    """Ensemble model combining LSTM, XGBoost, and Prophet predictions"""
    
    def __init__(self):
        """Initialize ensemble model"""
        self.models = {}
        self.weights = {}
        
    def add_model(self, name, model_info):
        """
        Add a trained model to the ensemble (only regression models allowed)
        
        Args:
            name: Model name ('LSTM', 'Prophet', or 'Ensemble')
            model_info: Dictionary containing model and results
            
        Raises:
            ValueError: If trying to add a classification model
        """
        if model_info['type'] != 'regression':
            raise ValueError(f"Only regression models can be added to ensemble. {name} is a {model_info['type']} model.")
        
        self.models[name] = model_info
    
    def calculate_weights(self, method='inverse_rmse'):
        """
        Calculate weights for each model based on performance
        
        Args:
            method: Weighting method ('equal', 'inverse_rmse', 'inverse_mae')
            
        Returns:
            Dictionary of weights
        """
        if method == 'equal':
            num_models = len(self.models)
            self.weights = {name: 1.0 / num_models for name in self.models.keys()}
        
        elif method == 'inverse_rmse':
            rmse_values = {}
            
            for name, model_info in self.models.items():
                if model_info['type'] == 'regression':
                    rmse_values[name] = model_info['results'].get('RMSE', 1.0)
                else:
                    rmse_values[name] = 1.0 - model_info['results'].get('Accuracy', 0.5)
            
            inverse_rmse = {name: 1.0 / (rmse + 1e-6) for name, rmse in rmse_values.items()}
            total = sum(inverse_rmse.values())
            self.weights = {name: val / total for name, val in inverse_rmse.items()}
        
        elif method == 'inverse_mae':
            mae_values = {}
            
            for name, model_info in self.models.items():
                if model_info['type'] == 'regression':
                    mae_values[name] = model_info['results'].get('MAE', 1.0)
                else:
                    mae_values[name] = 1.0 - model_info['results'].get('Accuracy', 0.5)
            
            inverse_mae = {name: 1.0 / (mae + 1e-6) for name, mae in mae_values.items()}
            total = sum(inverse_mae.values())
            self.weights = {name: val / total for name, val in inverse_mae.items()}
        
        return self.weights
    
    def predict(self, weighting_method='inverse_rmse'):
        """
        Generate ensemble predictions
        
        Args:
            weighting_method: Method for calculating weights
            
        Returns:
            Dictionary with ensemble predictions and individual model predictions
        """
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        self.calculate_weights(weighting_method)
        
        predictions_dict = {}
        min_length = float('inf')
        
        for name, model_info in self.models.items():
            if model_info['type'] == 'regression':
                preds = model_info['results']['predictions']
            else:
                preds = model_info['results']['probabilities'][:, 1]
            
            predictions_dict[name] = preds
            min_length = min(min_length, len(preds))
        
        for name in predictions_dict:
            predictions_dict[name] = predictions_dict[name][:min_length]
        
        ensemble_pred = np.zeros(min_length)
        
        for name, preds in predictions_dict.items():
            ensemble_pred += self.weights[name] * preds
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions_dict,
            'weights': self.weights
        }
    
    def evaluate(self, actual_values):
        """
        Evaluate ensemble performance
        
        Args:
            actual_values: Actual values to compare against
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict()
        ensemble_pred = predictions['ensemble']
        
        min_length = min(len(ensemble_pred), len(actual_values))
        ensemble_pred = ensemble_pred[:min_length]
        actual_values = actual_values[:min_length]
        
        rmse = np.sqrt(mean_squared_error(actual_values, ensemble_pred))
        mae = mean_absolute_error(actual_values, ensemble_pred)
        mape = np.mean(np.abs((actual_values - ensemble_pred) / actual_values)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'predictions': ensemble_pred,
            'actual': actual_values
        }
