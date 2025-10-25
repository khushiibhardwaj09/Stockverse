# Stock Market Prediction System

## Overview

This is an AI-powered stock market prediction system that uses multiple machine learning models to forecast stock prices and trends. The application is built with Streamlit for the web interface and implements three different prediction models: LSTM (Long Short-Term Memory neural networks), XGBoost (gradient boosting), and Prophet (time series forecasting). Users can explore historical stock data, engineer technical indicators, train models, and generate predictions through an interactive dashboard.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application
- **UI Pattern**: Multi-tab interface with session state management
- **Tabs**:
  - Data Exploration: Stock selection and historical data visualization
  - Feature Engineering: Technical indicator creation and configuration
  - Model Training: ML model training interface
  - Predictions: Price forecasting and trend visualization
  - Model Comparison: Performance metrics across different models
- **State Management**: Uses `st.session_state` to persist data handlers, feature engineers, stock data, and trained models across user interactions
- **Visualization**: Plotly for interactive charts (graph_objects and express modules)

### Backend Architecture
- **Modular Design**: Separated into distinct functional modules:
  - `app.py`: Main Streamlit application and UI orchestration
  - `data_handler.py`: Data fetching and preprocessing logic
  - `feature_engineering.py`: Technical indicator calculation
  - `models.py`: Machine learning model implementations
- **Data Processing Pipeline**:
  1. Raw data fetching → 2. Feature engineering → 3. Model training → 4. Prediction generation
- **Design Pattern**: Object-oriented approach with dedicated classes for each major component (StockDataHandler, FeatureEngineer, LSTMModel, XGBoostModel, ProphetModel, ModelEvaluator)

### Machine Learning Architecture
- **Multi-Model Strategy**: Three complementary models for ensemble predictions
  - **LSTM**: Deep learning approach for sequential pattern recognition
    - Configurable sequence length, units, and dropout
    - Built with TensorFlow/Keras
    - Early stopping for preventing overfitting
  - **XGBoost**: Gradient boosting for feature-based predictions
    - Handles engineered technical indicators
  - **Prophet**: Facebook's time series forecasting library
    - Captures seasonality and trends
- **Feature Engineering**: Technical analysis indicators including:
  - Moving Averages (SMA with windows: 5, 10, 20, 50, 200)
  - Exponential Moving Averages (EMA: 12, 26)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Stochastic Oscillator
  - Bollinger Bands
  - Average True Range
- **Data Preprocessing**: MinMaxScaler for normalization before model training
- **Model Evaluation**: Comprehensive metrics including MSE, MAE, accuracy, F1-score, and classification reports

### Data Storage Solutions
- **In-Memory Storage**: No persistent database; all data stored in Streamlit session state
- **Data Sources**: Real-time fetching from Yahoo Finance API
- **Caching Strategy**: Session-based caching of fetched data, engineered features, and trained models to avoid redundant API calls and computations

## External Dependencies

### Third-Party Services
- **Yahoo Finance API** (`yfinance`): Primary data source for historical stock prices, company information, and market data
  - No authentication required
  - Free tier with rate limiting considerations

### Key Python Libraries
- **Web Framework**: 
  - `streamlit`: Web application framework
- **Data Processing**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
- **Visualization**:
  - `plotly`: Interactive charting (graph_objects and express)
- **Machine Learning**:
  - `tensorflow`/`keras`: LSTM neural network implementation
  - `xgboost`: Gradient boosting model
  - `prophet`: Facebook's time series forecasting
  - `scikit-learn`: Data preprocessing (MinMaxScaler) and evaluation metrics
- **Technical Analysis**:
  - `ta` (Technical Analysis Library): Pre-built technical indicators (MACD, RSI, Bollinger Bands, etc.)

### Infrastructure
- **Deployment Platform**: Designed for Replit hosting
- **Runtime**: Python 3.x environment
- **No Database**: Application operates entirely in-memory with no persistent storage layer
