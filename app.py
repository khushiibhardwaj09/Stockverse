"""
Stock Market Prediction System
AI-powered stock price prediction with multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_handler import StockDataHandler, get_popular_stocks
from feature_engineering import FeatureEngineer
from models import LSTMModel, XGBoostModel, ProphetModel, ModelEvaluator, EnsembleModel
from backtesting import Backtester, PortfolioOptimizer

st.set_page_config(page_title="Stock Prediction System", layout="wide")

st.title("📈 AI-Powered Stock Market Prediction System")
st.markdown("Predict stock prices and trends using LSTM, XGBoost, and Prophet models")

if 'data_handler' not in st.session_state:
    st.session_state.data_handler = StockDataHandler()

if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

if 'featured_data' not in st.session_state:
    st.session_state.featured_data = None

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

tabs = st.tabs(["📊 Data Exploration", "🔧 Feature Engineering", "🤖 Model Training", "🔮 Predictions", "📈 Model Comparison", "🚀 Advanced Features"])

with tabs[0]:
    st.header("Data Exploration")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Stock Selection")
        
        popular_stocks = get_popular_stocks()
        selected_stock = st.selectbox("Select Stock Ticker", popular_stocks)
        
        custom_ticker = st.text_input("Or enter custom ticker")
        if custom_ticker:
            selected_stock = custom_ticker.upper()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        
        if st.button("Fetch Data", type="primary"):
            with st.spinner(f"Fetching data for {selected_stock}..."):
                df = st.session_state.data_handler.fetch_stock_data(
                    selected_stock,
                    start_date,
                    end_date
                )
                
                if df is not None and not df.empty:
                    df = st.session_state.data_handler.handle_missing_data(df)
                    st.session_state.stock_data = df
                    st.session_state.current_ticker = selected_stock
                    st.success(f"Successfully fetched {len(df)} days of data!")
                else:
                    st.error("Failed to fetch data. Please check the ticker symbol.")
    
    with col2:
        if st.session_state.stock_data is not None:
            df = st.session_state.stock_data
            
            st.subheader(f"{st.session_state.current_ticker} - Stock Data Overview")
            
            info = st.session_state.data_handler.get_company_info(st.session_state.current_ticker)
            if 'error' not in info:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Company", info['name'])
                col_b.metric("Sector", info['sector'])
                col_c.metric("Industry", info['industry'])
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
            
            if len(df) >= 2:
                change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100)
                col_b.metric("Change", f"{change_pct:.2f}%")
            else:
                col_b.metric("Change", "N/A")
            
            col_c.metric("High", f"${df['High'].max():.2f}")
            col_d.metric("Low", f"${df['Low'].min():.2f}")
            
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ))
            
            fig.update_layout(
                title=f"{st.session_state.current_ticker} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            
            fig_volume.update_layout(
                title="Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
            with st.expander("View Raw Data"):
                st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.header("Feature Engineering")
    
    if st.session_state.stock_data is None:
        st.warning("Please fetch stock data first in the Data Exploration tab.")
    else:
        st.markdown("### Technical Indicators")
        st.write("Add technical indicators to enhance prediction accuracy")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Select Indicators")
            
            add_ma = st.checkbox("Moving Averages (SMA)", value=True)
            add_ema = st.checkbox("Exponential MA (EMA)", value=True)
            add_rsi = st.checkbox("RSI", value=True)
            add_macd = st.checkbox("MACD", value=True)
            add_bb = st.checkbox("Bollinger Bands", value=True)
            add_stoch = st.checkbox("Stochastic Oscillator", value=False)
            add_atr = st.checkbox("ATR", value=False)
            add_volume = st.checkbox("Volume Features", value=True)
            add_price = st.checkbox("Price Features", value=True)
            
            if st.button("Generate Features", type="primary"):
                with st.spinner("Generating features..."):
                    df = st.session_state.stock_data.copy()
                    
                    if add_ma:
                        df = st.session_state.feature_engineer.add_moving_averages(df)
                    if add_ema:
                        df = st.session_state.feature_engineer.add_exponential_moving_averages(df)
                    if add_rsi:
                        df = st.session_state.feature_engineer.add_rsi(df)
                    if add_macd:
                        df = st.session_state.feature_engineer.add_macd(df)
                    if add_bb:
                        df = st.session_state.feature_engineer.add_bollinger_bands(df)
                    if add_stoch:
                        df = st.session_state.feature_engineer.add_stochastic_oscillator(df)
                    if add_atr:
                        df = st.session_state.feature_engineer.add_atr(df)
                    if add_volume:
                        df = st.session_state.feature_engineer.add_volume_features(df)
                    if add_price:
                        df = st.session_state.feature_engineer.add_price_features(df)
                    
                    df = df.dropna()
                    st.session_state.featured_data = df
                    st.success(f"Features generated! Total features: {len(df.columns)}")
        
        with col2:
            if st.session_state.featured_data is not None:
                df = st.session_state.featured_data
                
                st.subheader("Technical Indicators Visualization")
                
                indicator_view = st.selectbox(
                    "Select Indicator to View",
                    ["Price with Moving Averages", "RSI", "MACD", "Bollinger Bands", "Volume"]
                )
                
                fig = go.Figure()
                
                if indicator_view == "Price with Moving Averages":
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')))
                    
                    for col in df.columns:
                        if 'SMA' in col or 'EMA' in col:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[col], name=col, line=dict(width=1)))
                    
                    fig.update_layout(title="Price with Moving Averages", xaxis_title="Date", yaxis_title="Price")
                
                elif indicator_view == "RSI" and 'RSI' in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')))
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI")
                
                elif indicator_view == "MACD" and 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='orange')))
                    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Diff'], name='MACD Histogram'))
                    fig.update_layout(title="MACD Indicator", xaxis_title="Date", yaxis_title="Value")
                
                elif indicator_view == "Bollinger Bands" and 'BB_High' in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], name='Upper Band', line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Mid'], name='Middle Band', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], name='Lower Band', line=dict(color='green', dash='dash')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='black')))
                    fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
                
                elif indicator_view == "Volume":
                    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'))
                    if 'Volume_SMA_20' in df.columns:
                        fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume_SMA_20'], name='20-Day Avg', line=dict(color='red')))
                    fig.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Feature Data"):
                    st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.header("Model Training")
    
    if st.session_state.featured_data is None:
        st.warning("Please generate features first in the Feature Engineering tab.")
    else:
        model_choice = st.selectbox(
            "Select Model to Train",
            ["LSTM (Time Series)", "XGBoost (Classification)", "Prophet (Trend Forecasting)"]
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Training Configuration")
            
            train_split = st.slider("Training Data Split", 0.6, 0.9, 0.8, 0.05)
            
            if model_choice == "LSTM (Time Series)":
                st.markdown("### LSTM Parameters")
                sequence_length = st.slider("Sequence Length", 10, 120, 60, 10)
                lstm_units = st.slider("LSTM Units", 25, 200, 50, 25)
                dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
                epochs = st.slider("Epochs", 10, 100, 50, 10)
                
            elif model_choice == "XGBoost (Classification)":
                st.markdown("### XGBoost Parameters")
                n_estimators = st.slider("Number of Estimators", 50, 300, 100, 50)
                max_depth = st.slider("Max Depth", 3, 10, 5, 1)
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                
            elif model_choice == "Prophet (Trend Forecasting)":
                st.markdown("### Prophet Parameters")
                changepoint_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.005)
                seasonality_mode = st.selectbox("Seasonality Mode", ["multiplicative", "additive"])
            
            train_button = st.button("Train Model", type="primary")
        
        with col2:
            if train_button:
                df = st.session_state.featured_data.copy()
                
                with st.spinner(f"Training {model_choice}..."):
                    try:
                        if model_choice == "LSTM (Time Series)":
                            min_required = sequence_length + 50
                            if len(df) < min_required:
                                st.error(f"Insufficient data for LSTM training. Need at least {min_required} days of data, but only have {len(df)} days. Please fetch more historical data or reduce sequence length.")
                            else:
                                data_prep = st.session_state.data_handler.prepare_data_for_lstm(
                                    df,
                                    target_column='Close',
                                    sequence_length=sequence_length,
                                    train_split=train_split
                                )
                                
                                if len(data_prep['X_train']) == 0 or len(data_prep['X_test']) == 0:
                                    st.error(f"Not enough data after creating sequences. Please fetch more historical data or reduce sequence length.")
                                else:
                                    model = LSTMModel(
                                        sequence_length=sequence_length,
                                        units=lstm_units,
                                        dropout=dropout
                                    )
                                    
                                    history = model.train(
                                        data_prep['X_train'],
                                        data_prep['y_train'],
                                        epochs=epochs
                                    )
                                    
                                    results = model.evaluate(
                                        data_prep['X_test'],
                                        data_prep['y_test'],
                                        scaler=data_prep['scaler']
                                    )
                                    
                                    st.session_state.trained_models['LSTM'] = {
                                        'model': model,
                                        'results': results,
                                        'data_prep': data_prep,
                                        'type': 'regression'
                                    }
                                    
                                    st.success("LSTM model trained successfully!")
                                    
                                    col_a, col_b, col_c = st.columns(3)
                                    col_a.metric("RMSE", f"${results['RMSE']:.2f}")
                                    col_b.metric("MAE", f"${results['MAE']:.2f}")
                                    col_c.metric("MAPE", f"{results['MAPE']:.2f}%")
                                    
                                    fig_loss = go.Figure()
                                    fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                                    if 'val_loss' in history.history:
                                        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                                    fig_loss.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                                    st.plotly_chart(fig_loss, use_container_width=True)
                                    
                                    fig_pred = go.Figure()
                                    fig_pred.add_trace(go.Scatter(y=results['actual'], name='Actual', mode='lines'))
                                    fig_pred.add_trace(go.Scatter(y=results['predictions'], name='Predicted', mode='lines'))
                                    fig_pred.update_layout(title="Predictions vs Actual", xaxis_title="Time", yaxis_title="Price (USD)")
                                    st.plotly_chart(fig_pred, use_container_width=True)
                        
                        elif model_choice == "XGBoost (Classification)":
                            min_required = 50
                            if len(df) < min_required:
                                st.error(f"Insufficient data for XGBoost training. Need at least {min_required} days of data, but only have {len(df)} days. Please fetch more historical data.")
                            else:
                                feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
                                
                                if len(feature_cols) == 0:
                                    st.error("No features available for training. Please generate features in the Feature Engineering tab.")
                                else:
                                    data_prep = st.session_state.data_handler.prepare_data_for_classification(
                                        df,
                                        feature_columns=feature_cols,
                                        train_split=train_split
                                    )
                                    
                                    if len(data_prep['X_train']) == 0 or len(data_prep['X_test']) == 0:
                                        st.error(f"Not enough data after preparation. Please fetch more historical data.")
                                    else:
                                        model = XGBoostModel(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            learning_rate=learning_rate
                                        )
                                        
                                        model.train(data_prep['X_train'], data_prep['y_train'])
                                        
                                        results = model.evaluate(data_prep['X_test'], data_prep['y_test'])
                                        
                                        st.session_state.trained_models['XGBoost'] = {
                                            'model': model,
                                            'results': results,
                                            'data_prep': data_prep,
                                            'feature_cols': feature_cols,
                                            'type': 'classification'
                                        }
                                        
                                        st.success("XGBoost model trained successfully!")
                                        
                                        col_a, col_b = st.columns(2)
                                        col_a.metric("Accuracy", f"{results['Accuracy']*100:.2f}%")
                                        col_b.metric("F1 Score", f"{results['F1_Score']:.4f}")
                                        
                                        predictions_df = pd.DataFrame({
                                            'Actual': ['Down' if x == 0 else 'Up' for x in results['actual']],
                                            'Predicted': ['Down' if x == 0 else 'Up' for x in results['predictions']],
                                            'Confidence': results['probabilities'].max(axis=1)
                                        })
                                        
                                        st.subheader("Classification Report")
                                        report_df = pd.DataFrame(results['classification_report']).transpose()
                                        st.dataframe(report_df)
                                        
                                        feature_importance = model.get_feature_importance()
                                        top_features = sorted(zip(feature_cols, feature_importance), key=lambda x: x[1], reverse=True)[:10]
                                        
                                        fig_importance = go.Figure()
                                        fig_importance.add_trace(go.Bar(
                                            x=[f[1] for f in top_features],
                                            y=[f[0] for f in top_features],
                                            orientation='h'
                                        ))
                                        fig_importance.update_layout(title="Top 10 Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
                                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        elif model_choice == "Prophet (Trend Forecasting)":
                            min_required = 30
                            if len(df) < min_required:
                                st.error(f"Insufficient data for Prophet training. Need at least {min_required} days of data, but only have {len(df)} days. Please fetch more historical data.")
                            else:
                                prophet_df = st.session_state.data_handler.prepare_data_for_prophet(df)
                                
                                train_df, test_df = st.session_state.data_handler.get_train_test_split(prophet_df, train_split)
                                
                                if len(train_df) < 10 or len(test_df) < 5:
                                    st.error(f"Not enough data after split. Please fetch more historical data or adjust train/test split ratio.")
                                else:
                                    model = ProphetModel(
                                        changepoint_prior_scale=changepoint_scale,
                                        seasonality_mode=seasonality_mode
                                    )
                                    
                                    model.train(train_df)
                                    
                                    results = model.evaluate(train_df, test_df)
                                    
                                    st.session_state.trained_models['Prophet'] = {
                                        'model': model,
                                        'results': results,
                                        'train_df': train_df,
                                        'test_df': test_df,
                                        'type': 'regression'
                                    }
                                    
                                    st.success("Prophet model trained successfully!")
                                    
                                    col_a, col_b, col_c = st.columns(3)
                                    col_a.metric("RMSE", f"${results['RMSE']:.2f}")
                                    col_b.metric("MAE", f"${results['MAE']:.2f}")
                                    col_c.metric("MAPE", f"{results['MAPE']:.2f}%")
                                    
                                    fig_prophet = go.Figure()
                                    forecast = results['forecast']
                                    fig_prophet.add_trace(go.Scatter(x=test_df['ds'], y=results['actual'], name='Actual', mode='lines'))
                                    fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted', mode='lines'))
                                    fig_prophet.add_trace(go.Scatter(
                                        x=forecast['ds'],
                                        y=forecast['yhat_upper'],
                                        fill=None,
                                        mode='lines',
                                        line_color='rgba(0,100,200,0.2)',
                                        showlegend=False
                                    ))
                                    fig_prophet.add_trace(go.Scatter(
                                        x=forecast['ds'],
                                        y=forecast['yhat_lower'],
                                        fill='tonexty',
                                        mode='lines',
                                        line_color='rgba(0,100,200,0.2)',
                                        name='Confidence Interval'
                                    ))
                                    fig_prophet.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
                                    st.plotly_chart(fig_prophet, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

with tabs[3]:
    st.header("Make Predictions")
    
    if len(st.session_state.trained_models) == 0:
        st.warning("Please train at least one model first in the Model Training tab.")
    else:
        model_select = st.selectbox(
            "Select Trained Model",
            list(st.session_state.trained_models.keys())
        )
        
        model_info = st.session_state.trained_models[model_select]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Settings")
            
            if model_select == "Prophet":
                forecast_days = st.slider("Forecast Days into Future", 1, 90, 30)
                
                if st.button("Generate Forecast", type="primary"):
                    with st.spinner("Generating forecast..."):
                        model = model_info['model']
                        forecast = model.predict(periods=forecast_days)
                        
                        st.session_state.current_forecast = forecast
            else:
                st.info(f"Model Type: {model_info['type'].capitalize()}")
                st.write("Predictions are shown on the test set.")
        
        with col2:
            if model_select in ['LSTM', 'XGBoost']:
                results = model_info['results']
                
                st.subheader("Model Performance")
                
                if model_info['type'] == 'regression':
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("RMSE", f"${results['RMSE']:.2f}")
                    col_b.metric("MAE", f"${results['MAE']:.2f}")
                    col_c.metric("MAPE", f"{results['MAPE']:.2f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=results['actual'], name='Actual', mode='lines'))
                    fig.add_trace(go.Scatter(y=results['predictions'], name='Predicted', mode='lines'))
                    fig.update_layout(
                        title=f"{model_select} Predictions vs Actual",
                        xaxis_title="Time",
                        yaxis_title="Price (USD)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    col_a, col_b = st.columns(2)
                    col_a.metric("Accuracy", f"{results['Accuracy']*100:.2f}%")
                    col_b.metric("F1 Score", f"{results['F1_Score']:.4f}")
                    
                    predictions_df = pd.DataFrame({
                        'Actual Direction': ['Down ↓' if x == 0 else 'Up ↑' for x in results['actual']],
                        'Predicted Direction': ['Down ↓' if x == 0 else 'Up ↑' for x in results['predictions']],
                        'Confidence': [f"{x*100:.1f}%" for x in results['probabilities'].max(axis=1)]
                    })
                    
                    st.subheader("Recent Predictions")
                    st.dataframe(predictions_df.tail(20), use_container_width=True)
            
            elif model_select == "Prophet" and 'current_forecast' in st.session_state:
                forecast = st.session_state.current_forecast
                
                fig = go.Figure()
                
                if 'train_df' in model_info:
                    train_df = model_info['train_df']
                    fig.add_trace(go.Scatter(
                        x=train_df['ds'],
                        y=train_df['y'],
                        name='Historical Data',
                        mode='lines',
                        line=dict(color='blue')
                    ))
                
                future_forecast = forecast.tail(st.session_state.get('forecast_days', 30))
                
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat'],
                    name='Forecast',
                    mode='lines',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255,0,0,0.2)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_forecast['ds'],
                    y=future_forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255,0,0,0.2)',
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title="Prophet Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Forecast Data")
                forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                st.dataframe(forecast_display, use_container_width=True)

with tabs[4]:
    st.header("Model Comparison")
    
    if len(st.session_state.trained_models) < 2:
        st.info("Train at least 2 models to compare their performance.")
    else:
        st.subheader("Performance Metrics Comparison")
        
        comparison_data = []
        
        for model_name, model_info in st.session_state.trained_models.items():
            results = model_info['results']
            
            if model_info['type'] == 'regression':
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Regression',
                    'RMSE': f"${results['RMSE']:.2f}",
                    'MAE': f"${results['MAE']:.2f}",
                    'MAPE': f"{results['MAPE']:.2f}%",
                    'Accuracy': '-',
                    'F1 Score': '-'
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Classification',
                    'RMSE': '-',
                    'MAE': '-',
                    'MAPE': '-',
                    'Accuracy': f"{results['Accuracy']*100:.2f}%",
                    'F1 Score': f"{results['F1_Score']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        regression_models = {k: v for k, v in st.session_state.trained_models.items() if v['type'] == 'regression'}
        
        if len(regression_models) >= 2:
            st.subheader("Regression Models - Visual Comparison")
            
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for idx, (model_name, model_info) in enumerate(regression_models.items()):
                results = model_info['results']
                fig.add_trace(go.Scatter(
                    y=results['predictions'],
                    name=f'{model_name} Predictions',
                    mode='lines',
                    line=dict(color=colors[idx % len(colors)])
                ))
            
            first_model = list(regression_models.values())[0]
            fig.add_trace(go.Scatter(
                y=first_model['results']['actual'],
                name='Actual',
                mode='lines',
                line=dict(color='black', width=2)
            ))
            
            fig.update_layout(
                title="Model Predictions Comparison",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.header("Advanced Features")
    
    advanced_feature = st.selectbox(
        "Select Feature",
        ["Ensemble Model", "Backtesting", "Portfolio Optimization"]
    )
    
    if advanced_feature == "Ensemble Model":
        st.subheader("Ensemble Model - Combine Multiple Models")
        
        if len(st.session_state.trained_models) < 2:
            st.warning("Train at least 2 models to create an ensemble.")
        else:
            st.write("Create an ensemble that combines predictions from multiple models for improved accuracy.")
            
            available_models = list(st.session_state.trained_models.keys())
            selected_models = st.multiselect(
                "Select Models for Ensemble",
                available_models,
                default=available_models
            )
            
            weighting_method = st.selectbox(
                "Weighting Method",
                ["inverse_rmse", "inverse_mae", "equal"],
                help="inverse_rmse: Weight by inverse RMSE (better models get more weight)"
            )
            
            if st.button("Create Ensemble", type="primary") and len(selected_models) >= 2:
                with st.spinner("Creating ensemble..."):
                    try:
                        ensemble = EnsembleModel()
                        
                        for model_name in selected_models:
                            ensemble.add_model(model_name, st.session_state.trained_models[model_name])
                        
                        predictions = ensemble.predict(weighting_method=weighting_method)
                        
                        first_model = st.session_state.trained_models[selected_models[0]]
                        actual_values = first_model['results']['actual']
                        
                        ensemble_results = ensemble.evaluate(actual_values)
                        
                        st.session_state.trained_models['Ensemble'] = {
                            'model': ensemble,
                            'results': ensemble_results,
                            'type': 'regression',
                            'weights': predictions['weights']
                        }
                        
                        st.success("Ensemble model created successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RMSE", f"${ensemble_results['RMSE']:.2f}")
                        col2.metric("MAE", f"${ensemble_results['MAE']:.2f}")
                        col3.metric("MAPE", f"{ensemble_results['MAPE']:.2f}%")
                        
                        st.subheader("Model Weights")
                        weights_df = pd.DataFrame({
                            'Model': list(predictions['weights'].keys()),
                            'Weight': [f"{w*100:.2f}%" for w in predictions['weights'].values()]
                        })
                        st.dataframe(weights_df, use_container_width=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=ensemble_results['actual'], name='Actual', mode='lines', line=dict(color='black', width=2)))
                        fig.add_trace(go.Scatter(y=ensemble_results['predictions'], name='Ensemble Prediction', mode='lines', line=dict(color='purple', width=2)))
                        
                        for model_name in selected_models:
                            model_info = st.session_state.trained_models[model_name]
                            if model_info['type'] == 'regression':
                                fig.add_trace(go.Scatter(
                                    y=model_info['results']['predictions'][:len(ensemble_results['predictions'])],
                                    name=model_name,
                                    mode='lines',
                                    line=dict(width=1, dash='dash'),
                                    opacity=0.5
                                ))
                        
                        fig.update_layout(
                            title="Ensemble vs Individual Model Predictions",
                            xaxis_title="Time",
                            yaxis_title="Price (USD)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error creating ensemble: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    elif advanced_feature == "Backtesting":
        st.subheader("Backtesting Framework")
        
        if len(st.session_state.trained_models) == 0:
            st.warning("Train at least one model first.")
        else:
            st.write("Test your trading strategy using historical data to evaluate performance.")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                model_for_backtest = st.selectbox(
                    "Select Model for Backtesting",
                    list(st.session_state.trained_models.keys())
                )
                
                initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
                commission = st.slider("Commission Rate (%)", 0.0, 1.0, 0.1, 0.05) / 100
                
                strategy_type = st.selectbox(
                    "Strategy Type",
                    ["long_only", "long_short"],
                    help="long_only: Buy when prediction is higher, long_short: Buy/Sell based on prediction"
                )
                
                run_backtest = st.button("Run Backtest", type="primary")
            
            with col2:
                if run_backtest:
                    with st.spinner("Running backtest..."):
                        try:
                            model_info = st.session_state.trained_models[model_for_backtest]
                            
                            if st.session_state.featured_data is None:
                                st.error("No data available for backtesting.")
                            else:
                                df = st.session_state.featured_data.copy()
                                
                                if model_info['type'] == 'regression':
                                    predictions = model_info['results']['predictions']
                                else:
                                    predictions = model_info['results']['probabilities'][:, 1] * df['Close'].iloc[-len(model_info['results']['probabilities']):].mean()
                                
                                test_df = df.iloc[-len(predictions):].copy()
                                
                                backtester = Backtester(initial_capital=initial_capital, commission=commission)
                                results = backtester.run_strategy(test_df, predictions, strategy_type=strategy_type)
                                
                                st.success("Backtest completed!")
                                
                                col_a, col_b, col_c, col_d = st.columns(4)
                                col_a.metric("Final Value", f"${results['final_value']:.2f}")
                                col_b.metric("Total Return", f"{results['total_return']:.2f}%")
                                col_c.metric("Buy & Hold Return", f"{results['buy_hold_return']:.2f}%")
                                col_d.metric("Number of Trades", results['num_trades'])
                                
                                col_e, col_f = st.columns(2)
                                col_e.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                                col_f.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
                                
                                win_rate = backtester.calculate_win_rate()
                                st.metric("Win Rate", f"{win_rate:.2f}%")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=results['portfolio_values'],
                                    name='Portfolio Value',
                                    mode='lines',
                                    line=dict(color='green', width=2)
                                ))
                                fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
                                fig.update_layout(
                                    title="Portfolio Value Over Time",
                                    xaxis_title="Time",
                                    yaxis_title="Portfolio Value ($)",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if results['trades']:
                                    st.subheader("Trade History")
                                    trades_df = backtester.get_trade_summary()
                                    st.dataframe(trades_df, use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"Error during backtesting: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
    
    elif advanced_feature == "Portfolio Optimization":
        st.subheader("Portfolio Optimization")
        st.write("Optimize portfolio allocation across multiple stocks using Modern Portfolio Theory.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            num_stocks = st.number_input("Number of Stocks", min_value=2, max_value=10, value=3)
            
            tickers = []
            for i in range(num_stocks):
                ticker = st.text_input(f"Stock {i+1} Ticker", value=get_popular_stocks()[i] if i < len(get_popular_stocks()) else "", key=f"ticker_{i}")
                if ticker:
                    tickers.append(ticker.upper())
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Max Sharpe Ratio", "Min Volatility"]
            )
            
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0, 0.5) / 100
            
            optimize_button = st.button("Optimize Portfolio", type="primary")
        
        with col2:
            if optimize_button and len(tickers) >= 2:
                with st.spinner("Optimizing portfolio..."):
                    try:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365)
                        
                        prices_data = {}
                        for ticker in tickers:
                            df = st.session_state.data_handler.fetch_stock_data(ticker, start_date, end_date)
                            if df is not None and not df.empty:
                                prices_data[ticker] = df['Close']
                        
                        if len(prices_data) < 2:
                            st.error("Could not fetch data for enough stocks. Please check tickers.")
                        else:
                            prices_df = pd.DataFrame(prices_data)
                            prices_df = prices_df.dropna()
                            
                            optimizer = PortfolioOptimizer()
                            returns_df = optimizer.calculate_returns(prices_df)
                            
                            if optimization_method == "Max Sharpe Ratio":
                                result = optimizer.optimize_sharpe_ratio(returns_df, risk_free_rate=risk_free_rate)
                            else:
                                result = optimizer.optimize_min_volatility(returns_df)
                            
                            st.success("Portfolio optimized!")
                            
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
                            col_b.metric("Volatility", f"{result['volatility']*100:.2f}%")
                            col_c.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                            
                            st.subheader("Optimal Portfolio Weights")
                            weights_df = pd.DataFrame({
                                'Stock': list(prices_df.columns),
                                'Weight': [f"{w*100:.2f}%" for w in result['weights']],
                                'Allocation ($)': [f"${w*10000:.2f}" for w in result['weights']]
                            })
                            st.dataframe(weights_df, use_container_width=True)
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=list(prices_df.columns),
                                values=result['weights'],
                                hole=0.3
                            )])
                            fig.update_layout(title="Portfolio Allocation", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if 'all_results' in result:
                                fig_efficient = go.Figure()
                                fig_efficient.add_trace(go.Scatter(
                                    x=result['all_results'][1],
                                    y=result['all_results'][0],
                                    mode='markers',
                                    marker=dict(
                                        size=3,
                                        color=result['all_results'][2],
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title="Sharpe Ratio")
                                    ),
                                    name='Portfolios'
                                ))
                                
                                fig_efficient.add_trace(go.Scatter(
                                    x=[result['volatility']],
                                    y=[result['expected_return']],
                                    mode='markers',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    name='Optimal Portfolio'
                                ))
                                
                                fig_efficient.update_layout(
                                    title="Efficient Frontier",
                                    xaxis_title="Volatility (Risk)",
                                    yaxis_title="Expected Return",
                                    height=500
                                )
                                st.plotly_chart(fig_efficient, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error optimizing portfolio: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

st.sidebar.title("About")
st.sidebar.info("""
This Stock Market Prediction System uses advanced AI/ML models to forecast stock prices and trends.

**Models:**
- **LSTM**: Deep learning for time series
- **XGBoost**: Classification for price direction
- **Prophet**: Trend forecasting with seasonality

**Features:**
- Real-time data from Yahoo Finance
- Technical indicators (RSI, MACD, Bollinger Bands)
- Multiple model comparison
- Ensemble models
- Backtesting framework
- Portfolio optimization
- Interactive visualizations
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Guide")
st.sidebar.markdown("""
1. **Data Exploration**: Fetch stock data
2. **Feature Engineering**: Add technical indicators
3. **Model Training**: Train AI models
4. **Predictions**: Generate forecasts
5. **Comparison**: Compare model performance
6. **Advanced**: Ensemble, backtesting, portfolio optimization
""")
