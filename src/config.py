"""
Configuration settings for the inflation dashboard.
Handles both local development and Streamlit Cloud deployment.
"""

import os
from pathlib import Path

# Try Streamlit secrets first (for cloud deployment)
try:
    import streamlit as st
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except (ImportError, KeyError, FileNotFoundError):
    # Fall back to environment variable (for local development)
    from dotenv import load_dotenv
    load_dotenv()
    FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    # For demo purposes, provide a fallback message
    FRED_API_KEY = None
    print("⚠️ FRED_API_KEY not found - using sample data for demo")

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FORECASTS_DIR = DATA_DIR / "forecasts"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

# FRED API configuration
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Economic series to download
ECONOMIC_SERIES = {
    # Core inflation measures
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
    "CPILFESL": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average",
    "PCEPI": "Personal Consumption Expenditures: Chain-type Price Index",
    "PCEPILFE": "Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)",
    
    # Monetary policy indicators
    "FEDFUNDS": "Federal Funds Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "M2SL": "M2 Money Stock",
    
    # Labor market
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "All Employees, Total Nonfarm",
    "AHETPI": "Average Hourly Earnings of Production and Nonsupervisory Employees, Total Private",
    
    # Economic activity
    "GDP": "Gross Domestic Product",
    "INDPRO": "Industrial Production Index",
    "HOUST": "Housing Starts: Total: New Privately Owned Housing Units Started",
    
    # Commodity prices
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma",
    "GOLDAMGBD228NLBM": "Gold Fixing Price 3:00 P.M. (London time) in London Bullion Market, based in U.S. Dollars",
    
    # International
    "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
    "DEXJPUS": "Japan / U.S. Foreign Exchange Rate",
}

# Modeling configuration
FORECAST_HORIZON = 12  # months
TRAIN_TEST_SPLIT = 0.8

# Model parameters
MODEL_PARAMS = {
    "ARIMA": {
        "max_p": 5,
        "max_d": 2,
        "max_q": 5,
        "seasonal": True,
        "seasonal_periods": 12,
        "stepwise": True,
        "suppress_warnings": True,
        "error_action": "ignore"
    },
    "PROPHET": {
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05
    },
    "ENSEMBLE": {
        "weights": [0.5, 0.5]  # Equal weights for SARIMA and Prophet
    }
}

# Data download configuration
DOWNLOAD_CONFIG = {
    "observation_start": "2000-01-01",
    "observation_end": None,  # Will use current date
    "frequency": "m",  # Monthly
    "aggregation_method": "avg",
    "output_type": 1,  # Observations by real-time period
    "vintage_dates": None
}

# Feature engineering parameters
FEATURE_CONFIG = {
    "lags": [1, 3, 6, 12],
    "rolling_windows": [3, 6, 12],
    "difference_orders": [1, 12],
    "log_transform": ["M2SL", "GDP", "PAYEMS", "HOUST"]
}
