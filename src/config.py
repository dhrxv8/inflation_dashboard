"""
Configuration file for Inflation Sensitivity Dashboard
Contains all constants, settings, and data series definitions
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get the project root directory (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FORECASTS_DIR = DATA_DIR / "forecasts"

# Other directories
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# ============================================================================
# FRED API CONFIGURATION
# ============================================================================

FRED_API_KEY = os.getenv('FRED_API_KEY')
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# ============================================================================
# ECONOMIC DATA SERIES
# ============================================================================

# Core inflation indicators
SERIES = {
    # Consumer Price Index
    "CPI": "CPIAUCSL",              # Consumer Price Index for All Urban Consumers
    "CORE_CPI": "CPILFESL",         # Core CPI (excludes food and energy)
    
    # Personal Consumption Expenditures
    "PCE": "PCEPI",                 # Personal Consumption Expenditures Price Index
    "CORE_PCE": "PCEPILFE",         # Core PCE Price Index
    
    # Producer Price Index
    "PPI": "PPIACO",                # Producer Price Index for All Commodities
    "PPI_CORE": "PPIFCG",           # PPI Final Demand less Food and Energy
    
    # Money Supply
    "M1": "M1SL",                   # M1 Money Stock
    "M2": "M2SL",                   # M2 Money Stock
    
    # Interest Rates
    "FED_FUNDS": "FEDFUNDS",        # Federal Funds Rate
    "10Y_TREASURY": "DGS10",        # 10-Year Treasury Constant Maturity Rate
    "2Y_TREASURY": "DGS2",          # 2-Year Treasury Constant Maturity Rate
    
    # Employment
    "UNEMPLOYMENT": "UNRATE",        # Unemployment Rate
    "EMPLOYMENT": "PAYEMS",          # Total Nonfarm Payrolls
    
    # Economic Activity
    "GDP": "GDP",                    # Gross Domestic Product
    "INDUSTRIAL_PRODUCTION": "INDPRO", # Industrial Production Index
    
    # Commodity Prices
    "OIL_PRICE": "DCOILWTICO",      # Crude Oil Prices: West Texas Intermediate
    "GOLD_PRICE": "GOLDAMGBD228NLBM", # Gold Price
    
    # Housing
    "HOUSING_STARTS": "HOUST",       # Housing Starts
    "MORTGAGE_RATE": "MORTGAGE30US", # 30-Year Fixed Rate Mortgage Average
    
    # International
    "USD_INDEX": "DTWEXBGS",        # Trade Weighted U.S. Dollar Index
}

# Series descriptions for dashboard display
SERIES_DESCRIPTIONS = {
    "CPI": "Consumer Price Index - Primary inflation measure",
    "CORE_CPI": "Core CPI - Excludes volatile food and energy prices",
    "PCE": "Personal Consumption Expenditures - Fed's preferred inflation measure",
    "CORE_PCE": "Core PCE - Fed's primary inflation target",
    "PPI": "Producer Price Index - Measures wholesale inflation",
    "PPI_CORE": "Core PPI - Producer prices excluding food and energy",
    "M1": "M1 Money Supply - Most liquid money measures",
    "M2": "M2 Money Supply - Broader money supply measure",
    "FED_FUNDS": "Federal Funds Rate - Key monetary policy tool",
    "10Y_TREASURY": "10-Year Treasury Rate - Long-term interest rate benchmark",
    "2Y_TREASURY": "2-Year Treasury Rate - Short-term rate indicator",
    "UNEMPLOYMENT": "Unemployment Rate - Labor market health indicator",
    "EMPLOYMENT": "Total Employment - Job market strength",
    "GDP": "Gross Domestic Product - Overall economic output",
    "INDUSTRIAL_PRODUCTION": "Industrial Production - Manufacturing sector health",
    "OIL_PRICE": "WTI Oil Price - Key commodity affecting inflation",
    "GOLD_PRICE": "Gold Price - Inflation hedge and market sentiment",
    "HOUSING_STARTS": "Housing Starts - Real estate market activity",
    "MORTGAGE_RATE": "30-Year Mortgage Rate - Housing affordability indicator",
    "USD_INDEX": "US Dollar Index - Currency strength measure",
}

# ============================================================================
# MODELING AND FORECASTING SETTINGS
# ============================================================================

# Forecast horizon in months
FORECAST_HORIZON = 12

# Model parameters
MODEL_PARAMS = {
    "ARIMA": {
        "max_p": 5,
        "max_d": 2,
        "max_q": 5,
        "seasonal": True,
        "seasonal_periods": 12
    },
    "PROPHET": {
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05
    },
    "ENSEMBLE": {
        "models": ["ARIMA", "PROPHET", "LINEAR_TREND"],
        "weights": [0.4, 0.4, 0.2]
    }
}

# Data collection settings
DATA_SETTINGS = {
    "DEFAULT_START_DATE": "2000-01-01",
    "UPDATE_FREQUENCY": "daily",
    "CACHE_DURATION_HOURS": 6,
    "MIN_DATA_POINTS": 24  # Minimum months of data required
}

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Streamlit app settings
APP_CONFIG = {
    "page_title": "Inflation Sensitivity Dashboard",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Chart settings
CHART_CONFIG = {
    "default_height": 500,
    "color_palette": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ],
    "inflation_target": 2.0,  # Fed's inflation target (%)
    "theme": "plotly_white"
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# What-if scenario settings
SIMULATION_PARAMS = {
    "fed_funds_range": (0.0, 8.0),      # Federal funds rate range
    "m2_growth_range": (-5.0, 20.0),    # M2 money supply growth range (%)
    "oil_price_range": (30.0, 150.0),   # Oil price range ($)
    "unemployment_range": (2.0, 12.0),   # Unemployment rate range (%)
    "gdp_growth_range": (-3.0, 6.0),    # GDP growth range (%)
}

# Scenario presets
SCENARIO_PRESETS = {
    "Current Conditions": {
        "description": "Based on latest available data",
        "use_latest_data": True
    },
    "Recession": {
        "description": "Economic downturn scenario",
        "fed_funds": 0.5,
        "unemployment": 8.0,
        "gdp_growth": -2.0,
        "oil_price": 60.0
    },
    "High Growth": {
        "description": "Strong economic expansion",
        "fed_funds": 4.5,
        "unemployment": 3.5,
        "gdp_growth": 4.0,
        "oil_price": 90.0
    },
    "Supply Shock": {
        "description": "Oil/commodity price surge",
        "fed_funds": 3.0,
        "unemployment": 5.0,
        "gdp_growth": 1.0,
        "oil_price": 120.0
    }
}

# ============================================================================
# ERROR HANDLING AND LOGGING
# ============================================================================

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "dashboard.log"
}

# API rate limiting
API_LIMITS = {
    "requests_per_minute": 30,
    "retry_attempts": 3,
    "retry_delay": 1.0  # seconds
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

# Data quality checks
VALIDATION_RULES = {
    "min_data_points": 12,
    "max_missing_percentage": 0.1,  # 10% missing data allowed
    "outlier_threshold": 3.0,       # Standard deviations
    "acceptable_date_range": ("1990-01-01", "2030-12-31")
}