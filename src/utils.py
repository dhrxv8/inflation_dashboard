"""Utility functions for inflation dashboard"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def calculate_yoy_change(series: pd.Series) -> pd.Series:
    """Calculate year-over-year percentage change"""
    return series.pct_change(12) * 100

def calculate_mom_change(series: pd.Series) -> pd.Series:
    """Calculate month-over-month percentage change (annualized)"""
    return series.pct_change(1) * 100 * 12

def rolling_zscore(series: pd.Series, window: int = 24) -> pd.Series:
    """Calculate rolling z-score"""
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def format_date_range(start_date, end_date) -> str:
    """Format date range for display"""
    return f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}"
