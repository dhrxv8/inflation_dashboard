"""
Feature Engineering Module for Inflation Dashboard
Transforms raw economic data into meaningful features for modeling and analysis.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import argparse

import pandas as pd
import numpy as np
from scipy import stats

from src.config import (
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR, 
    SERIES,
    VALIDATION_RULES
)
from src.data_ingest import get_all

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class FeatureEngineer:
    """
    Handles feature engineering for economic time series data.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        # Create processed data directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Feature metadata for tracking
        self.feature_metadata = {}
    
    def add_pct_change(self, df: pd.DataFrame, periods: int = 12, suffix: str = None) -> pd.DataFrame:
        """
        Add percentage change calculations (inflation rates).
        
        Args:
            df: DataFrame with time series data
            periods: Number of periods for percentage change (12 = YoY, 1 = MoM)
            suffix: Suffix for new column names (auto-generated if None)
            
        Returns:
            DataFrame with additional percentage change columns
        """
        if suffix is None:
            if periods == 12:
                suffix = "_yoy"
            elif periods == 1:
                suffix = "_mom"
            else:
                suffix = f"_{periods}p"
        
        df_result = df.copy()
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Calculate percentage change
                pct_change_col = f"{col}{suffix}"
                df_result[pct_change_col] = df[col].pct_change(periods=periods) * 100
                
                # Track metadata
                self.feature_metadata[pct_change_col] = {
                    'source': col,
                    'transformation': f'pct_change_{periods}',
                    'periods': periods,
                    'description': f'{periods}-period percentage change of {col}'
                }
                
                logger.info(f"Added {pct_change_col} (periods={periods})")
        
        return df_result
    
    def add_real_values(self, df: pd.DataFrame, nominal_col: str, deflator_col: str) -> pd.DataFrame:
        """
        Convert nominal values to real (inflation-adjusted) values.
        
        Args:
            df: DataFrame containing the data
            nominal_col: Column name for nominal values
            deflator_col: Column name for price deflator (e.g., CPI)
            
        Returns:
            DataFrame with real value column added
        """
        df_result = df.copy()
        real_col = f"{nominal_col}_real"
        
        if nominal_col not in df.columns or deflator_col not in df.columns:
            logger.warning(f"Cannot create {real_col}: missing {nominal_col} or {deflator_col}")
            return df_result
        
        # Calculate real values (nominal / price_index * 100)
        # Using first available deflator value as base (=100)
        base_deflator = df[deflator_col].dropna().iloc[0]
        df_result[real_col] = (df[nominal_col] / df[deflator_col]) * base_deflator
        
        # Track metadata
        self.feature_metadata[real_col] = {
            'source': [nominal_col, deflator_col],
            'transformation': 'real_adjustment',
            'base_period': df[deflator_col].dropna().index[0],
            'description': f'Real (inflation-adjusted) {nominal_col} using {deflator_col}'
        }
        
        logger.info(f"Added {real_col} using {deflator_col} as deflator")
        return df_result
    
    def real_m2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate real M2 money supply (adjusted for inflation).
        
        Args:
            df: DataFrame containing M2 and CPI data
            
        Returns:
            DataFrame with real M2 column added
        """
        return self.add_real_values(df, 'M2SL', 'CPIAUCSL')
    
    def add_lags(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features for specified columns.
        
        Args:
            df: DataFrame with time series data
            columns: List of column names to create lags for
            lags: List of lag periods (e.g., [1, 3, 6, 12])
            
        Returns:
            DataFrame with lagged features added
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping lags")
                continue
                
            for lag in lags:
                lag_col = f"{col}_lag{lag}"
                df_result[lag_col] = df[col].shift(lag)
                
                # Track metadata
                self.feature_metadata[lag_col] = {
                    'source': col,
                    'transformation': 'lag',
                    'lag_periods': lag,
                    'description': f'{lag}-period lag of {col}'
                }
        
        logger.info(f"Added lags {lags} for columns {columns}")
        return df_result
    
    def add_rolling_stats(self, df: pd.DataFrame, columns: List[str], 
                         windows: List[int], stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Add rolling window statistics.
        
        Args:
            df: DataFrame with time series data
            columns: List of column names to calculate rolling stats for
            windows: List of window sizes (e.g., [3, 6, 12])
            stats: List of statistics to calculate ('mean', 'std', 'min', 'max')
            
        Returns:
            DataFrame with rolling statistics added
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping rolling stats")
                continue
                
            for window in windows:
                for stat in stats:
                    stat_col = f"{col}_{stat}{window}"
                    
                    if stat == 'mean':
                        df_result[stat_col] = df[col].rolling(window=window).mean()
                    elif stat == 'std':
                        df_result[stat_col] = df[col].rolling(window=window).std()
                    elif stat == 'min':
                        df_result[stat_col] = df[col].rolling(window=window).min()
                    elif stat == 'max':
                        df_result[stat_col] = df[col].rolling(window=window).max()
                    
                    # Track metadata
                    self.feature_metadata[stat_col] = {
                        'source': col,
                        'transformation': f'rolling_{stat}',
                        'window': window,
                        'description': f'{window}-period rolling {stat} of {col}'
                    }
        
        logger.info(f"Added rolling {stats} for windows {windows}")
        return df_result
    
    def add_z_scores(self, df: pd.DataFrame, columns: List[str], window: int = 60) -> pd.DataFrame:
        """
        Add rolling z-scores (standardized values).
        
        Args:
            df: DataFrame with time series data
            columns: List of column names to calculate z-scores for
            window: Rolling window for calculating mean and std
            
        Returns:
            DataFrame with z-score columns added
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping z-score")
                continue
            
            zscore_col = f"{col}_zscore"
            
            # Calculate rolling z-score
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            df_result[zscore_col] = (df[col] - rolling_mean) / rolling_std
            
            # Track metadata
            self.feature_metadata[zscore_col] = {
                'source': col,
                'transformation': 'rolling_zscore',
                'window': window,
                'description': f'{window}-period rolling z-score of {col}'
            }
        
        logger.info(f"Added z-scores for {columns} (window={window})")
        return df_result
    
    def add_volatility(self, df: pd.DataFrame, columns: List[str], window: int = 12) -> pd.DataFrame:
        """
        Add volatility measures (rolling standard deviation of returns).
        
        Args:
            df: DataFrame with time series data
            columns: List of column names to calculate volatility for
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility columns added
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping volatility")
                continue
            
            vol_col = f"{col}_vol{window}"
            
            # Calculate returns and their rolling volatility
            returns = df[col].pct_change()
            df_result[vol_col] = returns.rolling(window=window).std() * np.sqrt(12)  # Annualized
            
            # Track metadata
            self.feature_metadata[vol_col] = {
                'source': col,
                'transformation': 'volatility',
                'window': window,
                'description': f'{window}-period annualized volatility of {col}'
            }
        
        logger.info(f"Added volatility for {columns} (window={window})")
        return df_result
    
    def add_trend_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Add trend-based features (momentum, acceleration).
        
        Args:
            df: DataFrame with time series data
            columns: List of column names to calculate trends for
            
        Returns:
            DataFrame with trend features added
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping trends")
                continue
            
            # Momentum (3-month vs 12-month change)
            momentum_col = f"{col}_momentum"
            change_3m = df[col].pct_change(3)
            change_12m = df[col].pct_change(12)
            df_result[momentum_col] = change_3m - change_12m
            
            # Acceleration (change in rate of change)
            accel_col = f"{col}_accel"
            df_result[accel_col] = df[col].pct_change(1).diff()
            
            # Track metadata
            self.feature_metadata[momentum_col] = {
                'source': col,
                'transformation': 'momentum',
                'description': f'Momentum (3m vs 12m change) of {col}'
            }
            
            self.feature_metadata[accel_col] = {
                'source': col,
                'transformation': 'acceleration',
                'description': f'Acceleration (change in change) of {col}'
            }
        
        logger.info(f"Added trend features for {columns}")
        return df_result
    
    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal/cyclical features based on date index.
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with seasonal features added
        """
        df_result = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, cannot add seasonal features")
            return df_result
        
        # Month indicators
        df_result['month'] = df.index.month
        df_result['quarter'] = df.index.quarter
        
        # Cyclical encoding of month (preserves cyclical nature)
        df_result['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df_result['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Year and decade trends
        df_result['year'] = df.index.year
        df_result['years_since_2000'] = df.index.year - 2000
        
        # Day of year for daily data
        if len(df) > 365:  # Likely daily data
            df_result['day_of_year'] = df.index.dayofyear
            df_result['is_month_end'] = df.index.is_month_end.astype(int)
        
        logger.info("Added seasonal features")
        return df_result
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and flag outliers in the data.
        
        Args:
            df: DataFrame with time series data
            columns: List of column names to check for outliers
            method: Method for outlier detection ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags added
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            outlier_col = f"{col}_outlier"
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                # Map z-scores back to original dataframe
                outliers = pd.Series(False, index=df.index)
                outliers.loc[df[col].dropna().index] = z_scores > threshold
                df_result[outlier_col] = outliers
                
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_result[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        logger.info(f"Added outlier detection for {columns} using {method}")
        return df_result
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality against defined rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'date_range': (df.index.min(), df.index.max()) if len(df) > 0 else (None, None),
            'columns': list(df.columns),
            'issues': []
        }
        
        # Check minimum data points
        min_points = VALIDATION_RULES['min_data_points']
        if len(df) < min_points:
            validation_results['issues'].append(f"Insufficient data: {len(df)} < {min_points}")
        
        # Check missing data percentage
        max_missing_pct = VALIDATION_RULES['max_missing_percentage']
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > max_missing_pct:
                validation_results['issues'].append(
                    f"High missing data in {col}: {missing_pct:.1%} > {max_missing_pct:.1%}"
                )
        
        # Check for outliers
        outlier_threshold = VALIDATION_RULES['outlier_threshold']
        for col in df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_count = (z_scores > outlier_threshold).sum()
            if outlier_count > 0:
                validation_results['issues'].append(
                    f"Outliers detected in {col}: {outlier_count} points"
                )
        
        logger.info(f"Data validation complete: {len(validation_results['issues'])} issues found")
        return validation_results


def build_feature_table() -> pd.DataFrame:
    """
    Main pipeline to build comprehensive feature table from raw data.
    
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load raw data
    logger.info("Loading raw economic data...")
    raw_data_files = list(RAW_DATA_DIR.glob("*.csv"))
    
    if not raw_data_files:
        logger.error("No raw data files found. Run data ingestion first.")
        raise FileNotFoundError("No raw data files found in data/raw/")
    
    # Load and combine all series
    combined_data = pd.DataFrame()
    
    for file_path in raw_data_files:
        try:
            series_name = file_path.stem  # Filename without extension
            series_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if len(series_data) > 0:
                combined_data = pd.concat([combined_data, series_data], axis=1, sort=True)
                logger.info(f"Loaded {series_name}: {len(series_data)} observations")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    if combined_data.empty:
        raise ValueError("No valid data could be loaded")
    
    logger.info(f"Combined dataset: {len(combined_data)} rows, {len(combined_data.columns)} columns")
    
    # Feature Engineering Pipeline
    logger.info("Applying feature engineering transformations...")
    
    # 1. Add percentage changes (inflation rates)
    combined_data = engineer.add_pct_change(combined_data, periods=12)  # YoY
    combined_data = engineer.add_pct_change(combined_data, periods=1)   # MoM
    
    # 2. Add real (inflation-adjusted) values
    if 'M2SL' in combined_data.columns and 'CPIAUCSL' in combined_data.columns:
        combined_data = engineer.real_m2(combined_data)
    
    # Add other real values
    if 'PAYEMS' in combined_data.columns and 'CPIAUCSL' in combined_data.columns:
        combined_data = engineer.add_real_values(combined_data, 'PAYEMS', 'CPIAUCSL')
    
    # 3. Add lagged features for key indicators
    key_indicators = ['CPIAUCSL', 'FEDFUNDS', 'UNRATE', 'M2SL']
    available_indicators = [col for col in key_indicators if col in combined_data.columns]
    combined_data = engineer.add_lags(combined_data, available_indicators, [1, 3, 6, 12])
    
    # 4. Add rolling statistics
    combined_data = engineer.add_rolling_stats(
        combined_data, available_indicators, [3, 6, 12], ['mean', 'std']
    )
    
    # 5. Add z-scores for key indicators
    combined_data = engineer.add_z_scores(combined_data, available_indicators)
    
    # 6. Add volatility measures
    combined_data = engineer.add_volatility(combined_data, available_indicators)
    
    # 7. Add trend features
    combined_data = engineer.add_trend_features(combined_data, available_indicators)
    
    # 8. Add seasonal features
    combined_data = engineer.add_seasonal_features(combined_data)
    
    # 9. Add outlier detection
    combined_data = engineer.detect_outliers(combined_data, available_indicators)
    
    # Data quality validation
    validation_results = engineer.validate_data_quality(combined_data)
    logger.info(f"Final dataset: {validation_results['total_rows']} rows, "
               f"{len(validation_results['columns'])} features")
    
    if validation_results['issues']:
        logger.warning(f"Data quality issues: {validation_results['issues']}")
    
    # Save processed data
    output_path = PROCESSED_DATA_DIR / "feature_table.csv"
    combined_data.to_csv(output_path)
    logger.info(f"Saved feature table to {output_path}")
    
    # Save feature metadata
    metadata_path = PROCESSED_DATA_DIR / "feature_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metadata = {}
        for key, value in engineer.feature_metadata.items():
            serializable_metadata[key] = {}
            for k, v in value.items():
                if isinstance(v, (pd.Timestamp, np.datetime64)):
                    serializable_metadata[key][k] = str(v)
                elif isinstance(v, (np.integer, np.floating)):
                    serializable_metadata[key][k] = v.item()
                else:
                    serializable_metadata[key][k] = v
        
        json.dump(serializable_metadata, f, indent=2)
    logger.info(f"Saved feature metadata to {metadata_path}")
    
    return combined_data


def main():
    """Command-line interface for feature engineering."""
    parser = argparse.ArgumentParser(description="Feature engineering for economic data")
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="Rebuild feature table from scratch"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation checks only"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        # Load existing feature table and validate
        feature_path = PROCESSED_DATA_DIR / "feature_table.csv"
        if feature_path.exists():
            df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
            engineer = FeatureEngineer()
            results = engineer.validate_data_quality(df)
            print(f"Validation results: {results}")
        else:
            print("No feature table found. Run feature engineering first.")
    else:
        # Build feature table
        try:
            feature_table = build_feature_table()
            print(f"✅ Feature engineering complete!")
            print(f"   Features created: {len(feature_table.columns)}")
            print(f"   Date range: {feature_table.index.min()} to {feature_table.index.max()}")
            print(f"   Observations: {len(feature_table)}")
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            raise


if __name__ == "__main__":
    main()