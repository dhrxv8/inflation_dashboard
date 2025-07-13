"""
FRED Data Ingestion Module
Fetches economic data from Federal Reserve Economic Database (FRED)
with intelligent caching and ETag support for efficient updates.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional
import argparse
import sys

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    FRED_API_KEY, 
    FRED_BASE_URL, 
    RAW_DATA_DIR, 
    SERIES, 
    API_LIMITS,
    DATA_SETTINGS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDDataFetcher:
    """
    Handles fetching economic data from FRED API with caching and rate limiting.
    """
    
    def __init__(self, api_key: str = FRED_API_KEY):
        """
        Initialize the FRED data fetcher.
        
        Args:
            api_key: FRED API key for authentication
        """
        if not api_key:
            raise ValueError("FRED API key is required. Set FRED_API_KEY environment variable.")
        
        self.api_key = api_key
        self.base_url = FRED_BASE_URL
        self.session = self._create_session()
        
        # Create data directories if they don't exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry strategy and rate limiting.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=API_LIMITS["retry_attempts"],
            backoff_factor=API_LIMITS["retry_delay"],
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _rate_limit(self) -> None:
        """
        Implement rate limiting to respect FRED API limits.
        """
        current_time = time.time()
        
        # Reset counter if a new minute has started
        if current_time - self.minute_start > 60:
            self.requests_this_minute = 0
            self.minute_start = current_time
        
        # Check if we've hit the rate limit
        if self.requests_this_minute >= API_LIMITS["requests_per_minute"]:
            sleep_time = 60 - (current_time - self.minute_start) + 1
            logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            self.requests_this_minute = 0
            self.minute_start = time.time()
        
        # Increment request counter
        self.requests_this_minute += 1
    
    def _get_cache_info(self, series_id: str) -> Dict[str, Optional[str]]:
        """
        Get cached data information including ETag.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Dictionary with cache file path and ETag
        """
        cache_file = RAW_DATA_DIR / f"{series_id}.csv"
        etag_file = RAW_DATA_DIR / f"{series_id}_etag.txt"
        
        cache_info = {
            "cache_file": cache_file,
            "etag_file": etag_file,
            "etag": None,
            "has_cache": cache_file.exists()
        }
        
        # Read stored ETag if it exists
        if etag_file.exists():
            try:
                cache_info["etag"] = etag_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Could not read ETag for {series_id}: {e}")
        
        return cache_info
    
    def _save_cache(self, series_id: str, data: pd.DataFrame, etag: str) -> None:
        """
        Save data and ETag to cache files.
        
        Args:
            series_id: FRED series identifier
            data: DataFrame to cache
            etag: ETag from FRED response
        """
        cache_info = self._get_cache_info(series_id)
        
        try:
            # Save data
            data.to_csv(cache_info["cache_file"], index=False)
            
            # Save ETag
            if etag:
                cache_info["etag_file"].write_text(etag)
            
            logger.info(f"Cached data for {series_id} ({len(data)} observations)")
            
        except Exception as e:
            logger.error(f"Failed to cache data for {series_id}: {e}")
    
    def _load_cache(self, series_id: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Cached DataFrame or None if not available
        """
        cache_info = self._get_cache_info(series_id)
        
        if not cache_info["has_cache"]:
            return None
        
        try:
            data = pd.read_csv(cache_info["cache_file"])
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {series_id}: {e}")
            return None
    
    def get_series(self, series_id: str, force: bool = False) -> pd.DataFrame:
        """
        Fetch a single economic series from FRED with intelligent caching.
        
        Args:
            series_id: FRED series identifier (e.g., "CPIAUCSL")
            force: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with date index and series values
            
        Raises:
            requests.RequestException: If API request fails
            ValueError: If series_id is invalid or data is empty
        """
        logger.info(f"Fetching series: {series_id}")
        
        # Get cache information
        cache_info = self._get_cache_info(series_id)
        
        # Return cached data if available and not forcing refresh
        if not force and cache_info["has_cache"]:
            cached_data = self._load_cache(series_id)
            if cached_data is not None:
                logger.info(f"Using cached data for {series_id}")
                
                # Still check if data needs updating using ETag
                if cache_info["etag"]:
                    if not self._needs_update(series_id, cache_info["etag"]):
                        return cached_data
        
        # Implement rate limiting
        self._rate_limit()
        
        # Prepare request parameters
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": DATA_SETTINGS["DEFAULT_START_DATE"]
        }
        
        # Add ETag header if we have one
        headers = {}
        if cache_info["etag"] and not force:
            headers["If-None-Match"] = cache_info["etag"]
        
        try:
            # Make API request
            url = f"{self.base_url}/series/observations"
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            
            # Handle 304 Not Modified - data hasn't changed
            if response.status_code == 304:
                logger.info(f"Data for {series_id} unchanged (304 Not Modified)")
                cached_data = self._load_cache(series_id)
                if cached_data is not None:
                    return cached_data
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            data_json = response.json()
            
            # Check for FRED API errors
            if "error_message" in data_json:
                raise ValueError(f"FRED API error: {data_json['error_message']}")
            
            # Extract observations
            observations = data_json.get("observations", [])
            if not observations:
                raise ValueError(f"No data returned for series {series_id}")
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            # Clean and process data
            df = self._process_series_data(df, series_id)
            
            # Cache the data with ETag
            etag = response.headers.get("ETag")
            self._save_cache(series_id, df, etag)
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            
            # Return cached data if available as fallback
            cached_data = self._load_cache(series_id)
            if cached_data is not None:
                logger.warning(f"Using stale cache for {series_id} due to API error")
                return cached_data
            
            raise
    
    def _needs_update(self, series_id: str, etag: str) -> bool:
        """
        Check if series needs updating by making a HEAD request with ETag.
        
        Args:
            series_id: FRED series identifier
            etag: Stored ETag
            
        Returns:
            True if data needs updating, False otherwise
        """
        try:
            self._rate_limit()
            
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
            
            headers = {"If-None-Match": etag}
            
            url = f"{self.base_url}/series/observations"
            response = self.session.head(url, params=params, headers=headers, timeout=10)
            
            # 304 means not modified
            return response.status_code != 304
            
        except Exception as e:
            logger.warning(f"Could not check update status for {series_id}: {e}")
            # If we can't check, assume we need to update
            return True
    
    def _process_series_data(self, df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """
        Clean and process raw FRED data.
        
        Args:
            df: Raw DataFrame from FRED API
            series_id: FRED series identifier
            
        Returns:
            Processed DataFrame with date index
        """
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values (FRED uses "." for missing)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna(subset=['value'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Set date as index
        df = df.set_index('date')
        
        # Rename value column to series_id
        df = df.rename(columns={'value': series_id})
        
        # Keep only the value column
        df = df[[series_id]]
        
        logger.info(f"Processed {len(df)} observations for {series_id}")
        
        return df

def get_series(series_id: str, force: bool = False) -> pd.DataFrame:
    """
    Convenience function to fetch a single series.
    
    Args:
        series_id: FRED series identifier
        force: If True, bypass cache and fetch fresh data
        
    Returns:
        DataFrame with economic data
    """
    fetcher = FREDDataFetcher()
    return fetcher.get_series(series_id, force=force)

def get_all(force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch all series defined in config.SERIES.
    
    Args:
        force: If True, bypass cache and fetch fresh data for all series
        
    Returns:
        Dictionary mapping series names to DataFrames
    """
    fetcher = FREDDataFetcher()
    results = {}
    
    logger.info(f"Fetching {len(SERIES)} economic series...")
    
    for name, series_id in SERIES.items():
        try:
            logger.info(f"Fetching {name} ({series_id})...")
            data = fetcher.get_series(series_id, force=force)
            results[name] = data
            
            # Brief pause between requests to be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to fetch {name} ({series_id}): {e}")
            # Continue with other series
            continue
    
    logger.info(f"Successfully fetched {len(results)} out of {len(SERIES)} series")
    return results

def main():
    """
    Command-line interface for data ingestion.
    """
    parser = argparse.ArgumentParser(description="Fetch FRED economic data")
    parser.add_argument(
        "--refresh", 
        action="store_true", 
        help="Force refresh all cached data"
    )
    parser.add_argument(
        "--series",
        type=str,
        help="Fetch specific series by ID (e.g., CPIAUCSL)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available series"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available economic series:")
        for name, series_id in SERIES.items():
            print(f"  {name}: {series_id}")
        return
    
    if args.series:
        # Fetch specific series
        try:
            data = get_series(args.series, force=args.refresh)
            print(f"✅ Successfully fetched {args.series}")
            print(f"   Data range: {data.index.min()} to {data.index.max()}")
            print(f"   Observations: {len(data)}")
        except Exception as e:
            print(f"❌ Failed to fetch {args.series}: {e}")
            sys.exit(1)
    else:
        # Fetch all series
        try:
            results = get_all(force=args.refresh)
            print(f"✅ Successfully fetched {len(results)} series")
            
            # Print summary
            for name, data in results.items():
                series_id = SERIES[name]
                print(f"   {name} ({series_id}): {len(data)} observations")
                
        except Exception as e:
            print(f"❌ Error during batch fetch: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()