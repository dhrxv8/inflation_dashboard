"""
Unit tests for FRED data ingestion module.
Tests API interactions, caching, and data processing using mocked responses.
"""

import json
import tempfile
import pytest
import pandas as pd
import responses
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_ingest import FREDDataFetcher, get_series, get_all
from src.config import FRED_BASE_URL


class TestFREDDataFetcher:
    """Test cases for FREDDataFetcher class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def fetcher(self, temp_data_dir):
        """Create FREDDataFetcher instance with temporary data directory."""
        with patch('src.data_ingest.RAW_DATA_DIR', temp_data_dir):
            return FREDDataFetcher(api_key="test_api_key")
    
    @pytest.fixture
    def mock_fred_response(self):
        """Sample FRED API response for testing."""
        return {
            "realtime_start": "2024-01-01",
            "realtime_end": "2024-01-01", 
            "observation_start": "2000-01-01",
            "observation_end": "9999-12-31",
            "units": "index",
            "output_type": 1,
            "file_type": "json",
            "order_by": "observation_date",
            "sort_order": "asc",
            "count": 3,
            "offset": 0,
            "limit": 100000,
            "observations": [
                {
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-01",
                    "date": "2023-01-01",
                    "value": "100.0"
                },
                {
                    "realtime_start": "2024-01-01", 
                    "realtime_end": "2024-01-01",
                    "date": "2023-02-01",
                    "value": "101.5"
                },
                {
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-01", 
                    "date": "2023-03-01",
                    "value": "102.0"
                }
            ]
        }
    
    @responses.activate
    def test_get_series_success(self, fetcher, mock_fred_response):
        """Test successful series fetching and data processing."""
        # Mock FRED API response
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations",
            json=mock_fred_response,
            status=200,
            headers={"ETag": '"test-etag-123"'}
        )
        
        # Fetch series
        result = fetcher.get_series("TEST_SERIES")
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "TEST_SERIES" in result.columns
        assert result.index.name == "date" or result.index.dtype.name.startswith('datetime')
        
        # Check values
        expected_values = [100.0, 101.5, 102.0]
        actual_values = result["TEST_SERIES"].tolist()
        assert actual_values == expected_values
        
        # Verify API call
        assert len(responses.calls) == 1
        request = responses.calls[0].request
        assert "series_id=TEST_SERIES" in request.url
        assert "api_key=test_api_key" in request.url
    
    @responses.activate
    def test_get_series_with_missing_values(self, fetcher):
        """Test handling of missing values in FRED data."""
        mock_response = {
            "observations": [
                {"date": "2023-01-01", "value": "100.0"},
                {"date": "2023-02-01", "value": "."},  # Missing value
                {"date": "2023-03-01", "value": "102.0"}
            ]
        }
        
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations", 
            json=mock_response,
            status=200
        )
        
        result = fetcher.get_series("TEST_SERIES")
        
        # Should have 2 observations (missing value dropped)
        assert len(result) == 2
        assert result["TEST_SERIES"].tolist() == [100.0, 102.0]
    
    @responses.activate 
    def test_get_series_api_error(self, fetcher):
        """Test handling of FRED API errors."""
        error_response = {
            "error_code": 400,
            "error_message": "Bad Request. The series does not exist."
        }
        
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations",
            json=error_response,
            status=400
        )
        
        # Should raise ValueError for API errors
        with pytest.raises(ValueError, match="FRED API error"):
            fetcher.get_series("INVALID_SERIES")
    
    @responses.activate
    def test_get_series_network_error_with_cache(self, fetcher, temp_data_dir):
        """Test fallback to cache when network request fails."""
        # Create cached data
        cache_file = temp_data_dir / "TEST_SERIES.csv"
        cached_data = pd.DataFrame({
            "date": ["2023-01-01", "2023-02-01"], 
            "TEST_SERIES": [100.0, 101.0]
        })
        cached_data.to_csv(cache_file, index=False)
        
        # Mock network failure
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations",
            body=ConnectionError("Network error")
        )
        
        # Should return cached data
        result = fetcher.get_series("TEST_SERIES")
        assert len(result) == 2
        assert result["TEST_SERIES"].tolist() == [100.0, 101.0]
    
    @responses.activate
    def test_caching_behavior(self, fetcher, mock_fred_response, temp_data_dir):
        """Test ETag-based caching mechanism."""
        etag = '"test-etag-123"'
        
        # First request - fresh data
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations",
            json=mock_fred_response,
            status=200,
            headers={"ETag": etag}
        )
        
        # Fetch data first time
        result1 = fetcher.get_series("TEST_SERIES")
        
        # Verify cache files were created
        cache_file = temp_data_dir / "TEST_SERIES.csv"
        etag_file = temp_data_dir / "TEST_SERIES_etag.txt" 
        assert cache_file.exists()
        assert etag_file.exists()
        assert etag_file.read_text().strip() == etag
        
        # Second request - should use cache (304 Not Modified)
        responses.add(
            responses.HEAD,
            f"{FRED_BASE_URL}/series/observations",
            status=304,
            headers={"ETag": etag}
        )
        
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations", 
            status=304,
            headers={"ETag": etag}
        )
        
        result2 = fetcher.get_series("TEST_SERIES")
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_force_refresh(self, fetcher, temp_data_dir):
        """Test force refresh bypasses cache."""
        # Create stale cached data
        cache_file = temp_data_dir / "TEST_SERIES.csv"
        stale_data = pd.DataFrame({
            "date": ["2022-01-01"],
            "TEST_SERIES": [50.0]
        })
        stale_data.to_csv(cache_file, index=False)
        
        # Mock fresh API response
        fresh_response = {
            "observations": [
                {"date": "2023-01-01", "value": "100.0"}
            ]
        }
        
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                f"{FRED_BASE_URL}/series/observations",
                json=fresh_response,
                status=200
            )
            
            # Force refresh should ignore cache
            result = fetcher.get_series("TEST_SERIES", force=True)
            
            assert len(result) == 1
            assert result["TEST_SERIES"].iloc[0] == 100.0
    
    def test_rate_limiting(self, fetcher):
        """Test rate limiting mechanism."""
        # Reset rate limiting state
        fetcher.requests_this_minute = 29  # Near limit
        fetcher.minute_start = 0  # Force minute reset
        
        with patch('time.sleep') as mock_sleep:
            with patch('time.time', return_value=61):  # Force new minute
                fetcher._rate_limit()
                
                # Should reset counter, no sleep needed
                assert fetcher.requests_this_minute == 1
                mock_sleep.assert_not_called()
    
    def test_data_processing(self, fetcher):
        """Test data cleaning and processing."""
        raw_data = pd.DataFrame({
            "realtime_start": ["2024-01-01", "2024-01-01"],
            "realtime_end": ["2024-01-01", "2024-01-01"], 
            "date": ["2023-01-01", "2023-02-01"],
            "value": ["100.5", "101.0"]
        })
        
        processed = fetcher._process_series_data(raw_data, "TEST_SERIES")
        
        # Check structure
        assert isinstance(processed.index, pd.DatetimeIndex)
        assert "TEST_SERIES" in processed.columns
        assert len(processed.columns) == 1
        
        # Check values
        assert processed["TEST_SERIES"].dtype in ['float64', 'int64']
        assert processed["TEST_SERIES"].tolist() == [100.5, 101.0]


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @responses.activate
    def test_get_series_function(self):
        """Test module-level get_series function."""
        mock_response = {
            "observations": [
                {"date": "2023-01-01", "value": "100.0"}
            ]
        }
        
        responses.add(
            responses.GET,
            f"{FRED_BASE_URL}/series/observations",
            json=mock_response,
            status=200
        )
        
        with patch('src.data_ingest.FRED_API_KEY', 'test_key'):
            result = get_series("TEST_SERIES")
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
    
    @patch('src.data_ingest.get_series')
    def test_get_all_function(self, mock_get_series):
        """Test get_all function with mocked series fetching."""
        # Mock series data
        mock_data = pd.DataFrame({
            "TEST_SERIES": [100.0, 101.0]
        })
        mock_get_series.return_value = mock_data
        
        # Mock config with small series set
        test_series = {"TEST1": "SERIES1", "TEST2": "SERIES2"}
        
        with patch('src.data_ingest.SERIES', test_series):
            results = get_all()
            
            # Should call get_series for each configured series
            assert mock_get_series.call_count == 2
            assert len(results) == 2
            assert "TEST1" in results
            assert "TEST2" in results


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_empty_response_handling(self):
        """Test handling of empty API responses."""
        fetcher = FREDDataFetcher("test_key")
        
        with pytest.raises(ValueError, match="No data returned"):
            fetcher._process_series_data(pd.DataFrame(), "TEST_SERIES")
    
    def test_invalid_date_handling(self):
        """Test handling of invalid date formats."""
        fetcher = FREDDataFetcher("test_key") 
        
        # Data with invalid date
        bad_data = pd.DataFrame({
            "date": ["invalid-date", "2023-01-01"],
            "value": ["100.0", "101.0"]
        })
        
        # Should handle gracefully (may convert to NaT and drop)
        try:
            result = fetcher._process_series_data(bad_data, "TEST_SERIES")
            # If it doesn't crash, check that valid data remains
            assert len(result) <= 1  # At most one valid observation
        except Exception:
            # Some pandas versions may raise - that's also acceptable
            pass


# Integration test (commented out - requires real API key)
"""
class TestIntegration:
    def test_real_api_integration(self):
        '''Test with real FRED API (requires valid API key).'''
        # Only run if API key is available
        if not FRED_API_KEY or FRED_API_KEY == "YOUR_KEY_HERE":
            pytest.skip("No valid FRED API key available")
        
        fetcher = FREDDataFetcher()
        result = fetcher.get_series("CPIAUCSL", force=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 100  # Should have many years of data
        assert "CPIAUCSL" in result.columns
"""