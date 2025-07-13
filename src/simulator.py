"""
Policy Shock Simulator for Inflation Dashboard
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional, List
import argparse
import logging

from src.config import PROCESSED_DATA_DIR, FORECASTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicySimulator:
    def __init__(self):
        self.beta_coefficients = None
        self.baseline_forecast = None
        self._load_beta_coefficients()
        logger.info("Policy simulator initialized")
    
    def _load_beta_coefficients(self):
        beta_file = PROCESSED_DATA_DIR / "beta_coefficients.json"
        if beta_file.exists():
            with open(beta_file, 'r') as f:
                self.beta_coefficients = json.load(f)
            logger.info("Loaded beta coefficients from calibration")
        else:
            logger.warning("Beta coefficients not found, using defaults")
            self.beta_coefficients = {
                'fed_funds_beta': {
                    'lag_3': -0.075,
                    'lag_6': -0.125, 
                    'lag_12': -0.175,
                    'total': -0.25
                },
                'money_supply_beta': {
                    'lag_3': 0.03,
                    'lag_6': 0.06,
                    'lag_12': 0.09, 
                    'total': 0.15
                },
                'fiscal_beta': {
                    'lag_3': 0.10,
                    'lag_6': 0.15,
                    'lag_12': 0.08,
                    'total': 0.15
                }
            }
    
    def shock_path(self, base_forecast: np.array, d_ffr: float, d_m2: float, 
                   fiscal: float, periods: int = 12) -> Dict[str, Union[np.array, Dict]]:
        """
        Apply policy shocks to baseline forecast and decompose contributions.
        
        Args:
            base_forecast: Baseline inflation forecast array
            d_ffr: Change in federal funds rate (percentage points)
            d_m2: Change in money supply growth rate (percentage points)
            fiscal: Fiscal policy shock (% of GDP)
            periods: Number of forecast periods
            
        Returns:
            Dictionary with 'new_forecast' array and 'contrib' decomposition
        """
        # Ensure we have the right number of periods
        if len(base_forecast) < periods:
            extended_forecast = np.zeros(periods)
            extended_forecast[:len(base_forecast)] = base_forecast
            extended_forecast[len(base_forecast):] = base_forecast[-1]
            base_forecast = extended_forecast
        else:
            base_forecast = base_forecast[:periods]
        
        # Initialize impact arrays
        fed_impact = np.zeros(periods)
        m2_impact = np.zeros(periods)
        fiscal_impact = np.zeros(periods)
        
        # Apply Fed funds rate shock with lag structure
        fed_beta = self.beta_coefficients['fed_funds_beta']
        for i in range(periods):
            if i >= 3:  # 3-month lag
                fed_impact[i] += d_ffr * fed_beta['lag_3']
            if i >= 6:  # 6-month lag
                fed_impact[i] += d_ffr * fed_beta['lag_6']
            if i >= 12:  # 12-month lag
                fed_impact[i] += d_ffr * fed_beta['lag_12']
        
        # Apply money supply shock with lag structure
        m2_beta = self.beta_coefficients['money_supply_beta']
        for i in range(periods):
            if i >= 3:  # 3-month lag
                m2_impact[i] += d_m2 * m2_beta['lag_3']
            if i >= 6:  # 6-month lag
                m2_impact[i] += d_m2 * m2_beta['lag_6']
            if i >= 12:  # 12-month lag
                m2_impact[i] += d_m2 * m2_beta['lag_12']
        
        # Apply fiscal policy shock with lag structure
        fiscal_beta = self.beta_coefficients['fiscal_beta']
        for i in range(periods):
            if i >= 3:  # 3-month lag
                fiscal_impact[i] += fiscal * fiscal_beta['lag_3']
            if i >= 6:  # 6-month lag
                fiscal_impact[i] += fiscal * fiscal_beta['lag_6']
            if i >= 12:  # 12-month lag
                fiscal_impact[i] += fiscal * fiscal_beta['lag_12']
        
        # Calculate total impact and new forecast
        total_impact = fed_impact + m2_impact + fiscal_impact
        new_forecast = base_forecast + total_impact
        
        # Contribution decomposition
        contrib = {
            'baseline': base_forecast.copy(),
            'fed_funds': fed_impact.copy(),
            'money_supply': m2_impact.copy(),
            'fiscal': fiscal_impact.copy(),
            'total_shock': total_impact.copy(),
            'shock_params': {
                'd_ffr': d_ffr,
                'd_m2': d_m2,
                'fiscal': fiscal
            }
        }
        
        return {
            'new_forecast': new_forecast,
            'contrib': contrib
        }
    
    def load_baseline_forecast(self, forecast_path: Optional[Path] = None) -> pd.DataFrame:
        """Load baseline inflation forecast."""
        if forecast_path is None:
            forecast_files = list(FORECASTS_DIR.glob("CPIAUCSL_yoy_forecasts.csv"))
            if not forecast_files:
                raise FileNotFoundError("No forecast files found. Run modeling first.")
            forecast_path = forecast_files[0]
        
        baseline = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        self.baseline_forecast = baseline
        logger.info(f"Loaded baseline forecast: {len(baseline)} periods")
        return baseline
    
    def run_scenario(self, scenario_name: str, d_ffr: float, d_m2: float, 
                    fiscal: float, model: str = 'Prophet') -> Dict:
        """Run a complete policy scenario simulation."""
        if self.baseline_forecast is None:
            self.load_baseline_forecast()
        
        # Get baseline forecast for specified model
        forecast_col = f"{model}_forecast"
        if forecast_col not in self.baseline_forecast.columns:
            available_models = [col.replace('_forecast', '') for col in self.baseline_forecast.columns if '_forecast' in col]
            raise ValueError(f"Model '{model}' not found. Available: {available_models}")
        
        base_forecast = self.baseline_forecast[forecast_col].values
        
        # Run shock simulation
        results = self.shock_path(base_forecast, d_ffr, d_m2, fiscal)
        
        # Package results
        scenario_results = {
            'scenario_name': scenario_name,
            'model_used': model,
            'dates': self.baseline_forecast.index,
            'baseline_forecast': results['contrib']['baseline'],
            'new_forecast': results['new_forecast'],
            'contributions': results['contrib'],
            'summary': {
                'baseline_avg': np.mean(results['contrib']['baseline']),
                'new_avg': np.mean(results['new_forecast']),
                'total_impact': np.mean(results['contrib']['total_shock']),
                'fed_impact': np.mean(results['contrib']['fed_funds']),
                'm2_impact': np.mean(results['contrib']['money_supply']),
                'fiscal_impact': np.mean(results['contrib']['fiscal'])
            }
        }
        
        logger.info(f"Scenario '{scenario_name}' completed")
        return scenario_results

def main():
    """Command-line interface for policy simulation."""
    parser = argparse.ArgumentParser(description="Run policy shock simulations")
    parser.add_argument("--d_ffr", type=float, default=1.0, help="Fed funds rate change (pp)")
    parser.add_argument("--d_m2", type=float, default=0.0, help="M2 growth change (pp)")
    parser.add_argument("--fiscal", type=float, default=0.0, help="Fiscal shock (% GDP)")
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = PolicySimulator()
    
    try:
        result = simulator.run_scenario(
            "Test Scenario",
            args.d_ffr,
            args.d_m2,
            args.fiscal
        )
        
        print(f"\n🎯 SCENARIO RESULTS: {result['scenario_name']}")
        print("=" * 50)
        print(f"Policy Changes:")
        print(f"  Fed Funds Rate: {args.d_ffr:+.1f} pp")
        print(f"  M2 Growth: {args.d_m2:+.1f} pp")
        print(f"  Fiscal Policy: {args.fiscal:+.1f} % GDP")
        print(f"\nImpact on Inflation:")
        print(f"  Baseline Average: {result['summary']['baseline_avg']:.2f}%")
        print(f"  New Average: {result['summary']['new_avg']:.2f}%")
        print(f"  Total Impact: {result['summary']['total_impact']:+.2f} pp")
        print(f"  Fed Impact: {result['summary']['fed_impact']:+.3f} pp")
        print(f"  M2 Impact: {result['summary']['m2_impact']:+.3f} pp")
        print(f"  Fiscal Impact: {result['summary']['fiscal_impact']:+.3f} pp")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")

if __name__ == "__main__":
    main()
