"""
Streamlit Inflation Dashboard with Live Policy Simulation
Interactive interface for inflation forecasting and policy scenario analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config import FORECASTS_DIR, PROCESSED_DATA_DIR
from src.simulator import PolicySimulator

# Page configuration
st.set_page_config(
    page_title="🎯 Inflation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .scenario-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .impact-positive {
        color: #d73027;
        font-weight: bold;
    }
    .impact-negative {
        color: #1a9641;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Economic term explanations
EXPLANATIONS = {
    "cpi": "**Consumer Price Index (CPI)**: Measures the average change in prices paid by consumers for goods and services. When CPI rises, it means inflation is occurring.",
    "core_cpi": "**Core CPI**: CPI excluding volatile food and energy prices. This gives a clearer picture of underlying inflation trends.",
    "fed_funds": "**Federal Funds Rate**: The interest rate at which banks lend to each other overnight. The Fed uses this as its main tool to control inflation.",
    "m2": "**M2 Money Supply**: Includes cash, checking deposits, savings deposits, and money market securities. More money in circulation can lead to higher inflation.",
    "unemployment": "**Unemployment Rate**: Percentage of people actively looking for work but can't find jobs. Lower unemployment often leads to higher wages and inflation.",
    "sarima": "**SARIMA Model**: A statistical model that captures trends and seasonal patterns in inflation data. Good for identifying cyclical inflation patterns.",
    "prophet": "**Prophet Model**: Facebook's forecasting model that handles complex seasonality and holidays. Excellent for capturing irregular patterns in economic data.",
    "ensemble": "**Ensemble Model**: Combines predictions from multiple models (SARIMA + Prophet) to create more robust forecasts by averaging their strengths.",
    "policy_shock": "**Policy Shock**: Sudden changes in monetary or fiscal policy that can affect inflation. Examples: Fed rate hikes, stimulus spending, quantitative easing.",
    "basis_points": "**Basis Points**: 1 basis point = 0.01%. So 100 basis points = 1%. Used to describe small changes in interest rates.",
    "inflation_target": "**Fed Inflation Target**: The Federal Reserve aims for 2% annual inflation, which they consider optimal for economic growth.",
    "lag_effect": "**Policy Lag**: Changes in Fed policy take time to affect the economy. Interest rate changes typically impact inflation 6-18 months later.",
    "tornado_plot": "**Tornado Plot**: Shows which policy changes have the biggest impact on inflation. Wider bars = bigger impact. Helps prioritize policy tools."
}

def show_explanation(key: str, inline: bool = False):
    """Display explanation for economic terms"""
    if key in EXPLANATIONS:
        if inline:
            st.info(EXPLANATIONS[key])
        else:
            with st.expander(f"💡 What is this?"):
                st.markdown(EXPLANATIONS[key])

@st.cache_data
def load_forecast_data():
    """Load forecast data with caching"""
    try:
        forecast_files = list(FORECASTS_DIR.glob("CPIAUCSL_yoy_forecasts.csv"))
        if not forecast_files:
            return None
        
        df = pd.read_csv(forecast_files[0], index_col=0, parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")
        return None

@st.cache_data
def load_historical_data():
    """Load historical inflation data"""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "feature_table.csv", index_col=0, parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

@st.cache_resource
def initialize_simulator():
    """Initialize policy simulator with caching"""
    try:
        simulator = PolicySimulator()
        simulator.load_baseline_forecast()
        return simulator
    except Exception as e:
        st.error(f"Error initializing simulator: {e}")
        return None

def create_kpi_metrics(forecast_df, historical_df, scenario_results=None):
    """Create KPI metric tiles with scenario comparison"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Current inflation (latest historical)
    if historical_df is not None and 'CPIAUCSL_yoy' in historical_df.columns:
        current_inflation = historical_df['CPIAUCSL_yoy'].dropna().iloc[-1]
        prev_inflation = historical_df['CPIAUCSL_yoy'].dropna().iloc[-2]
        inflation_delta = current_inflation - prev_inflation
    else:
        current_inflation = 3.2  # Fallback
        inflation_delta = 0.1
    
    with col1:
        st.metric(
            label="📈 Current Inflation",
            value=f"{current_inflation:.1f}%",
            delta=f"{inflation_delta:+.1f}pp",
            help="Latest year-over-year CPI inflation rate"
        )
        show_explanation("cpi")
    
    # 12-month forecast (baseline vs scenario)
    if forecast_df is not None:
        baseline_12m = forecast_df['Prophet_forecast'].iloc[-1]
        
        if scenario_results:
            scenario_12m = scenario_results['new_forecast'][-1]
            forecast_delta = scenario_12m - baseline_12m
            delta_text = f"{forecast_delta:+.1f}pp vs baseline"
            delta_color = "inverse" if forecast_delta > 0 else "normal"
        else:
            scenario_12m = baseline_12m
            forecast_delta = baseline_12m - current_inflation
            delta_text = f"{forecast_delta:+.1f}pp vs current"
            delta_color = "normal"
    else:
        scenario_12m = 4.3
        delta_text = "+1.1pp vs current"
        delta_color = "normal"
    
    with col2:
        st.metric(
            label="🔮 12-Month Forecast",
            value=f"{scenario_12m:.1f}%",
            delta=delta_text,
            delta_color=delta_color,
            help="Prophet model prediction for inflation in 12 months"
        )
        show_explanation("prophet")
    
    # Fed funds rate
    if historical_df is not None and 'FEDFUNDS' in historical_df.columns:
        fed_rate = historical_df['FEDFUNDS'].dropna().iloc[-1]
        prev_fed_rate = historical_df['FEDFUNDS'].dropna().iloc[-2]
        fed_delta = fed_rate - prev_fed_rate
    else:
        fed_rate = 5.25
        fed_delta = 0.0
    
    with col3:
        st.metric(
            label="🏦 Fed Funds Rate",
            value=f"{fed_rate:.2f}%",
            delta=f"{fed_delta:+.2f}pp",
            help="Current federal funds rate set by the Federal Reserve"
        )
        show_explanation("fed_funds")
    
    # Distance from target (baseline vs scenario)
    if scenario_results:
        target_distance = scenario_12m - 2.0
        delta_text = f"Scenario: {target_distance:+.1f}pp"
    else:
        target_distance = current_inflation - 2.0
        delta_text = "Fed Target: 2.0%"
    
    with col4:
        st.metric(
            label="🎯 Distance from Target",
            value=f"{target_distance:+.1f}pp",
            delta=delta_text,
            delta_color="inverse" if target_distance > 0 else "normal",
            help="How far forecast inflation is from the Fed's 2% target"
        )
        show_explanation("inflation_target")

def create_forecast_chart(forecast_df, historical_df, selected_models, forecast_horizon, scenario_results=None):
    """Create interactive forecast chart with scenario overlay"""
    st.subheader("📊 Inflation Forecast Visualization")
    
    fig = go.Figure()
    
    # Add historical data
    if historical_df is not None and 'CPIAUCSL_yoy' in historical_df.columns:
        historical_inflation = historical_df['CPIAUCSL_yoy'].dropna()
        # Show last 3 years of historical data
        recent_historical = historical_inflation.iloc[-36:]
        
        fig.add_trace(go.Scatter(
            x=recent_historical.index,
            y=recent_historical.values,
            mode='lines',
            name='Historical Inflation',
            line=dict(color='black', width=3),
            hovertemplate='%{x}<br>Inflation: %{y:.2f}%<extra></extra>'
        ))
    
    # Add baseline forecast models
    if forecast_df is not None:
        colors = {'Prophet': '#1f77b4', 'SARIMA': '#ff7f0e', 'Ensemble': '#2ca02c'}
        
        for model in selected_models:
            forecast_col = f"{model}_forecast"
            if forecast_col in forecast_df.columns:
                # Limit to selected horizon
                forecast_data = forecast_df[forecast_col].iloc[:forecast_horizon]
                
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    mode='lines',
                    name=f'{model} Baseline',
                    line=dict(color=colors.get(model, '#888888'), width=2, dash='dash'),
                    hovertemplate=f'{model} Baseline<br>%{{x}}<br>Forecast: %{{y:.2f}}%<extra></extra>'
                ))
    
    # Add scenario forecast if available
    if scenario_results is not None:
        scenario_dates = scenario_results['dates'][:len(scenario_results['new_forecast'])]
        
        fig.add_trace(go.Scatter(
            x=scenario_dates,
            y=scenario_results['new_forecast'],
            mode='lines',
            name='Scenario Forecast',
            line=dict(color='red', width=3),
            hovertemplate='Scenario<br>%{x}<br>Forecast: %{y:.2f}%<extra></extra>'
        ))
    
    # Add Fed target line
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="gray",
        annotation_text="Fed Target (2%)",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Inflation Forecast: Baseline vs Policy Scenario",
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_tornado_plot(scenario_results):
    """Create tornado plot showing policy impact decomposition"""
    if scenario_results is None:
        st.warning("No scenario results available for tornado plot")
        return
    
    contributions = scenario_results['contributions']
    
    # Calculate average impacts over forecast horizon
    fed_impact = np.mean(contributions['fed_funds'])
    m2_impact = np.mean(contributions['money_supply'])
    fiscal_impact = np.mean(contributions['fiscal'])
    
    # Prepare data for tornado plot
    policies = ['Fed Funds Rate', 'Money Supply (M2)', 'Fiscal Policy']
    impacts = [fed_impact, m2_impact, fiscal_impact]
    colors = ['#d73027' if x > 0 else '#1a9641' for x in impacts]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=policies,
        x=impacts,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{x:+.3f}pp" for x in impacts],
        textposition='auto',
        hovertemplate='%{y}<br>Impact: %{x:+.3f} pp<extra></extra>'
    ))
    
    fig.update_layout(
        title="🌪️ Policy Impact Tornado Plot",
        xaxis_title="Average Impact on Inflation (percentage points)",
        yaxis_title="Policy Tool",
        height=300,
        showlegend=False
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    st.markdown("""
    **📊 How to Read This Chart:**
    - 🟢 **Green bars**: Policies that reduce inflation (deflationary)
    - 🔴 **Red bars**: Policies that increase inflation (inflationary)  
    - **Longer bars**: Bigger impact on inflation
    - **Numbers**: Average impact over 12-month forecast period
    """)
    
    show_explanation("tornado_plot", inline=True)

def create_scenario_fan_chart(scenario_results, model_name="Prophet"):
    """Create fan chart for scenario with uncertainty bands"""
    if scenario_results is None:
        st.warning("No scenario results available for fan chart")
        return
    
    st.subheader(f"📈 Scenario Uncertainty Analysis")
    
    # Create matplotlib fan chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = scenario_results['dates'][:len(scenario_results['new_forecast'])]
    baseline = scenario_results['baseline_forecast']
    scenario = scenario_results['new_forecast']
    
    # Plot baseline
    ax.plot(dates, baseline, color='blue', linewidth=2, label='Baseline Forecast', alpha=0.7)
    
    # Plot scenario
    ax.plot(dates, scenario, color='red', linewidth=3, label='Scenario Forecast')
    
    # Create uncertainty bands (simplified - using ±0.5% around scenario)
    uncertainty = 0.5
    upper_band = scenario + uncertainty
    lower_band = scenario - uncertainty
    
    ax.fill_between(dates, lower_band, upper_band, alpha=0.3, color='red', label='Scenario 95% CI')
    
    # Add Fed target
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.7, label='Fed Target (2%)')
    
    ax.set_title('Policy Scenario: Inflation Forecast with Uncertainty', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Inflation Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Add impact summary
    total_impact = np.mean(scenario_results['contributions']['total_shock'])
    impact_direction = "increases" if total_impact > 0 else "decreases"
    impact_color = "🔴" if total_impact > 0 else "🟢"
    
    st.markdown(f"""
    **📊 Scenario Impact Summary:**
    
    {impact_color} **Policy scenario {impact_direction} average inflation by {abs(total_impact):.2f} percentage points**
    
    - **Baseline Average**: {np.mean(baseline):.2f}%
    - **Scenario Average**: {np.mean(scenario):.2f}%
    - **Net Change**: {total_impact:+.2f} percentage points
    """)

def create_policy_simulator():
    """Create interactive policy simulation interface"""
    st.subheader("🏛️ Policy Impact Simulator")
    
    st.markdown("""
    **Explore how different Fed and government policies affect inflation forecasts.**
    Adjust the sliders below and click "Run Shock" to see the impact on your inflation predictions.
    """)
    
    # Initialize simulator
    simulator = initialize_simulator()
    if simulator is None:
        st.error("❌ Could not initialize policy simulator. Please check data files.")
        return None, None
    
    # Policy controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏦 Monetary Policy**")
        fed_change = st.slider(
            "Fed Funds Rate Change",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            help="Change in federal funds rate (percentage points)"
        )
        st.caption("Current Fed Rate: ~5.25%")
        show_explanation("fed_funds")
        
        m2_change = st.slider(
            "Money Supply (M2) Growth Change",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            help="Change in M2 money supply growth rate"
        )
        st.caption("Recent M2 Growth: ~2-6% annually")
        show_explanation("m2")
    
    with col2:
        st.markdown("**🏛️ Fiscal Policy**")
        fiscal_change = st.slider(
            "Fiscal Stimulus (% of GDP)",
            min_value=-2.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            help="Government spending change as percentage of GDP"
        )
        st.caption("2020 CARES Act: ~10% of GDP")
        
        st.markdown("**Examples:**")
        st.markdown("- Infrastructure spending")
        st.markdown("- Tax cuts/increases") 
        st.markdown("- Unemployment benefits")
        st.markdown("- Economic stimulus payments")
    
    with col3:
        st.markdown("**📊 Current Policy Settings**")
        st.metric("Fed Rate Change", f"{fed_change:+.2f}pp")
        st.metric("M2 Growth Change", f"{m2_change:+.1f}pp") 
        st.metric("Fiscal Stimulus", f"{fiscal_change:+.1f}% GDP")
        
        # Scenario prediction (rough estimate)
        rough_impact = fed_change * -0.25 + m2_change * 0.15 + fiscal_change * 0.15
        st.metric("Estimated Impact", f"{rough_impact:+.2f}pp", help="Rough estimate before running full simulation")
        
        # Run simulation button
        run_simulation = st.button(
            "🚀 Run Policy Shock", 
            type="primary",
            help="Run full policy simulation with your settings"
        )
    
    # Run simulation if button clicked
    scenario_results = None
    if run_simulation:
        if abs(fed_change) < 0.01 and abs(m2_change) < 0.01 and abs(fiscal_change) < 0.01:
            st.warning("⚠️ All policy changes are zero. Try adjusting the sliders to see an impact!")
        else:
            with st.spinner("🔄 Running policy simulation..."):
                try:
                    # Get baseline forecast
                    baseline_forecast = simulator.baseline_forecast['Prophet_forecast'].values
                    
                    # Run shock simulation
                    shock_results = simulator.shock_path(
                        baseline_forecast, 
                        d_ffr=fed_change,
                        d_m2=m2_change, 
                        fiscal=fiscal_change
                    )
                    
                    # Package results
                    scenario_results = {
                        'baseline_forecast': shock_results['contrib']['baseline'],
                        'new_forecast': shock_results['new_forecast'],
                        'contributions': shock_results['contrib'],
                        'dates': simulator.baseline_forecast.index,
                        'policy_changes': {
                            'fed_funds': fed_change,
                            'm2_growth': m2_change,
                            'fiscal': fiscal_change
                        }
                    }
                    
                    st.success("✅ Policy simulation completed!")
                    
                except Exception as e:
                    st.error(f"❌ Simulation failed: {e}")
                    scenario_results = None
    
    return scenario_results, {
        'fed_change': fed_change,
        'm2_change': m2_change, 
        'fiscal_change': fiscal_change
    }

def display_scenario_results(scenario_results, policy_settings):
    """Display comprehensive scenario analysis results"""
    if scenario_results is None:
        return
    
    st.markdown("---")
    st.subheader("📊 Scenario Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Impact Overview", "🌪️ Policy Breakdown", "📊 Detailed Analysis"])
    
    with tab1:
        # Tornado plot
        create_tornado_plot(scenario_results)
        
        # Key statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Impact Summary")
            baseline_avg = np.mean(scenario_results['baseline_forecast'])
            scenario_avg = np.mean(scenario_results['new_forecast'])
            total_impact = np.mean(scenario_results['contributions']['total_shock'])
            
            st.markdown(f"""
            - **Baseline Average**: {baseline_avg:.2f}%
            - **Scenario Average**: {scenario_avg:.2f}%
            - **Total Impact**: {total_impact:+.2f} percentage points
            - **Peak Impact**: {np.max(np.abs(scenario_results['contributions']['total_shock'])):.2f}pp
            """)
        
        with col2:
            st.markdown("### 🎯 Policy Effectiveness")
            fed_impact = np.mean(scenario_results['contributions']['fed_funds'])
            m2_impact = np.mean(scenario_results['contributions']['money_supply'])
            fiscal_impact = np.mean(scenario_results['contributions']['fiscal'])
            
            impacts = [
                ("Fed Funds Rate", fed_impact, policy_settings['fed_change']),
                ("Money Supply", m2_impact, policy_settings['m2_change']),
                ("Fiscal Policy", fiscal_impact, policy_settings['fiscal_change'])
            ]
            
            for name, impact, change in impacts:
                if abs(change) > 0.01:
                    effectiveness = impact / change
                    st.markdown(f"- **{name}**: {effectiveness:.3f}pp per unit change")
                else:
                    st.markdown(f"- **{name}**: No change applied")
    
    with tab2:
        # Individual policy contributions over time
        st.markdown("### 📈 Policy Impact Over Time")
        
        fig = go.Figure()
        
        dates = scenario_results['dates'][:len(scenario_results['new_forecast'])]
        
        # Fed funds impact
        if np.any(scenario_results['contributions']['fed_funds'] != 0):
            fig.add_trace(go.Scatter(
                x=dates,
                y=scenario_results['contributions']['fed_funds'],
                mode='lines',
                name='Fed Funds Impact',
                line=dict(color='blue', width=2)
            ))
        
        # Money supply impact  
        if np.any(scenario_results['contributions']['money_supply'] != 0):
            fig.add_trace(go.Scatter(
                x=dates,
                y=scenario_results['contributions']['money_supply'],
                mode='lines',
                name='Money Supply Impact',
                line=dict(color='green', width=2)
            ))
        
        # Fiscal impact
        if np.any(scenario_results['contributions']['fiscal'] != 0):
            fig.add_trace(go.Scatter(
                x=dates,
                y=scenario_results['contributions']['fiscal'],
                mode='lines',
                name='Fiscal Impact',
                line=dict(color='orange', width=2)
            ))
        
        # Total impact
        fig.add_trace(go.Scatter(
            x=dates,
            y=scenario_results['contributions']['total_shock'],
            mode='lines',
            name='Total Impact',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        fig.update_layout(
            title="Policy Impact Decomposition Over Time",
            xaxis_title="Date",
            yaxis_title="Impact on Inflation (percentage points)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Lag effects explanation
        show_explanation("lag_effect", inline=True)
    
    with tab3:
        # Fan chart with scenario
        create_scenario_fan_chart(scenario_results)
        
        # Data table
        st.markdown("### 📊 Detailed Forecast Data")
        
        # Create summary table
        summary_data = {
            'Month': [f"Month {i+1}" for i in range(len(scenario_results['new_forecast']))],
            'Baseline (%)': [f"{x:.2f}" for x in scenario_results['baseline_forecast']],
            'Scenario (%)': [f"{x:.2f}" for x in scenario_results['new_forecast']],
            'Total Impact (pp)': [f"{x:+.3f}" for x in scenario_results['contributions']['total_shock']],
            'Fed Impact (pp)': [f"{x:+.3f}" for x in scenario_results['contributions']['fed_funds']],
            'M2 Impact (pp)': [f"{x:+.3f}" for x in scenario_results['contributions']['money_supply']],
            'Fiscal Impact (pp)': [f"{x:+.3f}" for x in scenario_results['contributions']['fiscal']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 Inflation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Professional inflation forecasting with live policy simulation**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("Configure your inflation analysis")
    
    # Model selection
    st.sidebar.subheader("🤖 Forecast Models")
    available_models = ['Prophet', 'SARIMA', 'Ensemble']
    selected_models = st.sidebar.multiselect(
        "Select models to display",
        available_models,
        default=['Prophet', 'Ensemble'],
        help="Choose which forecasting models to show in the charts"
    )
    
    # Forecast horizon
    st.sidebar.subheader("📅 Forecast Horizon")
    forecast_horizon = st.sidebar.selectbox(
        "Select forecast period",
        [6, 12, 18, 24],
        index=1,
        help="How many months ahead to forecast"
    )
    
    # Load data
    with st.spinner("Loading data..."):
        forecast_df = load_forecast_data()
        historical_df = load_historical_data()
    
    if forecast_df is None:
        st.error("⚠️ No forecast data found. Please run the modeling pipeline first.")
        st.code("python -m src.modeling --model all --target CPIAUCSL_yoy --save")
        return
    
    # Policy simulation section
    scenario_results, policy_settings = create_policy_simulator()
    
    st.markdown("---")
    
    # Main content with scenario-aware metrics
    st.subheader("📊 Key Economic Indicators")
    create_kpi_metrics(forecast_df, historical_df, scenario_results)
    
    st.markdown("---")
    
    # Main forecast chart with scenario overlay
    create_forecast_chart(forecast_df, historical_df, selected_models, forecast_horizon, scenario_results)
    
    # Display scenario results if available
    if scenario_results is not None:
        display_scenario_results(scenario_results, policy_settings)
    
    # Add sidebar information
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Status")
    
    if forecast_df is not None:
        st.sidebar.success(f"✅ Forecast data: {len(forecast_df)} periods")
    else:
        st.sidebar.warning("⚠️ No forecast data")
    
    if historical_df is not None:
        st.sidebar.success(f"✅ Historical data: {len(historical_df)} observations")
    else:
        st.sidebar.warning("⚠️ No historical data")
    
    # Add quick scenario presets
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Quick Scenarios")
    st.sidebar.markdown("*Try these common policy scenarios:*")
    
    if st.sidebar.button("🔴 Aggressive Tightening"):
        st.sidebar.markdown("Set: Fed +2%, M2 -1%, Fiscal -0.5%")
    
    if st.sidebar.button("🟢 Economic Stimulus"):
        st.sidebar.markdown("Set: Fed -0.5%, M2 +2%, Fiscal +1%")
    
    if st.sidebar.button("🟡 Balanced Approach"):
        st.sidebar.markdown("Set: Fed +0.5%, M2 0%, Fiscal -0.2%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🎯 <strong>Inflation Dashboard</strong> | Built with Streamlit, Prophet, SARIMA & Ensemble Models<br>
    📊 Data Source: Federal Reserve Economic Data (FRED) | 🔄 Last Updated: Real-time<br>
    🏛️ <strong>Policy Simulation Enabled</strong> | Explore Fed & Fiscal Policy Impacts
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()