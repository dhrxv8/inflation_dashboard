"""
Streamlit Inflation Dashboard with Live Policy Simulation
Interactive interface for inflation forecasting and policy scenario analysis
"""

import streamlit as st

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🎯 Inflation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else
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
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Try to import with error handling for deployment
try:
    from src.config import FORECASTS_DIR, PROCESSED_DATA_DIR
    from src.simulator import PolicySimulator
    IMPORTS_OK = True
except ImportError as e:
    st.warning(f"⚠️ Import notice: Running in demo mode - {e}")
    IMPORTS_OK = False
    # Create fallback directories
    FORECASTS_DIR = Path("data/forecasts")
    PROCESSED_DATA_DIR = Path("data/processed")

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
    """Load forecast data with cloud deployment fallback"""
    if not IMPORTS_OK:
        return load_sample_forecast_data()
    
    try:
        forecast_files = list(FORECASTS_DIR.glob("CPIAUCSL_yoy_forecasts.csv"))
        if not forecast_files:
            st.info("📊 Using sample forecast data for demonstration")
            return load_sample_forecast_data()
        
        df = pd.read_csv(forecast_files[0], index_col=0, parse_dates=True)
        return df
    except Exception as e:
        st.warning(f"Using sample data due to: {e}")
        return load_sample_forecast_data()

@st.cache_data
def load_historical_data():
    """Load historical inflation data with fallback"""
    if not IMPORTS_OK:
        return load_sample_historical_data()
    
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "feature_table.csv", index_col=0, parse_dates=True)
        return df
    except Exception as e:
        st.warning(f"Using sample historical data due to: {e}")
        return load_sample_historical_data()

@st.cache_data
def load_sample_forecast_data():
    """Create sample forecast data for demo"""
    dates = pd.date_range('2025-07-01', periods=12, freq='MS')
    sample_data = pd.DataFrame({
        'Prophet_forecast': [4.31, 4.25, 4.18, 4.12, 4.05, 3.98, 3.92, 3.85, 3.78, 3.72, 3.65, 3.58],
        'SARIMA_forecast': [2.31, 2.35, 2.42, 2.48, 2.55, 2.61, 2.68, 2.74, 2.81, 2.87, 2.94, 3.00],
        'Ensemble_forecast': [3.31, 3.30, 3.30, 3.30, 3.30, 3.29, 3.30, 3.29, 3.29, 3.29, 3.29, 3.29],
        'Prophet_lower': [3.81, 3.75, 3.68, 3.62, 3.55, 3.48, 3.42, 3.35, 3.28, 3.22, 3.15, 3.08],
        'Prophet_upper': [4.81, 4.75, 4.68, 4.62, 4.55, 4.48, 4.42, 4.35, 4.28, 4.22, 4.15, 4.08],
        'SARIMA_lower': [1.81, 1.85, 1.92, 1.98, 2.05, 2.11, 2.18, 2.24, 2.31, 2.37, 2.44, 2.50],
        'SARIMA_upper': [2.81, 2.85, 2.92, 2.98, 3.05, 3.11, 3.18, 3.24, 3.31, 3.37, 3.44, 3.50],
        'Ensemble_lower': [2.81, 2.80, 2.80, 2.80, 2.80, 2.79, 2.80, 2.79, 2.79, 2.79, 2.79, 2.79],
        'Ensemble_upper': [3.81, 3.80, 3.80, 3.80, 3.80, 3.79, 3.80, 3.79, 3.79, 3.79, 3.79, 3.79]
    }, index=dates)
    return sample_data

@st.cache_data  
def load_sample_historical_data():
    """Create sample historical data for demo"""
    dates = pd.date_range('2020-01-01', periods=60, freq='MS')
    # Create realistic inflation pattern
    base_inflation = 2.0
    covid_spike = np.zeros(60)
    covid_spike[20:30] = np.linspace(0, 7, 10)  # COVID spike
    covid_spike[30:45] = np.linspace(7, 1, 15)  # Gradual decline
    inflation_values = base_inflation + covid_spike + np.random.normal(0, 0.3, 60)
    
    # Create realistic fed funds pattern
    fed_rates = np.full(60, 5.25)
    fed_rates[20:35] = 0.25  # COVID emergency rates
    fed_rates[35:50] = np.linspace(0.25, 4.5, 15)  # Rate hiking cycle
    fed_rates += np.random.normal(0, 0.1, 60)
    
    historical_data = pd.DataFrame({
        'CPIAUCSL_yoy': inflation_values,
        'FEDFUNDS': fed_rates
    }, index=dates)
    return historical_data

@st.cache_resource
def initialize_simulator():
    """Initialize policy simulator with caching"""
    if not IMPORTS_OK:
        return None
    
    try:
        simulator = PolicySimulator()
        simulator.load_baseline_forecast()
        return simulator
    except Exception as e:
        st.warning(f"Policy simulator unavailable: {e}")
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
    if forecast_df is not None and 'Prophet_forecast' in forecast_df.columns:
        baseline_12m = forecast_df['Prophet_forecast'].iloc[-1]
        
        if scenario_results:
            scenario_12m = scenario_results.get('new_forecast_12m', baseline_12m)
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
    if scenario_results is not None and 'scenario_forecast' in scenario_results:
        scenario_dates = scenario_results['dates'][:len(scenario_results['scenario_forecast'])]
        
        fig.add_trace(go.Scatter(
            x=scenario_dates,
            y=scenario_results['scenario_forecast'],
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
    """Create enhanced tornado plot showing policy impact decomposition"""
    if scenario_results is None:
        st.info("🤷‍♂️ No policy changes to display. Adjust the sliders above and run the simulation!")
        return
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff7675 0%, #fd79a8 100%); padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;'>
        <h3 style='color: white; margin: 0; text-align: center;'>
            🌪️ Policy Impact Tornado Analysis
        </h3>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
            Which policies pack the biggest punch?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get impacts from scenario results
    fed_impact = scenario_results.get('fed_impact', 0)
    m2_impact = scenario_results.get('m2_impact', 0)
    fiscal_impact = scenario_results.get('fiscal_impact', 0)
    
    # Prepare data for tornado plot
    policies = []
    impacts = []
    colors = []
    
    if abs(fed_impact) > 0.001:
        policies.append('Fed Funds Rate')
        impacts.append(fed_impact)
        colors.append('#d73027' if fed_impact > 0 else '#1a9641')
    
    if abs(m2_impact) > 0.001:
        policies.append('Money Supply (M2)')
        impacts.append(m2_impact)
        colors.append('#d73027' if m2_impact > 0 else '#1a9641')
    
    if abs(fiscal_impact) > 0.001:
        policies.append('Fiscal Policy')
        impacts.append(fiscal_impact)
        colors.append('#d73027' if fiscal_impact > 0 else '#1a9641')
    
    if not policies:
        st.info("🤷‍♂️ No policy changes to display. Adjust the sliders above and run the simulation!")
        return
    
    # Create enhanced horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=policies,
        x=impacts,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f"{x:+.3f}pp" for x in impacts],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        name='Policy Impact'
    ))
    
    fig.update_layout(
        title="Policy Impact Tornado Plot",
        xaxis_title="Impact on Inflation (percentage points)",
        yaxis_title="",
        height=max(300, len(policies) * 80),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=150, r=50, t=100, b=80)
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced interpretation section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 How to Read This Chart")
        st.markdown("""
        - 🔴 **Red bars (right)**: Policies that **increase** inflation
        - 🟢 **Green bars (left)**: Policies that **reduce** inflation  
        - 📏 **Bar length**: Bigger impact = longer bar
        - 📈 **Numbers**: Average impact over forecast period
        """)
    
    with col2:
        st.markdown("### 🎯 Economic Insights")
        total_impact = fed_impact + m2_impact + fiscal_impact
        
        if abs(total_impact) > 0.2:
            impact_magnitude = "🔥 **Very Strong**"
        elif abs(total_impact) > 0.1:
            impact_magnitude = "⚡ **Moderate**"
        else:
            impact_magnitude = "💨 **Mild**"
        
        st.markdown(f"""
        **🎪 Total Scenario Impact:** {total_impact:+.3f}pp  
        **📊 Impact Magnitude:** {impact_magnitude}  
        **⏱️ Timeline:** 6-18 months for full effect
        """)

def create_policy_simulator():
    """Create enhanced interactive policy simulation interface"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 1rem; margin: 1rem 0;'>
        <h2 style='color: white; margin: 0; text-align: center;'>
            🏛️ Policy Impact Simulator
        </h2>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
            Explore how Fed and government policies shape inflation outcomes
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize simulator
    simulator = initialize_simulator()
    
    # Main explanation
    with st.expander("📚 How This Works", expanded=False):
        st.markdown("""
        This simulator uses **empirical relationships** from 40+ years of economic data to predict how policy changes affect inflation.
        
        **The Process:**
        1. **Adjust policy sliders** to test different scenarios
        2. **Click "Run Policy Shock"** to see the impact
        3. **View results** in charts and metrics above
        
        **Key Concepts:**
        - **Policy Lag**: Changes take 6-18 months to fully impact inflation
        - **Beta Coefficients**: Historical relationships between policies and inflation
        - **Ensemble Forecasting**: Combines multiple economic models for better predictions
        """)
    
    # Create policy sections
    st.markdown("### 🎛️ Policy Control Center")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏦 Monetary Policy**")
        fed_change = st.slider(
            "Fed Funds Rate Change",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            help="Change in federal funds rate (percentage points)",
            key="fed_rate_slider"
        )
        
        if fed_change > 0:
            st.markdown(f"📈 **Rate Hike (+{fed_change}%)** - Expected to reduce inflation")
        elif fed_change < 0:
            st.markdown(f"📉 **Rate Cut ({fed_change}%)** - Expected to increase inflation")
        else:
            st.markdown("➡️ **No Change** - Fed maintains current stance")
        
        st.caption("💡 Current Fed Rate: ~5.25%")
        show_explanation("fed_funds")
    
    with col2:
        st.markdown("**💰 Money Supply Policy**")
        m2_change = st.slider(
            "Money Supply (M2) Growth Change",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            help="Change in M2 money supply growth rate",
            key="m2_slider"
        )
        
        if m2_change > 0:
            st.markdown(f"💰 **Expansion (+{m2_change}%)** - More money in circulation")
        elif m2_change < 0:
            st.markdown(f"📉 **Contraction ({m2_change}%)** - Reduced money supply")
        else:
            st.markdown("➡️ **No Change** - Normal money growth")
        
        st.caption("💡 Recent M2 Growth: 2-6% annually")
        show_explanation("m2")
    
    with col3:
        st.markdown("**🏛️ Fiscal Policy**")
        fiscal_change = st.slider(
            "Fiscal Stimulus (% of GDP)",
            min_value=-2.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            help="Government spending change as percentage of GDP",
            key="fiscal_slider"
        )
        
        if fiscal_change > 0:
            st.markdown(f"💸 **Stimulus (+{fiscal_change}% GDP)** - Increased spending")
        elif fiscal_change < 0:
            st.markdown(f"✂️ **Restraint ({fiscal_change}% GDP)** - Reduced spending")
        else:
            st.markdown("➡️ **No Change** - Current fiscal stance")
        
        st.caption("💡 US GDP 2024: ~$27 trillion")
    
    # Policy examples
    with st.expander("🌍 Real Policy Examples", expanded=False):
        example_type = st.selectbox(
            "See Historical Examples:",
            ["COVID-19 Response (2020)", "2008 Financial Crisis", "1970s Inflation Fight", "Current Policy Debate"]
        )
        
        if example_type == "COVID-19 Response (2020)":
            st.markdown("""
            **🦠 COVID-19 Policy Response**
            - **Fed Rate**: Cut to 0% (emergency)
            - **M2 Growth**: +27% (massive QE)
            - **Fiscal**: +15% GDP (CARES Act)
            - **Result**: Inflation rose from 0.1% to 9.1%
            """)
        elif example_type == "2008 Financial Crisis":
            st.markdown("""
            **🏦 2008 Financial Crisis**
            - **Fed Rate**: Cut from 5.25% to 0%
            - **M2 Growth**: Moderate QE programs
            - **Fiscal**: +5% GDP stimulus
            - **Result**: Prevented deflation, slow recovery
            """)
        elif example_type == "1970s Inflation Fight":
            st.markdown("""
            **⚡ Volcker's Inflation Fight (1979-1982)**
            - **Fed Rate**: Raised to 20% (extreme)
            - **M2 Growth**: Severely restricted
            - **Fiscal**: Mixed policies
            - **Result**: Inflation fell from 14% to 3%
            """)
        elif example_type == "Current Policy Debate":
            st.markdown("""
            **🗳️ Current Policy Debates (2024-2025)**
            - **Fed Rate**: Whether to cut from 5.25%
            - **QE**: When to restart bond purchases
            - **Fiscal**: Infrastructure vs deficit reduction
            - **Challenge**: Balance growth vs inflation
            """)
    
    # Scenario Summary Panel
    st.markdown("---")
    st.markdown("### 📊 Scenario Summary & Controls")
    
    col4, col5, col6 = st.columns([1, 1, 1])
    
    with col4:
        st.markdown("**📈 Current Policy Settings**")
        st.metric("Fed Funds Rate Change", f"{fed_change:+.2f}pp")
        st.metric("M2 Growth Change", f"{m2_change:+.1f}pp")
        st.metric("Fiscal Policy Change", f"{fiscal_change:+.1f}% GDP")
    
    with col5:
        st.markdown("**🎯 Expected Direction**")
        
        # Calculate rough impact preview
        rough_fed_impact = fed_change * -0.25
        rough_m2_impact = m2_change * 0.15
        rough_fiscal_impact = fiscal_change * 0.15
        total_rough_impact = rough_fed_impact + rough_m2_impact + rough_fiscal_impact
        
        if total_rough_impact > 0.1:
            direction = "📈 Inflationary"
            color = "#e74c3c"
        elif total_rough_impact < -0.1:
            direction = "📉 Deflationary"
            color = "#27ae60"
        else:
            direction = "➡️ Neutral"
            color = "#95a5a6"
        
        st.markdown(f"""
        <div style='background-color: {color}20; padding: 1rem; border-radius: 0.5rem; text-align: center;'>
            <h4 style='color: {color}; margin: 0;'>{direction}</h4>
            <p style='margin: 0.5rem 0 0 0;'>Estimated: <strong>{total_rough_impact:+.2f}pp</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("⚠️ Rough estimate - run simulation for precise results")
    
    with col6:
        st.markdown("**🚀 Run Analysis**")
        
        # Enhanced run button
        run_simulation = st.button(
            "🔬 Run Policy Simulation", 
            type="primary",
            help="Generate detailed forecast using your policy settings",
            use_container_width=True
        )
        
        # Quick scenario buttons
        st.markdown("**⚡ Quick Scenarios:**")
        
        quick_col1, quick_col2 = st.columns(2)
        with quick_col1:
            if st.button("🔴 Hawkish Fed", help="Aggressive inflation fighting"):
                st.session_state.quick_scenario = "hawkish"
        with quick_col2:
            if st.button("🟢 Stimulus", help="Economic support package"):
                st.session_state.quick_scenario = "stimulus"
    
    # Handle quick scenarios
    if hasattr(st.session_state, 'quick_scenario'):
        if st.session_state.quick_scenario == "hawkish":
            st.info("💡 **Hawkish Fed Scenario Applied:** Set Fed Rate to +1.5%, M2 to -0.5%, Fiscal to -0.25%. Click 'Run Simulation' to see results!")
            fed_change, m2_change, fiscal_change = 1.5, -0.5, -0.25
        elif st.session_state.quick_scenario == "stimulus":
            st.info("💡 **Stimulus Scenario Applied:** Set Fed Rate to -0.5%, M2 to +2.0%, Fiscal to +1.0%. Click 'Run Simulation' to see results!")
            fed_change, m2_change, fiscal_change = -0.5, 2.0, 1.0
        
        # Clear the scenario after applying
        del st.session_state.quick_scenario
    
    # Run simulation logic
    scenario_results = None
    if run_simulation:
        if abs(fed_change) < 0.01 and abs(m2_change) < 0.01 and abs(fiscal_change) < 0.01:
            st.warning("⚠️ **No Policy Changes Detected**")
            st.info("💡 Try adjusting the sliders above to see policy impacts. Even small changes (0.25%) can have meaningful effects!")
        else:
            with st.spinner("🔄 Running advanced economic simulation..."):
                progress_bar = st.progress(0)
                
                try:
                    # Simulate processing steps
                    progress_bar.progress(25)
                    time.sleep(0.5)
                    
                    # Calculate impacts using empirical beta coefficients
                    progress_bar.progress(50)
                    
                    if simulator is not None:
                        # Use real simulator if available
                        baseline_forecast = simulator.baseline_forecast['Prophet_forecast'].values
                        shock_results = simulator.shock_path(
                            baseline_forecast, 
                            d_ffr=fed_change,
                            d_m2=m2_change, 
                            fiscal=fiscal_change
                        )
                        
                        scenario_results = {
                            'baseline_forecast': shock_results['contrib']['baseline'],
                            'scenario_forecast': shock_results['new_forecast'],
                            'fed_impact': np.mean(shock_results['contrib']['fed_funds']),
                            'm2_impact': np.mean(shock_results['contrib']['money_supply']),
                            'fiscal_impact': np.mean(shock_results['contrib']['fiscal']),
                            'total_impact': np.mean(shock_results['contrib']['total_shock']),
                            'new_forecast_12m': shock_results['new_forecast'][-1],
                            'dates': simulator.baseline_forecast.index,
                        }
                    else:
                        # Use simplified calculation for demo
                        fed_impact = fed_change * -0.25
                        m2_impact = m2_change * 0.15
                        fiscal_impact = fiscal_change * 0.15
                        total_impact = fed_impact + m2_impact + fiscal_impact
                        
                        baseline_12m = 4.3
                        new_forecast_12m = baseline_12m + total_impact
                        
                        scenario_results = {
                            'fed_impact': fed_impact,
                            'm2_impact': m2_impact,
                            'fiscal_impact': fiscal_impact,
                            'total_impact': total_impact,
                            'new_forecast_12m': new_forecast_12m,
                            'policy_changes': {
                                'fed_funds': fed_change,
                                'm2_growth': m2_change,
                                'fiscal': fiscal_change
                            }
                        }
                    
                    progress_bar.progress(75)
                    time.sleep(0.3)
                    progress_bar.progress(100)
                    
                    st.success("✅ **Policy simulation completed successfully!**")
                    st.balloons()  # Celebration effect
                    
                except Exception as e:
                    st.error(f"❌ **Simulation failed:** {e}")
                    st.info("💡 Using simplified calculation for demo")
                    scenario_results = None
                finally:
                    progress_bar.empty()
    
    return scenario_results, {
        'fed_change': fed_change,
        'm2_change': m2_change, 
        'fiscal_change': fiscal_change
    }

def display_scenario_results(scenario_results, policy_settings):
    """Display comprehensive scenario analysis results"""
    if scenario_results is None:
        return
    
    # Results header with summary
    st.markdown("""
    <div style='background: linear-gradient(135deg, #00cec9 0%, #55a3ff 100%); padding: 2rem; border-radius: 1rem; margin: 2rem 0 1rem 0;'>
        <h2 style='color: white; margin: 0; text-align: center;'>
            📊 Policy Scenario Analysis Results
        </h2>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;'>
            Comprehensive analysis of your policy scenario's economic impact
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick impact summary at the top
    total_impact = scenario_results.get('total_impact', 0)
    new_forecast = scenario_results.get('new_forecast_12m', 4.3)
    baseline_forecast = new_forecast - total_impact
    
    # Impact direction and magnitude
    if abs(total_impact) > 0.5:
        impact_level = "🔥 MAJOR"
        impact_color = "#e74c3c"
    elif abs(total_impact) > 0.2:
        impact_level = "⚡ SIGNIFICANT"
        impact_color = "#f39c12"
    elif abs(total_impact) > 0.05:
        impact_level = "💨 MODERATE"
        impact_color = "#3498db"
    else:
        impact_level = "🤏 MINIMAL"
        impact_color = "#95a5a6"
    
    direction = "INFLATIONARY" if total_impact > 0 else "DEFLATIONARY"
    direction_emoji = "📈" if total_impact > 0 else "📉"
    
    # Executive summary box
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {impact_color}20 0%, {impact_color}10 100%); 
                padding: 1.5rem; border-radius: 0.8rem; border-left: 5px solid {impact_color}; margin: 1rem 0;'>
        <h3 style='color: {impact_color}; margin: 0 0 1rem 0; text-align: center;'>
            {direction_emoji} {impact_level} {direction} IMPACT
        </h3>
        <div style='display: flex; justify-content: space-around; text-align: center;'>
            <div>
                <strong>Baseline Forecast</strong><br>
                <span style='font-size: 1.5em; color: #2c3e50;'>{baseline_forecast:.2f}%</span>
            </div>
            <div style='font-size: 2em; color: {impact_color}; align-self: center;'>→</div>
            <div>
                <strong>Scenario Forecast</strong><br>
                <span style='font-size: 1.5em; color: {impact_color};'>{new_forecast:.2f}%</span>
            </div>
            <div style='font-size: 2em; color: {impact_color}; align-self: center;'>=</div>
            <div>
                <strong>Net Impact</strong><br>
                <span style='font-size: 1.5em; color: {impact_color};'>{total_impact:+.2f}pp</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create organized tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "🌪️ Policy Breakdown", 
        "📊 Impact Analysis", 
        "🎯 Economic Context"
    ])
    
    with tab1:
        st.markdown("### 🌪️ Which Policies Drive the Impact?")
        
        # Enhanced tornado plot
        create_tornado_plot(scenario_results)
        
        # Policy effectiveness summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 Policy Changes Applied")
            
            fed_change = policy_settings['fed_change']
            m2_change = policy_settings['m2_change']
            fiscal_change = policy_settings['fiscal_change']
            
            if abs(fed_change) > 0.01:
                direction = "🔴 Hike" if fed_change > 0 else "🟢 Cut"
                st.markdown(f"**Fed Funds Rate:** {direction} of {abs(fed_change):.2f}pp")
            else:
                st.markdown("**Fed Funds Rate:** No change")
            
            if abs(m2_change) > 0.01:
                direction = "🔴 Expansion" if m2_change > 0 else "🟢 Contraction"
                st.markdown(f"**Money Supply:** {direction} of {abs(m2_change):.1f}pp")
            else:
                st.markdown("**Money Supply:** No change")
            
            if abs(fiscal_change) > 0.01:
                direction = "🔴 Stimulus" if fiscal_change > 0 else "🟢 Restraint"
                st.markdown(f"**Fiscal Policy:** {direction} of {abs(fiscal_change):.1f}% GDP")
            else:
                st.markdown("**Fiscal Policy:** No change")
        
        with col2:
            st.markdown("#### 🎯 Impact Breakdown")
            
            fed_impact = scenario_results.get('fed_impact', 0)
            m2_impact = scenario_results.get('m2_impact', 0)
            fiscal_impact = scenario_results.get('fiscal_impact', 0)
            
            impacts = [
                ("Fed Funds Rate", fed_impact, "🏦"),
                ("Money Supply", m2_impact, "💰"),
                ("Fiscal Policy", fiscal_impact, "🏛️")
            ]
            
            for name, impact, emoji in impacts:
                if abs(impact) > 0.001:
                    color = "#e74c3c" if impact > 0 else "#27ae60"
                    st.markdown(f"{emoji} **{name}**: <span style='color: {color};'>{impact:+.3f}pp</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"{emoji} **{name}**: No impact")
    
    with tab2:
        st.markdown("### 📊 Detailed Impact Analysis")
        
        # Policy effectiveness metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Fed Rate Effectiveness", 
                f"{scenario_results.get('fed_impact', 0):+.3f}pp",
                help="Impact per percentage point change in Fed rate"
            )
        
        with col2:
            st.metric(
                "Money Supply Effectiveness", 
                f"{scenario_results.get('m2_impact', 0):+.3f}pp",
                help="Impact per percentage point change in M2 growth"
            )
        
        with col3:
            st.metric(
                "Fiscal Effectiveness", 
                f"{scenario_results.get('fiscal_impact', 0):+.3f}pp",
                help="Impact per percentage point of GDP"
            )
        
        # Timeline explanation
        st.markdown("### ⏰ Policy Timeline Effects")
        st.markdown("""
        **Economic policy transmission works through multiple channels:**
        
        **🏦 Monetary Policy Lags:**
        - **0-3 months**: Financial markets react immediately
        - **3-6 months**: Business investment decisions change
        - **6-18 months**: Consumer spending and employment adjust
        - **12-24 months**: Full price level adjustment
        
        **🏛️ Fiscal Policy Lags:**
        - **0-6 months**: Direct government spending hits economy
        - **6-12 months**: Multiplier effects spread through economy  
        - **12-18 months**: Full employment and wage effects
        """)
    
    with tab3:
        st.markdown("### 🎯 Economic Context & Real-World Implications")
        
        # Real-world comparison
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🌍 Historical Context")
            
            if abs(total_impact) > 1.0:
                st.markdown("""
                **🔥 This is a MAJOR policy scenario comparable to:**
                - 1979-1981 Volcker inflation fight (-6pp impact)
                - 2008 Financial crisis response (+2pp impact)  
                - 2020 COVID-19 policy response (+8pp impact)
                """)
            elif abs(total_impact) > 0.5:
                st.markdown("""
                **⚡ This is a SIGNIFICANT policy shift similar to:**
                - Fed rate cycles during normal recessions
                - Major fiscal stimulus packages
                - Quantitative easing programs
                """)
            else:
                st.markdown("""
                **💨 This represents a MODERATE adjustment like:**
                - Typical FOMC meeting rate changes
                - Normal fiscal policy adjustments
                - Fine-tuning during economic cycles
                """)
        
        with col2:
            st.markdown("#### 📰 Market & Economic Implications")
            
            implications = []
            
            if abs(total_impact) > 0.3:
                implications.extend([
                    "📈 **Stock markets** would likely react strongly",
                    "💱 **Bond yields** would adjust significantly", 
                    "🏠 **Mortgage rates** would change noticeably"
                ])
            
            if total_impact > 0.2:
                implications.extend([
                    "🛒 **Consumer prices** would rise faster",
                    "💰 **Wage negotiations** would intensify",
                    "🏦 **Fed** might need to respond with counter-measures"
                ])
            elif total_impact < -0.2:
                implications.extend([
                    "🛒 **Consumer prices** would moderate",
                    "💼 **Employment** might be affected",
                    "🏦 **Fed** might ease policy to support growth"
                ])
            
            for implication in implications:
                st.markdown(implication)
            
            if not implications:
                st.markdown("💤 **Minimal market reaction** expected from this scenario")
        
        # Risk factors
        st.markdown("#### ⚠️ Key Risks & Assumptions")
        
        st.markdown("""
        **This analysis assumes:**
        - ✅ Historical relationships remain stable
        - ✅ No major supply shocks (oil, food, etc.)
        - ✅ Policy credibility is maintained
        - ✅ Global economic conditions remain stable
        
        **Potential risks:**
        - 🌪️ **Supply chain disruptions** could amplify inflation
        - 🌍 **Global conflicts** could affect energy/food prices
        - 🗳️ **Political changes** could alter policy credibility
        - 📊 **Structural economic changes** could break historical patterns
        """)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 Inflation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Professional inflation forecasting with live policy simulation**")
    
    if not IMPORTS_OK:
        st.info("📊 Running in demo mode with sample data - full functionality available with complete setup")
    
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
        st.error("⚠️ No forecast data found. Using sample data for demonstration.")
        forecast_df = load_sample_forecast_data()
    
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
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This Dashboard")
    
    st.markdown("""
    This professional inflation dashboard combines:
    
    **🔬 Advanced Analytics:**
    - Real-time economic data from Federal Reserve (FRED API)
    - Machine learning models: SARIMA, Prophet, and Ensemble forecasting
    - Empirically-calibrated policy impact coefficients from 40+ years of data
    
    **🎯 Interactive Features:**
    - Live policy simulation with tornado plots
    - Educational explanations for all economic terms
    - Professional visualizations with uncertainty quantification
    
    **📊 Use Cases:**
    - Central bank policy analysis
    - Investment research and portfolio allocation
    - Academic research and education
    - Economic scenario planning
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🎯 <strong>Inflation Dashboard</strong> | Built with Streamlit, Advanced ML Models & Economic Theory<br>
    📊 Data: Federal Reserve Economic Data (FRED) | 🔬 Graduate-Level Econometric Analysis<br>
    🏛️ <strong>Policy Simulation Engine</strong> | Explore Fed & Fiscal Policy Impacts
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()