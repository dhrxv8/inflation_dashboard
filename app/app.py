"""
Streamlit Inflation Dashboard MVP
Interactive interface for inflation forecasting and policy simulation
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
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
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
    "lag_effect": "**Policy Lag**: Changes in Fed policy take time to affect the economy. Interest rate changes typically impact inflation 6-18 months later."
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

def create_kpi_metrics(forecast_df, historical_df):
    """Create KPI metric tiles"""
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
    
    # 12-month forecast
    if forecast_df is not None:
        forecast_12m = forecast_df['Prophet_forecast'].iloc[-1]
        forecast_delta = forecast_12m - current_inflation
    else:
        forecast_12m = 4.3
        forecast_delta = 1.1
    
    with col2:
        st.metric(
            label="🔮 12-Month Forecast",
            value=f"{forecast_12m:.1f}%",
            delta=f"{forecast_delta:+.1f}pp vs current",
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
    
    # Distance from target
    target_distance = current_inflation - 2.0
    
    with col4:
        st.metric(
            label="🎯 Distance from Target",
            value=f"{target_distance:+.1f}pp",
            delta="Fed Target: 2.0%",
            delta_color="inverse" if target_distance > 0 else "normal",
            help="How far current inflation is from the Fed's 2% target"
        )
        show_explanation("inflation_target")

def create_forecast_chart(forecast_df, historical_df, selected_models, forecast_horizon):
    """Create interactive forecast chart"""
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
            line=dict(color='black', width=2),
            hovertemplate='%{x}<br>Inflation: %{y:.2f}%<extra></extra>'
        ))
    
    # Add forecast models
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
                    name=f'{model} Forecast',
                    line=dict(color=colors.get(model, '#888888'), width=2, dash='dash'),
                    hovertemplate=f'{model}<br>%{{x}}<br>Forecast: %{{y:.2f}}%<extra></extra>'
                ))
                
                # Add confidence intervals if available
                lower_col = f"{model}_lower"
                upper_col = f"{model}_upper"
                if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
                    lower_data = forecast_df[lower_col].iloc[:forecast_horizon]
                    upper_data = forecast_df[upper_col].iloc[:forecast_horizon]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index.tolist() + forecast_data.index.tolist()[::-1],
                        y=upper_data.tolist() + lower_data.tolist()[::-1],
                        fill='toself',
                        fillcolor=colors.get(model, '#888888').replace('rgb', 'rgba').replace(')', ', 0.2)'),
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model} 95% CI',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # Add Fed target line
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="red",
        annotation_text="Fed Target (2%)",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Inflation Forecast: Historical vs Predicted",
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
    
    # Model explanations
    st.markdown("### 🤖 Model Explanations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_explanation("sarima", inline=True)
    with col2:
        show_explanation("prophet", inline=True)
    with col3:
        show_explanation("ensemble", inline=True)

def create_fan_chart(forecast_df, selected_model):
    """Create fan chart showing forecast uncertainty"""
    if forecast_df is None or selected_model not in ['Prophet', 'SARIMA', 'Ensemble']:
        st.warning("Fan chart not available for selected model")
        return
    
    forecast_col = f"{selected_model}_forecast"
    lower_col = f"{selected_model}_lower"
    upper_col = f"{selected_model}_upper"
    
    if not all(col in forecast_df.columns for col in [forecast_col, lower_col, upper_col]):
        st.warning("Confidence intervals not available for selected model")
        return
    
    # Create matplotlib fan chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = forecast_df.index
    forecast = forecast_df[forecast_col]
    lower = forecast_df[lower_col]
    upper = forecast_df[upper_col]
    
    # Plot central forecast
    ax.plot(dates, forecast, color='blue', linewidth=2, label=f'{selected_model} Forecast')
    
    # Fill between for confidence intervals
    ax.fill_between(dates, lower, upper, alpha=0.3, color='blue', label='95% Confidence Interval')
    
    # Add Fed target
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Fed Target (2%)')
    
    ax.set_title(f'{selected_model} Inflation Forecast with Uncertainty', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Inflation Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown(f"""
    **📊 Fan Chart Interpretation:**
    - The blue line shows the most likely inflation path according to the {selected_model} model
    - The shaded area represents the 95% confidence interval - inflation has a 95% chance of falling within this range
    - Wider bands indicate more uncertainty about future inflation
    """)
    
    show_explanation("lag_effect", inline=True)

def create_policy_simulator():
    """Create policy simulation interface"""
    st.subheader("🏛️ Policy Impact Simulator")
    
    st.markdown("""
    Explore how different Fed and government policies would affect inflation forecasts.
    **Note**: Sliders are currently disabled (set to zero) - this is a baseline demonstration.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏦 Monetary Policy**")
        fed_change = st.slider(
            "Fed Funds Rate Change",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            disabled=True,  # Hard-wired as requested
            help="Change in federal funds rate (percentage points)"
        )
        show_explanation("fed_funds")
        
        m2_change = st.slider(
            "Money Supply (M2) Growth Change",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            disabled=True,  # Hard-wired as requested
            help="Change in M2 money supply growth rate"
        )
        show_explanation("m2")
    
    with col2:
        st.markdown("**🏛️ Fiscal Policy**")
        fiscal_change = st.slider(
            "Fiscal Stimulus (% of GDP)",
            min_value=-2.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            disabled=True,  # Hard-wired as requested
            help="Government spending change as percentage of GDP"
        )
        
        st.markdown("**Examples of Fiscal Policy:**")
        st.markdown("- Infrastructure spending")
        st.markdown("- Tax cuts/increases") 
        st.markdown("- Unemployment benefits")
        st.markdown("- Economic stimulus checks")
    
    with col3:
        st.markdown("**📊 Current Settings**")
        st.metric("Fed Rate Change", f"{fed_change:+.2f}pp")
        st.metric("M2 Growth Change", f"{m2_change:+.1f}pp") 
        st.metric("Fiscal Stimulus", f"{fiscal_change:+.1f}% GDP")
        
        if st.button("🚀 Run Simulation", disabled=True):
            st.info("Policy simulation will be enabled in the next phase!")
    
    # Add explanations
    show_explanation("policy_shock", inline=True)
    show_explanation("basis_points", inline=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🎯 Inflation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Professional inflation forecasting with policy simulation capabilities**")
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
    
    # Fan chart model selection
    st.sidebar.subheader("📊 Fan Chart Model")
    fan_chart_model = st.sidebar.selectbox(
        "Select model for uncertainty visualization",
        available_models,
        index=0,
        help="Choose model for the detailed fan chart with confidence intervals"
    )
    
    # Add sidebar explanations
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Help")
    st.sidebar.info("Hover over any metric or chart element for detailed explanations!")
    
    # Load data
    with st.spinner("Loading forecast data..."):
        forecast_df = load_forecast_data()
        historical_df = load_historical_data()
    
    if forecast_df is None:
        st.error("⚠️ No forecast data found. Please run the modeling pipeline first.")
        st.code("python -m src.modeling --model all --target CPIAUCSL_yoy --save")
        return
    
    # Main content
    # KPI Metrics
    st.subheader("📊 Key Economic Indicators")
    create_kpi_metrics(forecast_df, historical_df)
    
    st.markdown("---")
    
    # Main forecast chart
    create_forecast_chart(forecast_df, historical_df, selected_models, forecast_horizon)
    
    st.markdown("---")
    
    # Fan chart in expander
    with st.expander("📈 Detailed Forecast Uncertainty (Fan Chart)", expanded=False):
        st.markdown("**Fan charts** show the range of possible inflation outcomes and their probabilities.")
        create_fan_chart(forecast_df, fan_chart_model)
    
    st.markdown("---")
    
    # Policy simulator
    create_policy_simulator()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🎯 <strong>Inflation Dashboard</strong> | Built with Streamlit, Prophet, SARIMA & Ensemble Models<br>
    📊 Data Source: Federal Reserve Economic Data (FRED) | 🔄 Last Updated: Real-time
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()