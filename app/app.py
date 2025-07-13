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
import time

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
# Add these missing functions before the main() function:

def create_policy_timeline_chart(scenario_results):
    """Create timeline showing how policy effects build over time"""
    if scenario_results is None:
        return
    
    st.markdown("### ⏰ Policy Impact Timeline")
    st.markdown("See how different policies affect inflation over time with realistic lag effects.")
    
    contributions = scenario_results['contributions']
    dates = scenario_results['dates'][:len(scenario_results['new_forecast'])]
    
    # Create stacked area chart showing cumulative effects
    fig = go.Figure()
    
    # Fed funds impact
    if np.any(contributions['fed_funds'] != 0):
        fig.add_trace(go.Scatter(
            x=dates,
            y=contributions['fed_funds'],
            fill='tonexty',
            mode='lines',
            name='Fed Funds Rate',
            line=dict(color='#1f77b4', width=0),
            fillcolor='rgba(31, 119, 180, 0.3)',
            hovertemplate='Fed Impact: %{y:+.3f}pp<br>%{x}<extra></extra>'
        ))
    
    # Money supply impact
    if np.any(contributions['money_supply'] != 0):
        fig.add_trace(go.Scatter(
            x=dates,
            y=contributions['money_supply'],
            fill='tonexty',
            mode='lines',
            name='Money Supply (M2)',
            line=dict(color='#ff7f0e', width=0),
            fillcolor='rgba(255, 127, 14, 0.3)',
            hovertemplate='M2 Impact: %{y:+.3f}pp<br>%{x}<extra></extra>'
        ))
    
    # Fiscal impact
    if np.any(contributions['fiscal'] != 0):
        fig.add_trace(go.Scatter(
            x=dates,
            y=contributions['fiscal'],
            fill='tonexty',
            mode='lines',
            name='Fiscal Policy',
            line=dict(color='#2ca02c', width=0),
            fillcolor='rgba(44, 160, 44, 0.3)',
            hovertemplate='Fiscal Impact: %{y:+.3f}pp<br>%{x}<extra></extra>'
        ))
    
    # Total impact line
    fig.add_trace(go.Scatter(
        x=dates,
        y=contributions['total_shock'],
        mode='lines',
        name='Total Impact',
        line=dict(color='red', width=3),
        hovertemplate='Total Impact: %{y:+.3f}pp<br>%{x}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
    
    fig.update_layout(
        title="Policy Impact Timeline: How Effects Build Over Time",
        xaxis_title="Date",
        yaxis_title="Impact on Inflation (percentage points)",
        height=400,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights about timing
    peak_month = np.argmax(np.abs(contributions['total_shock'])) + 1
    peak_impact = contributions['total_shock'][peak_month - 1]
    
    st.markdown(f"""
    **⏰ Timeline Insights:**
    - **Peak Impact**: Month {peak_month} ({peak_impact:+.3f}pp)
    - **Lag Pattern**: Effects build gradually due to economic adjustment time
    - **Policy Persistence**: Sustained policy changes have cumulative effects
    """)

def create_scenario_comparison_table(scenario_results):
    """Create detailed comparison table of baseline vs scenario"""
    if scenario_results is None:
        return
    
    st.markdown("### 📊 Detailed Forecast Comparison")
    
    # Prepare data for table
    baseline = scenario_results['baseline_forecast']
    scenario = scenario_results['new_forecast']
    total_impact = scenario_results['contributions']['total_shock']
    
    comparison_data = {
        'Month': [f"Month {i+1}" for i in range(len(scenario))],
        'Date': [date.strftime('%Y-%m') for date in scenario_results['dates'][:len(scenario)]],
        'Baseline Forecast (%)': [f"{x:.2f}" for x in baseline],
        'Scenario Forecast (%)': [f"{x:.2f}" for x in scenario],
        'Impact (pp)': [f"{x:+.3f}" for x in total_impact],
        'Change (%)': [f"{((scenario[i]/baseline[i] - 1) * 100):+.1f}" for i in range(len(scenario))]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(comparison_df, use_container_width=True, height=400)
    
    # Add summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_baseline = np.mean(baseline)
        st.metric("Average Baseline", f"{avg_baseline:.2f}%")
    
    with col2:
        avg_scenario = np.mean(scenario)
        change = avg_scenario - avg_baseline
        st.metric("Average Scenario", f"{avg_scenario:.2f}%", f"{change:+.2f}pp")
    
    with col3:
        max_impact = np.max(np.abs(total_impact))
        st.metric("Peak Impact", f"{max_impact:.3f}pp")

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

# Replace the incomplete create_tornado_plot function with this:

def create_tornado_plot(scenario_results):
    """Create enhanced tornado plot showing policy impact decomposition"""
    if scenario_results is None:
        st.warning("No scenario results available for tornado plot")
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
    
    contributions = scenario_results['contributions']
    policy_changes = scenario_results['policy_changes']
    
    # Calculate average impacts over forecast horizon
    fed_impact = np.mean(contributions['fed_funds'])
    m2_impact = np.mean(contributions['money_supply'])
    fiscal_impact = np.mean(contributions['fiscal'])
    total_impact = np.mean(contributions['total_shock'])
    
    # Prepare data for tornado plot with more detailed labels
    policies = []
    impacts = []
    colors = []
    hover_texts = []
    
    # Fed Funds Rate
    if abs(policy_changes['fed_funds']) > 0.01:
        direction = "hike" if policy_changes['fed_funds'] > 0 else "cut"
        policies.append(f"Fed Rate {direction.title()}<br>({policy_changes['fed_funds']:+.2f}pp)")
        impacts.append(fed_impact)
        colors.append('#d73027' if fed_impact > 0 else '#1a9641')
        hover_texts.append(f"Fed Rate Change: {policy_changes['fed_funds']:+.2f}pp<br>Average Impact: {fed_impact:+.3f}pp<br>Peak Impact: {np.max(np.abs(contributions['fed_funds'])):+.3f}pp")
    
    # Money Supply
    if abs(policy_changes['m2_growth']) > 0.01:
        direction = "expansion" if policy_changes['m2_growth'] > 0 else "contraction"
        policies.append(f"M2 {direction.title()}<br>({policy_changes['m2_growth']:+.1f}pp)")
        impacts.append(m2_impact)
        colors.append('#d73027' if m2_impact > 0 else '#1a9641')
        hover_texts.append(f"M2 Growth Change: {policy_changes['m2_growth']:+.1f}pp<br>Average Impact: {m2_impact:+.3f}pp<br>Peak Impact: {np.max(np.abs(contributions['money_supply'])):+.3f}pp")
    
    # Fiscal Policy
    if abs(policy_changes['fiscal']) > 0.01:
        direction = "stimulus" if policy_changes['fiscal'] > 0 else "restraint"
        policies.append(f"Fiscal {direction.title()}<br>({policy_changes['fiscal']:+.1f}% GDP)")
        impacts.append(fiscal_impact)
        colors.append('#d73027' if fiscal_impact > 0 else '#1a9641')
        hover_texts.append(f"Fiscal Change: {policy_changes['fiscal']:+.1f}% GDP<br>Average Impact: {fiscal_impact:+.3f}pp<br>Peak Impact: {np.max(np.abs(contributions['fiscal'])):+.3f}pp")
    
    if not policies:
        st.info("🤷‍♂️ No policy changes to display. Adjust the sliders above and run the simulation!")
        return
    
    # Create enhanced horizontal bar chart
    fig = go.Figure()
    
    # Main tornado bars
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
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        name='Policy Impact'
    ))
    
    # Add impact magnitude indicators
    max_impact = max(abs(x) for x in impacts) if impacts else 0
    
    fig.update_layout(
        title={
            'text': "Policy Impact Tornado Plot<br><sub>Average impact over 12-month forecast period</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Impact on Inflation (percentage points)",
        yaxis_title="",
        height=max(300, len(policies) * 80),  # Dynamic height based on number of policies
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=150, r=50, t=100, b=80)
    )
    
    # Add vertical line at zero with style
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    # Add annotations for interpretation
    if max_impact > 0:
        # Add range indicators
        fig.add_annotation(
            x=max_impact * 0.7,
            y=len(policies),
            text="<b>Inflationary →</b>",
            showarrow=False,
            font=dict(color='#d73027', size=12),
            xanchor='center'
        )
        
        fig.add_annotation(
            x=-max_impact * 0.7,
            y=len(policies),
            text="<b>← Deflationary</b>",
            showarrow=False,
            font=dict(color='#1a9641', size=12),
            xanchor='center'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced interpretation section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 How to Read This Chart")
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
            <strong>🔴 Red bars (right):</strong> Policies that <strong>increase</strong> inflation<br>
            <strong>🟢 Green bars (left):</strong> Policies that <strong>reduce</strong> inflation<br>
            <strong>📏 Bar length:</strong> Bigger impact = longer bar<br>
            <strong>📈 Numbers:</strong> Average impact over 12 months
        </div>
        """, unsafe_allow_html=True)
        
        # Policy effectiveness ranking
        if len(impacts) > 1:
            policy_effectiveness = list(zip(policies, [abs(x) for x in impacts]))
            policy_effectiveness.sort(key=lambda x: x[1], reverse=True)
            
            st.markdown("**🏆 Policy Effectiveness Ranking:**")
            for i, (policy, magnitude) in enumerate(policy_effectiveness):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                policy_clean = policy.split('<br>')[0]  # Remove the change amount for cleaner display
                st.markdown(f"{rank_emoji} **{policy_clean}**: {magnitude:.3f}pp impact")
    
    with col2:
        st.markdown("### 🎯 Economic Insights")
        
        # Dynamic insights based on results
        strongest_policy = policies[impacts.index(max(impacts, key=abs))] if impacts else "None"
        strongest_impact = max(impacts, key=abs) if impacts else 0
        strongest_policy_clean = strongest_policy.split('<br>')[0] if '<br>' in strongest_policy else strongest_policy
        
        if abs(strongest_impact) > 0.2:
            impact_magnitude = "🔥 **Very Strong**"
        elif abs(strongest_impact) > 0.1:
            impact_magnitude = "⚡ **Moderate**"
        else:
            impact_magnitude = "💨 **Mild**"
        
        st.markdown(f"""
        <div style='background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
            <strong>🎯 Most Powerful Tool:</strong> {strongest_policy_clean}<br>
            <strong>📊 Impact Magnitude:</strong> {impact_magnitude}<br>
            <strong>⏱️ Timeline:</strong> 6-18 months for full effect<br>
            <strong>🎪 Total Scenario Impact:</strong> {total_impact:+.3f}pp
        </div>
        """, unsafe_allow_html=True)
        
        # Real-world context
        st.markdown("**🌍 Real-World Context:**")
        if abs(total_impact) > 0.5:
            st.markdown("• This scenario would be **major economic news**")
            st.markdown("• Similar to policy responses during recessions")
        elif abs(total_impact) > 0.2:
            st.markdown("• This would be a **significant policy shift**")
            st.markdown("• Markets would react strongly to these changes")
        else:
            st.markdown("• This would be a **moderate policy adjustment**")
            st.markdown("• Similar to typical Fed meeting decisions")

def create_policy_simulator():
    """Create enhanced interactive policy simulation interface with better layout"""
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
    if simulator is None:
        st.error("❌ Could not initialize policy simulator. Please check data files.")
        return None, None
    
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
    
    # Create three main policy sections with better styling
    st.markdown("### 🎛️ Policy Control Center")
    
    # Monetary Policy Section
    st.markdown("""
    <div style='background-color: #e8f4fd; padding: 1.5rem; border-radius: 0.8rem; border-left: 5px solid #1f77b4; margin: 1rem 0;'>
        <h4 style='color: #1f77b4; margin: 0 0 1rem 0;'>🏦 Monetary Policy Tools</h4>
        <p style='margin: 0; color: #2c3e50;'>The Federal Reserve's primary tools for controlling inflation through interest rates and money supply.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Federal Funds Rate Change**")
        fed_change = st.slider(
            "Adjust Fed Rate",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.25,
            help="The interest rate banks charge each other for overnight loans",
            key="fed_rate_slider"
        )
        
        # Dynamic explanation based on slider value
        if fed_change > 0:
            st.markdown(f"""
            <div style='background-color: #ffeaa7; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>📈 Rate Hike (+{fed_change}%)</strong><br>
                <span style='color: #2d3436;'>
                • Higher borrowing costs for businesses and consumers<br>
                • Reduced spending and investment<br>
                • <strong>Expected effect:</strong> <span style='color: #00b894;'>Reduces inflation</span><br>
                • <strong>Timeline:</strong> 6-18 months for full impact
                </span>
            </div>
            """, unsafe_allow_html=True)
        elif fed_change < 0:
            st.markdown(f"""
            <div style='background-color: #fab1a0; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>📉 Rate Cut ({fed_change}%)</strong><br>
                <span style='color: #2d3436;'>
                • Cheaper borrowing costs<br>
                • Increased spending and investment<br>
                • <strong>Expected effect:</strong> <span style='color: #e17055;'>Increases inflation</span><br>
                • <strong>Timeline:</strong> 3-12 months for impact
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #ddd; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>➡️ No Change (0%)</strong><br>
                <span style='color: #2d3436;'>Fed maintains current policy stance</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Current context
        st.caption("💡 **Current Fed Rate:** ~5.25% (as of 2024)")
        st.caption("📊 **Historical Range:** 0% (2008-2015) to 20% (1980)")
    
    with col2:
        st.markdown("**Money Supply (M2) Growth Change**")
        m2_change = st.slider(
            "Adjust M2 Growth",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            help="Change in the growth rate of money supply (cash + bank deposits)",
            key="m2_slider"
        )
        
        # Dynamic explanation
        if m2_change > 0:
            st.markdown(f"""
            <div style='background-color: #fab1a0; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>💰 Money Supply Expansion (+{m2_change}%)</strong><br>
                <span style='color: #2d3436;'>
                • More money in circulation (Quantitative Easing)<br>
                • Increased liquidity in financial system<br>
                • <strong>Expected effect:</strong> <span style='color: #e17055;'>Increases inflation</span><br>
                • <strong>Mechanism:</strong> "More money chasing same goods"
                </span>
            </div>
            """, unsafe_allow_html=True)
        elif m2_change < 0:
            st.markdown(f"""
            <div style='background-color: #ffeaa7; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>📉 Money Supply Contraction ({m2_change}%)</strong><br>
                <span style='color: #2d3436;'>
                • Reduced money in circulation<br>
                • Tighter financial conditions<br>
                • <strong>Expected effect:</strong> <span style='color: #00b894;'>Reduces inflation</span><br>
                • <strong>Mechanism:</strong> Less money available for spending
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #ddd; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>➡️ No Change (0%)</strong><br>
                <span style='color: #2d3436;'>Money supply grows at normal pace</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption("💡 **Recent M2 Growth:** 2-6% annually (normal)")
        st.caption("📊 **COVID-19 Peak:** +27% in 2020 (unprecedented stimulus)")
    
    # Fiscal Policy Section
    st.markdown("""
    <div style='background-color: #fdf2e9; padding: 1.5rem; border-radius: 0.8rem; border-left: 5px solid #e67e22; margin: 1.5rem 0 1rem 0;'>
        <h4 style='color: #e67e22; margin: 0 0 1rem 0;'>🏛️ Fiscal Policy Tools</h4>
        <p style='margin: 0; color: #2c3e50;'>Government spending and taxation policies that affect economic demand and inflation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("**Fiscal Stimulus (% of GDP)**")
        fiscal_change = st.slider(
            "Government Spending Change",
            min_value=-2.0,
            max_value=3.0,
            value=0.0,
            step=0.5,
            help="Change in government spending as percentage of GDP",
            key="fiscal_slider"
        )
        
        # Dynamic explanation
        if fiscal_change > 0:
            st.markdown(f"""
            <div style='background-color: #fab1a0; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>💸 Fiscal Stimulus (+{fiscal_change}% GDP)</strong><br>
                <span style='color: #2d3436;'>
                • Increased government spending or tax cuts<br>
                • More money in consumers' pockets<br>
                • <strong>Expected effect:</strong> <span style='color: #e17055;'>Increases inflation</span><br>
                • <strong>Examples:</strong> Infrastructure, stimulus checks, tax cuts
                </span>
            </div>
            """, unsafe_allow_html=True)
        elif fiscal_change < 0:
            st.markdown(f"""
            <div style='background-color: #ffeaa7; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>✂️ Fiscal Restraint ({fiscal_change}% GDP)</strong><br>
                <span style='color: #2d3436;'>
                • Reduced government spending or higher taxes<br>
                • Less money in circulation<br>
                • <strong>Expected effect:</strong> <span style='color: #00b894;'>Reduces inflation</span><br>
                • <strong>Examples:</strong> Spending cuts, tax increases
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #ddd; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                <strong>➡️ No Change (0%)</strong><br>
                <span style='color: #2d3436;'>Government maintains current fiscal stance</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption("💡 **US GDP 2024:** ~$27 trillion")
        st.caption("📊 **Major Examples:** CARES Act 2020 (~10% GDP), Infrastructure Bill (~3% GDP)")
    
    with col4:
        st.markdown("**Policy Examples & Real-World Context**")
        
        # Interactive policy examples
        example_type = st.selectbox(
            "See Real Policy Examples:",
            ["Select an example...", "COVID-19 Response (2020)", "2008 Financial Crisis", "1970s Inflation Fight", "Current Policy Debate"]
        )
        
        if example_type == "COVID-19 Response (2020)":
            st.markdown("""
            <div style='background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem;'>
                <strong>🦠 COVID-19 Policy Response</strong><br>
                • <strong>Fed Rate:</strong> Cut to 0% (emergency)<br>
                • <strong>M2 Growth:</strong> +27% (massive QE)<br>
                • <strong>Fiscal:</strong> +15% GDP (CARES Act)<br>
                • <strong>Result:</strong> Inflation rose from 0.1% to 9.1%
            </div>
            """, unsafe_allow_html=True)
        elif example_type == "2008 Financial Crisis":
            st.markdown("""
            <div style='background-color: #ffe8e8; padding: 1rem; border-radius: 0.5rem;'>
                <strong>🏦 2008 Financial Crisis</strong><br>
                • <strong>Fed Rate:</strong> Cut from 5.25% to 0%<br>
                • <strong>M2 Growth:</strong> Moderate QE programs<br>
                • <strong>Fiscal:</strong> +5% GDP stimulus<br>
                • <strong>Result:</strong> Prevented deflation, slow recovery
            </div>
            """, unsafe_allow_html=True)
        elif example_type == "1970s Inflation Fight":
            st.markdown("""
            <div style='background-color: #fff3e0; padding: 1rem; border-radius: 0.5rem;'>
                <strong>⚡ Volcker's Inflation Fight (1979-1982)</strong><br>
                • <strong>Fed Rate:</strong> Raised to 20% (extreme)<br>
                • <strong>M2 Growth:</strong> Severely restricted<br>
                • <strong>Fiscal:</strong> Mixed policies<br>
                • <strong>Result:</strong> Inflation fell from 14% to 3%
            </div>
            """, unsafe_allow_html=True)
        elif example_type == "Current Policy Debate":
            st.markdown("""
            <div style='background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem;'>
                <strong>🗳️ Current Policy Debates (2024-2025)</strong><br>
                • <strong>Fed Rate:</strong> Whether to cut from 5.25%<br>
                • <strong>QE:</strong> When to restart bond purchases<br>
                • <strong>Fiscal:</strong> Infrastructure vs deficit reduction<br>
                • <strong>Challenge:</strong> Balance growth vs inflation
            </div>
            """, unsafe_allow_html=True)
    
    # Scenario Summary Panel
    st.markdown("---")
    st.markdown("### 📊 Scenario Summary & Controls")
    
    col5, col6, col7 = st.columns([1, 1, 1])
    
    with col5:
        st.markdown("**📈 Current Policy Settings**")
        
        # Visual policy summary
        st.metric("Fed Funds Rate Change", f"{fed_change:+.2f}pp", 
                 help="Percentage point change in federal funds rate")
        st.metric("M2 Growth Change", f"{m2_change:+.1f}pp", 
                 help="Change in money supply growth rate")
        st.metric("Fiscal Policy Change", f"{fiscal_change:+.1f}% GDP", 
                 help="Government spending change as % of GDP")
    
    with col6:
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
        
        st.caption("⚠️ This is a rough estimate. Click 'Run Simulation' for precise results.")
    
    with col7:
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
                    
                    # Get baseline forecast
                    baseline_forecast = simulator.baseline_forecast['Prophet_forecast'].values
                    progress_bar.progress(50)
                    
                    # Run shock simulation
                    shock_results = simulator.shock_path(
                        baseline_forecast, 
                        d_ffr=fed_change,
                        d_m2=m2_change, 
                        fiscal=fiscal_change
                    )
                    progress_bar.progress(75)
                    
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
                    progress_bar.progress(100)
                    
                    st.success("✅ **Policy simulation completed successfully!**")
                    st.balloons()  # Celebration effect
                    
                except Exception as e:
                    st.error(f"❌ **Simulation failed:** {e}")
                    st.info("💡 Try reloading the page or checking your data files.")
                    scenario_results = None
                finally:
                    progress_bar.empty()
    
    return scenario_results, {
        'fed_change': fed_change,
        'm2_change': m2_change, 
        'fiscal_change': fiscal_change
    }

def display_scenario_results(scenario_results, policy_settings):
    """Display comprehensive scenario analysis results with enhanced layout"""
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
    baseline_avg = np.mean(scenario_results['baseline_forecast'])
    scenario_avg = np.mean(scenario_results['new_forecast'])
    total_impact = np.mean(scenario_results['contributions']['total_shock'])
    
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
                <strong>Baseline Average</strong><br>
                <span style='font-size: 1.5em; color: #2c3e50;'>{baseline_avg:.2f}%</span>
            </div>
            <div style='font-size: 2em; color: {impact_color}; align-self: center;'>→</div>
            <div>
                <strong>Scenario Average</strong><br>
                <span style='font-size: 1.5em; color: {impact_color};'>{scenario_avg:.2f}%</span>
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌪️ Policy Breakdown", 
        "📈 Timeline Analysis", 
        "📊 Detailed Data", 
        "🎯 Economic Context"
    ])
    
    with tab1:
        st.markdown("### 🌪️ Which Policies Drive the Impact?")
        
        # Enhanced tornado plot
        create_tornado_plot(scenario_results)
    
    with tab2:
        st.markdown("### 📈 How Policy Effects Build Over Time")
        
        # Timeline chart
        create_policy_timeline_chart(scenario_results)
    
    with tab3:
        st.markdown("### 📊 Detailed Forecast Data & Comparison")
        
        # Enhanced comparison table
        create_scenario_comparison_table(scenario_results)
    
    with tab4:
        st.markdown("### 🎯 Economic Context & Real-World Implications")
        
        st.markdown("**Real-world context and policy implications for your scenario.**")
        
        if abs(total_impact) > 0.5:
            st.markdown("🔥 **This is a MAJOR policy scenario** comparable to crisis responses.")
        elif abs(total_impact) > 0.2:
            st.markdown("⚡ **This is a SIGNIFICANT policy shift** that would make headlines.")
        else:
            st.markdown("💨 **This represents a MODERATE adjustment** typical of normal policy cycles.")

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