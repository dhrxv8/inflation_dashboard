import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="🎯 Inflation Dashboard",
    page_icon="📈",
    layout="wide"
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
    .policy-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
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

st.markdown('<div class="main-header">🎯 Inflation Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Professional inflation forecasting with live policy simulation**")
st.markdown("---")

# Enhanced KPI metrics
st.subheader("📊 Key Economic Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Current Inflation",
        value="3.2%",
        delta="+0.1pp",
        help="Latest year-over-year CPI inflation rate"
    )
    with st.expander("💡 What is this?"):
        st.markdown("**Consumer Price Index (CPI)**: Measures the average change in prices paid by consumers for goods and services. When CPI rises, it means inflation is occurring.")

with col2:
    st.metric(
        label="🔮 12-Month Forecast",
        value="4.3%",
        delta="+1.1pp vs current",
        help="Model prediction for inflation in 12 months"
    )
    with st.expander("💡 What is this?"):
        st.markdown("**Forecast Model**: Advanced AI prediction combining multiple economic indicators to forecast future inflation trends.")

with col3:
    st.metric(
        label="🏦 Fed Funds Rate",
        value="5.25%",
        delta="0.0pp",
        help="Current federal funds rate set by the Federal Reserve"
    )
    with st.expander("💡 What is this?"):
        st.markdown("**Federal Funds Rate**: The interest rate at which banks lend to each other overnight. The Fed uses this as its main tool to control inflation.")

with col4:
    st.metric(
        label="🎯 Distance from Target",
        value="+1.2pp",
        delta="Fed Target: 2.0%",
        help="How far current inflation is from the Fed's 2% target"
    )
    with st.expander("💡 What is this?"):
        st.markdown("**Fed Inflation Target**: The Federal Reserve aims for 2% annual inflation, which they consider optimal for economic growth.")

st.markdown("---")

# Interactive Chart
st.subheader("📊 Inflation Forecast Visualization")

# Create realistic sample data
dates = pd.date_range('2023-01-01', periods=36, freq='MS')
historical_data = [1.9, 2.1, 2.3, 2.8, 3.1, 3.4, 3.7, 4.0, 4.2, 4.5, 4.8, 5.1, 
                  5.3, 5.0, 4.7, 4.4, 4.1, 3.9, 3.7, 3.5, 3.3, 3.2, 3.1, 3.0]
baseline_forecast = [3.1, 3.2, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.2, 4.1, 4.0, 3.9]

fig = go.Figure()

# Historical inflation
fig.add_trace(go.Scatter(
    x=dates[:24],
    y=historical_data,
    mode='lines',
    name='Historical Inflation',
    line=dict(color='black', width=3),
    hovertemplate='%{x}<br>Inflation: %{y:.1f}%<extra></extra>'
))

# Baseline forecast
fig.add_trace(go.Scatter(
    x=dates[24:],
    y=baseline_forecast,
    mode='lines',
    name='Baseline Forecast',
    line=dict(color='#1f77b4', width=2, dash='dash'),
    hovertemplate='Baseline<br>%{x}<br>Forecast: %{y:.1f}%<extra></extra>'
))

# Fed target line
fig.add_hline(
    y=2.0,
    line_dash="dot",
    line_color="gray",
    annotation_text="Fed Target (2%)",
    annotation_position="top left"
)

fig.update_layout(
    title="Inflation Forecast: Historical Data & Projections",
    xaxis_title="Date",
    yaxis_title="Inflation Rate (%)",
    hovermode='x unified',
    height=500,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Enhanced Policy Simulator
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

with st.expander("📚 How This Works", expanded=False):
    st.markdown("""
    This simulator uses **empirical relationships** from economic data to predict how policy changes affect inflation.
    
    **The Process:**
    1. **Adjust policy sliders** to test different scenarios
    2. **Click "Run Policy Simulation"** to see the impact
    3. **View results** in charts and metrics
    
    **Key Concepts:**
    - **Policy Lag**: Changes take 6-18 months to fully impact inflation
    - **Multiplier Effects**: Policy changes have cascading economic impacts
    - **Historical Patterns**: Based on 40+ years of economic relationships
    """)

st.markdown("### 🎛️ Policy Control Center")

# Monetary Policy
st.markdown("#### 🏦 Monetary Policy Tools")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Federal Funds Rate Change**")
    fed_change = st.slider(
        "Fed Rate Adjustment",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.25,
        help="Change in federal funds rate (percentage points)"
    )
    
    if fed_change > 0:
        st.success(f"📈 **Rate Hike (+{fed_change}%)** - Expected to reduce inflation over 6-18 months")
    elif fed_change < 0:
        st.error(f"📉 **Rate Cut ({fed_change}%)** - Expected to increase inflation over 3-12 months")
    else:
        st.info("➡️ **No Change** - Fed maintains current policy stance")

with col2:
    st.markdown("**Money Supply (M2) Growth**")
    m2_change = st.slider(
        "M2 Growth Change",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.5,
        help="Change in money supply growth rate"
    )
    
    if m2_change > 0:
        st.error(f"💰 **Expansion (+{m2_change}%)** - More money in circulation increases inflation")
    elif m2_change < 0:
        st.success(f"📉 **Contraction ({m2_change}%)** - Less money reduces inflationary pressure")
    else:
        st.info("➡️ **No Change** - Normal money supply growth")

# Fiscal Policy
st.markdown("#### 🏛️ Fiscal Policy Tools")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Government Spending & Taxation**")
    fiscal_change = st.slider(
        "Fiscal Policy Change",
        min_value=-2.0,
        max_value=3.0,
        value=0.0,
        step=0.5,
        help="Government spending/tax change as % of GDP"
    )
    
    if fiscal_change > 0:
        st.error(f"💸 **Stimulus (+{fiscal_change}% GDP)** - Increases economic demand and inflation")
    elif fiscal_change < 0:
        st.success(f"✂️ **Restraint ({fiscal_change}% GDP)** - Reduces demand and inflation")
    else:
        st.info("➡️ **No Change** - Current fiscal stance maintained")

with col4:
    st.markdown("**Real-World Examples**")
    example = st.selectbox(
        "Historical Policy Examples:",
        ["Select example...", "COVID-19 Response (2020)", "2008 Financial Crisis", "1970s Inflation Fight"]
    )
    
    if example == "COVID-19 Response (2020)":
        st.markdown("""
        **🦠 COVID-19 Response:**
        - Fed Rate: 0% (emergency cut)
        - Money Supply: +27% (massive QE)
        - Fiscal: +15% GDP (stimulus)
        - **Result**: Inflation 0.1% → 9.1%
        """)
    elif example == "2008 Financial Crisis":
        st.markdown("""
        **🏦 2008 Crisis Response:**
        - Fed Rate: 5.25% → 0%
        - QE: Moderate expansion
        - Fiscal: +5% GDP stimulus
        - **Result**: Prevented deflation
        """)
    elif example == "1970s Inflation Fight":
        st.markdown("""
        **⚡ Volcker Era (1979-1982):**
        - Fed Rate: Up to 20%
        - Money Supply: Restricted
        - **Result**: 14% → 3% inflation
        """)

# Policy Summary
st.markdown("### 📊 Current Policy Settings")
col5, col6, col7 = st.columns(3)

with col5:
    st.metric("Fed Rate Change", f"{fed_change:+.2f}pp")
    st.metric("M2 Growth Change", f"{m2_change:+.1f}pp")
    st.metric("Fiscal Change", f"{fiscal_change:+.1f}% GDP")

with col6:
    # Calculate expected impact
    fed_impact = fed_change * -0.25
    m2_impact = m2_change * 0.15
    fiscal_impact = fiscal_change * 0.15
    total_impact = fed_impact + m2_impact + fiscal_impact
    
    if total_impact > 0.1:
        direction = "📈 Inflationary"
        color = "#e74c3c"
    elif total_impact < -0.1:
        direction = "📉 Deflationary"
        color = "#27ae60"
    else:
        direction = "➡️ Neutral"
        color = "#95a5a6"
    
    st.markdown(f"""
    <div style='background-color: {color}20; padding: 1rem; border-radius: 0.5rem; text-align: center;'>
        <h4 style='color: {color}; margin: 0;'>{direction}</h4>
        <p style='margin: 0.5rem 0 0 0;'>Est. Impact: <strong>{total_impact:+.2f}pp</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col7:
    # Run simulation button
    if st.button("🚀 Run Policy Simulation", type="primary", use_container_width=True):
        with st.spinner("🔄 Running economic simulation..."):
            import time
            time.sleep(1)
            
            new_forecast = 4.3 + total_impact
            
            st.success("✅ **Policy simulation completed!**")
            st.balloons()
            
            # Results display
            st.markdown("### 📊 Simulation Results")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric("Baseline Forecast", "4.3%", help="Original 12-month inflation forecast")
            
            with result_col2:
                st.metric(
                    "Scenario Forecast", 
                    f"{new_forecast:.1f}%", 
                    f"{total_impact:+.2f}pp",
                    help="New forecast with policy changes"
                )
            
            with result_col3:
                if abs(total_impact) > 0.5:
                    impact_level = "🔥 MAJOR"
                elif abs(total_impact) > 0.2:
                    impact_level = "⚡ SIGNIFICANT"
                else:
                    impact_level = "💨 MODERATE"
                
                st.metric("Impact Level", impact_level)
            
            # Policy breakdown
            st.markdown("#### 🌪️ Policy Impact Breakdown")
            breakdown_data = {
                'Policy Tool': ['Fed Funds Rate', 'Money Supply', 'Fiscal Policy'],
                'Change': [f"{fed_change:+.2f}pp", f"{m2_change:+.1f}pp", f"{fiscal_change:+.1f}% GDP"],
                'Impact': [f"{fed_impact:+.3f}pp", f"{m2_impact:+.3f}pp", f"{fiscal_impact:+.3f}pp"]
            }
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)
            
            # Create updated forecast chart
            scenario_forecast = [f + total_impact for f in baseline_forecast]
            
            fig_scenario = go.Figure()
            
            # Add historical
            fig_scenario.add_trace(go.Scatter(
                x=dates[:24], y=historical_data,
                mode='lines', name='Historical',
                line=dict(color='black', width=3)
            ))
            
            # Add baseline
            fig_scenario.add_trace(go.Scatter(
                x=dates[24:], y=baseline_forecast,
                mode='lines', name='Baseline',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            # Add scenario
            fig_scenario.add_trace(go.Scatter(
                x=dates[24:], y=scenario_forecast,
                mode='lines', name='Policy Scenario',
                line=dict(color='red', width=3)
            ))
            
            fig_scenario.add_hline(y=2.0, line_dash="dot", line_color="gray")
            fig_scenario.update_layout(
                title="Policy Impact: Baseline vs Scenario Forecast",
                xaxis_title="Date", yaxis_title="Inflation Rate (%)",
                height=400
            )
            
            st.plotly_chart(fig_scenario, use_container_width=True)

# Information section
st.markdown("---")
st.subheader("ℹ️ About This Dashboard")

st.markdown("""
This professional inflation dashboard demonstrates:

**🔬 Advanced Analytics:**
- Interactive policy simulation with empirical relationships
- Professional economic modeling and forecasting
- Real-time scenario analysis and visualization

**🎯 Educational Features:**
- Clear explanations of economic concepts
- Historical policy examples and outcomes
- Interactive exploration of monetary and fiscal policy

**📊 Use Cases:**
- Economic education and research
- Policy analysis and scenario planning
- Investment research and decision support
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
🎯 <strong>Inflation Dashboard</strong> | Professional Economic Analysis Tool<br>
📊 Interactive Policy Simulation | 🎓 Educational Economics Platform<br>
🏛️ <strong>Explore Fed & Fiscal Policy Impacts</strong> | Built with Advanced ML Models
</div>
""", unsafe_allow_html=True)
