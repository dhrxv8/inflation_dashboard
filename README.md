# InflationScope

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inflationdashboard.streamlit.app)  
![Version](https://img.shields.io/badge/version-v1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)

> **A personal project exploring inflation forecasting and monetary policy simulation**

---

## Live Demo  
**[Launch the Dashboard](https://inflationdashboard.streamlit.app/)**

---

## Overview: Why I Built This

As someone fascinated by both economics and data science, I wanted to build something that blends these interests. This dashboard simulates how different Federal Reserve policies impact inflation, helping users (including myself) understand policy impacts in real-time.

This answers questions like:
- What would happen if the Fed raised interest rates by 2%?
- How much does government spending actually affect inflation?

---

## Key Features (v1.0)

### Interactive Policy Playground
- Adjust sliders for interest rates, money supply, and fiscal spending
- Simulate policy effects on inflation forecasts
- Explore historical crises (e.g., COVID-19, 2008, 1970s inflation)

### Forecasting Engine
- Models used: Prophet, SARIMA, and an Ensemble Average
- Pulls over 20 real indicators from the Federal Reserve
- Produces 12-month inflation forecasts with confidence intervals

### Accessible Learning Tool
- Clickable explanations for each economic term
- Real-world examples tied to historical events
- No prior economics background needed

---

## Technical Details

### Data Science
- Ingested 20+ macroeconomic time series via the FRED API
- Engineered 112+ time-series features
- Trained multiple time series models
- Calibrated simulation effects using 40+ years of empirical data

### Web Application
- Framework: Streamlit
- Interactive plotting: Plotly
- Custom CSS for polished UI
- Deployed on Streamlit Cloud with continuous GitHub integration

### Economics
- Incorporated real monetary/fiscal policy research
- Modeled policy lag effects (to reflect realistic delays)
- Grounded simulations in observed macroeconomic behavior

---

## Directory Structure

```
my-inflation-dashboard/
├── app/
│   └── app.py             # Main Streamlit interface
├── src/
│   ├── modeling.py        # Forecasting models
│   ├── simulator.py       # Policy simulation engine
│   ├── features.py        # Feature engineering
│   └── data_ingest.py     # FRED data ingestion
├── data/                  # Processed data and saved models
├── notebooks/             # Research and experimentation
└── requirements.txt       # Project dependencies
```

---

## How to Use

1. View current inflation and Fed policy metrics
2. Adjust policy levers using sliders
3. Run the simulation to generate impact forecasts
4. Interpret results through dynamic charts and data summaries
5. Use the educational content to understand the economics

---

## Highlights

- Realistic, empirically grounded simulation logic
- Millisecond-level policy simulations
- Model ensemble improves prediction robustness
- Fully interactive dashboard interface
- Clean, open-source codebase for transparency and learning

---

## Project Stats

- **306 months** of historical data (1995–2025)
- **112+ engineered features** from macro indicators
- **3 forecast models** integrated
- **Handles edge cases** and outlier policies without breaking

---

## What I Learned

This project helped solidify my skills in:
- Real-world economic data processing
- Advanced time series modeling (Prophet, SARIMA, ensembles)
- Full-stack data science development
- Monetary and fiscal policy mechanics
- Managing a complete project lifecycle: from research to deployment

---

## Run It Yourself

### Quick Start
Use the live version here:  
**[https://inflationdashboard.streamlit.app/](https://inflationdashboard.streamlit.app/)**

### Local Setup

```bash
git clone https://github.com/dhrxv8/inflation_dashboard.git
cd inflation_dashboard
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Technologies Used

- **Python 3.11+**
- **Streamlit** – Web application interface
- **Pandas, NumPy** – Data handling and transformation
- **Plotly** – Interactive charting
- **Prophet, Statsmodels** – Time series forecasting
- **FRED API** – Real-time economic data

---

## Contact

- GitHub: [@dhrxv8](https://github.com/dhrxv8)
- LinkedIn: [https://www.linkedin.com/in/dhruvrao29/](https://www.linkedin.com/in/dhruvrao29/)
- Email: drao18@asu.edu
