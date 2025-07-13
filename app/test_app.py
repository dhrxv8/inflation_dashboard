import streamlit as st

st.title("🎯 Test Inflation Dashboard")
st.write("Hello! This is a test version.")

st.metric("Test Metric", "3.2%", "0.1%")

import pandas as pd
import numpy as np

# Create some test data
dates = pd.date_range('2023-01-01', periods=12, freq='M')
values = np.random.normal(3, 0.5, 12)
df = pd.DataFrame({'date': dates, 'inflation': values})

st.line_chart(df.set_index('date'))

st.success("✅ Streamlit is working!")
