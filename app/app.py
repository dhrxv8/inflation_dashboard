import streamlit as st

st.set_page_config(
    page_title="🎯 Inflation Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🎯 Inflation Dashboard - Deployment Test")
st.success("✅ Streamlit Cloud deployment successful!")

st.markdown("### 📊 Basic Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Inflation", "3.2%", "+0.1pp")

with col2:
    st.metric("Fed Rate", "5.25%", "0.0pp") 

with col3:
    st.metric("Target Distance", "+1.2pp")

if st.button("Test Button"):
    st.balloons()
    st.success("🎉 Interactive features working!")

st.markdown("---")
st.info("**Next step**: Gradually add more features once this basic version is deployed successfully.")
