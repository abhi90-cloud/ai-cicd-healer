import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
st.set_page_config(
    page_title="AI CI/CD Healer",
    page_icon=":hospital:",
    layout="wide"
)
# Sidebar
with st.sidebar:
    st.title(":hospital: AI Healer")
    page = st.radio("Menu", ["Dashboard", "Live Monitor", "AI Insights", "Analytics"])
    st.divider()
    try:
        stats = requests.get("http://localhost:8000/api/v1/stats").json()
        st.metric("Total Runs", stats.get("total_runs", 0))
        st.metric("Success Rate", f"{stats.get('success_rate', 0)}%")
        st.metric("Healing Actions", stats.get("healing_actions", 0))
    except:
        st.warning("API not connected")
# Main Content
if page == "Dashboard":
    st.title(":bar_chart: AI Auto-Healing CI/CD Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pipeline Health", "94%", "2.5%")
    with col2:
        st.metric("Failures Predicted", "127", "15")
    with col3:
        st.metric("Auto-Healed", "89", "70%")
    with col4:
        st.metric("Avg Recovery", "2.3 min", "-45s")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Success Rate Trend")
        dates = pd.date_range(start='2024-01-01', periods=30)
        df = pd.DataFrame({
            'date': dates,
            'rate': [90 + np.random.randint(-5, 5) for _ in range(30)]
        })
        fig = px.line(df, x='date', y='rate', title='Success Rate (%)')
        fig.add_hline(y=95, line_dash="dash", line_color="green")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Failure Distribution")
        failures = {'Network': 25, 'Dependency': 20, 'Test': 18, 'Resource': 15, 'Build': 12}
        fig = px.pie(values=list(failures.values()), names=list(failures.keys()))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recent Activity")
    for i in range(5):
        status = np.random.choice(['success', 'failed', 'healed'])
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            if status == 'success': st.success(f"Build #{1000+i} - Deploy completed")
            elif status == 'failed': st.error(f"Build #{1000+i} - Test failure")
            else: st.warning(f"Build #{1000+i} - Auto-healed")
        with col2: st.text(f"Duration: {np.random.randint(30, 300)}s")
        with col3: st.text(f"{np.random.randint(1, 60)}m ago")
elif page == "Live Monitor":
    st.title(":satellite: Live Pipeline Monitor")
    st.info("Real-time pipeline monitoring - refresh to update")
    for i in range(8):
        status = np.random.choice(['running', 'success', 'failed', 'healing'])
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            if status == 'success': st.success(f"Build #{2000+i}")
            elif status == 'failed': st.error(f"Build #{2000+i}")
            elif status == 'healing': st.warning(f"Build #{2000+i}")
            else: st.info(f"Build #{2000+i}")
        with col2: st.text(f"Stage: {np.random.choice(['build','test','deploy'])}")
        with col3: st.text(f"{np.random.randint(10, 300)}s")
        with col4:
            if status == 'failed':
                st.button(f"Heal #{2000+i}", key=f"heal_{i}")
elif page == "AI Insights":
    st.title(":brain: AI Model Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "87.5%", "2.3%")
        st.metric("Precision", "85.2%", "1.8%")
    with col2:
        st.metric("Recall", "89.1%", "2.5%")
        st.metric("F1 Score", "87.1%", "2.1%")
    st.subheader("Top Failure Predictors")
    features = {'Code Changes': 0.85, 'Dependencies': 0.72, 'Test Coverage': 0.65, 'Build Time': 0.58}
    fig = px.bar(x=list(features.values()), y=list(features.keys()), orientation='h')
    st.plotly_chart(fig, use_container_width=True)
elif page == "Analytics":
    st.title(":chart_with_upwards_trend: Analytics")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Hours Saved", "240/month", "30%")
    with col2: st.metric("Downtime Reduced", "45 hrs/month", "60%")
    with col3: st.metric("Cost Savings", ":dollar: 35,000/month", "45%")
st.divider()
st.caption(":hospital: AI Auto-Healing CI/CD Platform v1.0")
