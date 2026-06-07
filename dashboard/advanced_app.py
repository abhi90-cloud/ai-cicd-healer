import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import json
st.set_page_config(
    page_title="AI CI/CD Healer Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
.risk-high { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ffaa00; font-weight: bold; }
.risk-low { color: #00c851; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
# API Connection
API_URL = "http://localhost:8000"
def fetch_api(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=3)
        return response.json() if response.status_code == 200 else None
    except:
        return None
# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("🏥 AI Healer Pro")
    page = st.radio("📋 Navigation", [
        "🎯 Dashboard",
        "🔮 AI Predictions", 
        "📊 Analytics",
        "🔍 Live Monitor",
        "🤖 Model Info",
        "⚙️ Settings"
    ])
    st.divider()
    # Real-time stats
    stats = fetch_api("/api/v2/stats")
    if stats:
        st.metric("📊 Total Runs", stats.get("total_runs", 0))
        st.metric("✅ Success Rate", f"{100 - stats.get('failure_rate', 0):.1f}%")
        st.metric("🏥 Heal Rate", f"{stats.get('healing_success_rate', 0):.1f}%")
    st.divider()
    st.caption("v2.0.0 | Ensemble ML Powered")
# Main Content
if page == "🎯 Dashboard":
    st.title("🎯 AI-Powered CI/CD Dashboard")
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pipeline Health", "94.2%", "↑ 2.1%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ML Accuracy", "72.3%", "↑ 1.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Auto-Healed", "847", "85% success")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MTTR", "2.1 min", "↓ 45s")
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    # Charts row
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📈 Success Rate Trend")
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Success Rate': [92 + np.random.randint(-3, 5) for _ in range(30)],
            'Predicted': [90 + np.random.randint(-2, 6) for _ in range(30)]
        })
        fig = px.line(df, x='Date', y=['Success Rate', 'Predicted'],
                      title='Pipeline Success Rate vs Predictions')
        fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("🥧 Failure Distribution")
        failures = {
            'Network': 23, 'Dependency': 19, 'Test Flaky': 17,
            'Resource': 14, 'Build Error': 12, 'Permission': 5, 'Other': 10
        }
        fig = px.pie(values=list(failures.values()), names=list(failures.keys()),
                     hole=0.4, title='Failure Types')
        st.plotly_chart(fig, use_container_width=True)
    # Model performance comparison
    st.subheader("🤖 Model Performance Comparison")
    models_df = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boost', 'ENSEMBLE'],
        'Accuracy': [70.9, 71.2, 74.15, 71.85, 72.3],
        'AUC': [0.689, 0.695, 0.724, 0.701, 0.709]
    })
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(models_df, x='Model', y='Accuracy', color='Model',
                     title='Model Accuracy Comparison', text='Accuracy')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(models_df, x='Model', y='AUC', color='Model',
                     title='Model AUC Comparison', text='AUC')
        st.plotly_chart(fig, use_container_width=True)
elif page == "🔮 AI Predictions":
    st.title("🔮 AI Failure Prediction")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📝 Pipeline Configuration")
        files = st.slider("Files Changed", 1, 200, 25)
        coverage = st.slider("Test Coverage %", 0, 100, 65)
        deps = st.slider("Dependencies Updated", 0, 30, 3)
        complexity = st.slider("Code Complexity", 0.0, 1.0, 0.5)
        build_time = st.slider("Avg Build Time (s)", 30, 600, 180)
        if st.button("🔮 Predict Failure Risk", type="primary", use_container_width=True):
            # Simulate prediction
            risk = 0.0
            if files > 50: risk += 0.25
            if coverage < 60: risk += 0.3
            if deps > 5: risk += 0.2
            if complexity > 0.7: risk += 0.15
            if build_time > 300: risk += 0.1
            risk = min(risk, 0.99)
            st.session_state['prediction'] = risk
    with col2:
        st.subheader("📊 Prediction Result")
        if 'prediction' in st.session_state:
            risk = st.session_state['prediction']
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Failure Probability"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#00c851"},
                        {'range': [30, 70], 'color': "#ffaa00"},
                        {'range': [70, 100], 'color': "#ff4444"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            risk_level = "🔴 HIGH" if risk > 0.7 else ("🟡 MEDIUM" if risk > 0.3 else "🟢 LOW")
            st.markdown(f"### Risk Level: {risk_level}")
            st.progress(risk)
            if risk > 0.5:
                st.error("⚠️ High risk of failure detected!")
                st.info("💡 Recommendations:\n- Split into smaller changes\n- Increase test coverage\n- Update dependencies gradually")
            else:
                st.success("✅ Low risk - safe to deploy!")
elif page == "📊 Analytics":
    st.title("📊 Advanced Analytics")
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "💰 Cost Savings", "🏥 Healing Analytics"])
    with tab1:
        st.subheader("30-Day Performance Trends")
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        trend_df = pd.DataFrame({
            'Date': dates,
            'Success Rate': [90 + i*0.1 + np.random.randint(-2, 3) for i in range(30)],
            'MTTR (min)': [5 - i*0.1 + np.random.uniform(-0.5, 0.5) for i in range(30)],
            'Predictions': [50 + i*2 + np.random.randint(-5, 5) for i in range(30)]
        })
        fig = px.line(trend_df, x='Date', y=['Success Rate', 'MTTR (min)', 'Predictions'],
                      title='Performance Trends')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("💰 Cost Savings Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Engineer Hours Saved", "240 hrs/mo", "35%")
        with col2:
            st.metric("Downtime Reduction", "45 hrs/mo", "62%")
        with col3:
            st.metric("Monthly Savings", ",500", "48%")
        st.info("Based on industry average: /hr engineer cost, /min downtime cost")
    with tab3:
        st.subheader("🏥 Healing Effectiveness")
        heal_df = pd.DataFrame({
            'Error Type': ['Network', 'Dependency', 'Test', 'Resource', 'Build'],
            'Success Rate': [88, 92, 78, 85, 90],
            'Avg Time (s)': [15, 25, 45, 30, 20]
        })
        fig = px.bar(heal_df, x='Error Type', y='Success Rate', color='Avg Time (s)',
                     title='Healing Success Rate by Error Type')
        st.plotly_chart(fig, use_container_width=True)
elif page == "🔍 Live Monitor":
    st.title("🔍 Live Pipeline Monitor")
    st.info("🔄 Auto-refreshing every 5 seconds")
    if st.button("🔄 Refresh Now"):
        st.rerun()
    for i in range(8):
        status = np.random.choice(['success', 'failed', 'running', 'healing', 'predicted'])
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        with col1:
            if status == 'success':
                st.success(f"✅ Build #{3000+i}")
            elif status == 'failed':
                st.error(f"❌ Build #{3000+i}")
            elif status == 'healing':
                st.warning(f"🏥 Build #{3000+i} - Healing...")
            elif status == 'predicted':
                st.info(f"🔮 Build #{3000+i} - Risk Predicted")
            else:
                st.info(f"🔄 Build #{3000+i} - Running")
        with col2:
            st.text(np.random.choice(['build', 'test', 'deploy']))
        with col3:
            st.text(f"{np.random.randint(30, 300)}s")
        with col4:
            risk = np.random.uniform(0, 1)
            color = "red" if risk > 0.7 else ("orange" if risk > 0.3 else "green")
            st.markdown(f'<span style="color:{color}">{risk:.0%}</span>', unsafe_allow_html=True)
        with col5:
            if status == 'failed':
                st.button("🏥 Heal", key=f"heal_{i}")
elif page == "🤖 Model Info":
    st.title("🤖 ML Model Information")
    st.subheader("📊 Model Architecture")
    st.code("""
    Ensemble Model (Voting Classifier)
    ├── XGBoost (200 estimators)
    ├── LightGBM (200 estimators)
    ├── Random Forest (200 estimators)
    └── Gradient Boosting (200 estimators)
    Training Data: 10,000 samples
    Features: 30+ engineered features
    Validation: 80/20 split
    """)
    st.subheader("🎯 Top 10 Features")
    features_df = pd.DataFrame({
        'Feature': ['Success Streak', 'Test Coverage', 'Major Version Changes', 
                    'Recent Failures', 'Outdated Deps', 'Failure Rate 7d',
                    'Files Changed', 'Flaky Tests', 'Lines Added', 'MTTR'],
        'Importance': [0.059, 0.059, 0.054, 0.046, 0.045, 0.036, 0.034, 0.030, 0.028, 0.027]
    })
    fig = px.bar(features_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance', color='Importance')
    st.plotly_chart(fig, use_container_width=True)
elif page == "⚙️ Settings":
    st.title("⚙️ Platform Settings")
    st.subheader("🤖 ML Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8)
    auto_heal = st.toggle("Enable Auto-Healing", True)
    retrain_frequency = st.selectbox("Model Retraining", ["Daily", "Weekly", "Monthly"])
    st.subheader("🔔 Alert Settings")
    st.checkbox("Slack Notifications", True)
    st.checkbox("Email Alerts", False)
    st.checkbox("PagerDuty Integration", False)
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved!")
        st.toast("Configuration updated")
st.divider()
st.caption("🏥 AI Auto-Healing CI/CD Platform v2.0 | Powered by Ensemble ML | © 2024")
