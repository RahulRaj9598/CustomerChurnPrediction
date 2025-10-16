import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib, pickle, warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    font-weight: bold;
    margin-bottom: 2rem;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 1rem 0;
}
.success-prediction {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}
.danger-prediction {
    background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Data and Model
# ------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tel_churn.csv")
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        return df
    except FileNotFoundError:
        st.error("❌ 'tel_churn.csv' not found. Place it in the same directory.")
        return None

@st.cache_resource
def load_model():
    try:
        # Load model metadata
        with open("model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Try to load any available model file
        import os
        model_files = [f for f in os.listdir('.') if f.startswith('best_churn_model_') and f.endswith('.pkl')]
        
        if not model_files:
            st.error("❌ No trained model found. Please train a model first.")
            return None, None
        
        # Prioritize tuned models over regular models
        tuned_models = [f for f in model_files if 'tuned' in f]
        regular_models = [f for f in model_files if 'tuned' not in f]
        
        # Try tuned models first, then regular models
        preferred_order = tuned_models + regular_models
        
        for model_file in preferred_order:
            try:
                model = joblib.load(model_file)
                st.success(f"✅ Loaded model: {model_file}")
                return model, metadata
            except Exception as e:
                st.warning(f"⚠️ Could not load {model_file}: {e}")
                continue
        
        st.error("❌ Could not load any model file.")
        return None, None
        
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ------------------------------------------------------------
# Sidebar Input Form
# ------------------------------------------------------------
def customer_input_form():
    """Sidebar input form for customer attributes."""
    st.sidebar.header("🎯 Customer Information")
    st.sidebar.markdown("Fill in details below to simulate a customer profile for churn prediction.")

    try:
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except:
        st.error("Feature names file missing.")
        return None

    inputs = {}

    # --- Demographics ---
    st.sidebar.markdown("### 👤 Demographics")
    st.sidebar.caption("Basic customer information like age group and gender.")
    inputs["SeniorCitizen"] = st.sidebar.selectbox("Is the customer a senior citizen?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    gender = st.sidebar.radio("Select Gender", ["Female", "Male"])
    inputs["gender_Female"], inputs["gender_Male"] = (1, 0) if gender == "Female" else (0, 1)

    # --- Family ---
    st.sidebar.markdown("### 👨‍👩‍👧‍👦 Family Status")
    st.sidebar.caption("Information about partner and dependents.")
    partner = st.sidebar.radio("Has a Partner?", ["Yes", "No"])
    dependents = st.sidebar.radio("Has Dependents?", ["Yes", "No"])
    inputs["Partner_Yes"], inputs["Partner_No"] = (1, 0) if partner == "Yes" else (0, 1)
    inputs["Dependents_Yes"], inputs["Dependents_No"] = (1, 0) if dependents == "Yes" else (0, 1)

    # --- Services ---
    st.sidebar.markdown("### 📞 Phone Services")
    st.sidebar.caption("Customer’s connection type and line usage.")
    phone = st.sidebar.radio("Phone Service Available?", ["Yes", "No"])
    multi_line = st.sidebar.selectbox("Multiple Lines Availability", ["Yes", "No", "No phone service"])
    inputs["PhoneService_Yes"], inputs["PhoneService_No"] = (1, 0) if phone == "Yes" else (0, 1)
    for opt in ["Yes", "No", "No phone service"]:
        inputs[f"MultipleLines_{opt}"] = 1 if multi_line == opt else 0

    # --- Internet ---
    st.sidebar.markdown("### 🌐 Internet Service")
    st.sidebar.caption("Customer’s internet connection type.")
    internet = st.sidebar.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No Internet"])
    for opt in ["DSL", "Fiber optic", "No"]:
        inputs[f"InternetService_{opt}"] = 1 if internet.startswith(opt) else 0

    # --- Online Services ---
    st.sidebar.markdown("### 🔒 Online Add-on Services")
    st.sidebar.caption("Extra services like online backup, tech support, etc.")
    for service in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]:
        choice = st.sidebar.selectbox(f"{service.replace('Online', 'Online ')}", ["Yes", "No", "No internet service"])
        for opt in ["Yes", "No", "No internet service"]:
            inputs[f"{service}_{opt}"] = 1 if choice == opt else 0

    # --- Streaming ---
    st.sidebar.markdown("### 📺 Streaming Preferences")
    for service in ["StreamingTV", "StreamingMovies"]:
        choice = st.sidebar.selectbox(f"{service.replace('Streaming', 'Streaming ')}", ["Yes", "No", "No internet service"])
        for opt in ["Yes", "No", "No internet service"]:
            inputs[f"{service}_{opt}"] = 1 if choice == opt else 0

    # --- Billing ---
    st.sidebar.markdown("### 💳 Billing & Payment Details")
    st.sidebar.caption("Contract type and payment preferences.")
    contract = st.sidebar.radio("Contract Duration", ["Month-to-month", "One year", "Two year"])
    for opt in ["Month-to-month", "One year", "Two year"]:
        inputs[f"Contract_{opt}"] = 1 if contract == opt else 0

    paperless = st.sidebar.radio("Paperless Billing Enabled?", ["Yes", "No"])
    inputs["PaperlessBilling_Yes"], inputs["PaperlessBilling_No"] = (1, 0) if paperless == "Yes" else (0, 1)

    payment = st.sidebar.selectbox("Preferred Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    for method in ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]:
        inputs[f"PaymentMethod_{method}"] = 1 if payment == method else 0

    # --- Financial Info ---
    st.sidebar.markdown("### 💰 Financial Details")
    st.sidebar.caption("Charges and total amount billed.")
    inputs["MonthlyCharges"] = st.sidebar.slider("Monthly Charges ($)", 18.25, 118.75, 65.00, 0.01)
    inputs["TotalCharges"] = st.sidebar.number_input("Total Charges ($)", 18.8, 8684.8, 2283.3, 0.01)

    # --- Tenure ---
    st.sidebar.markdown("### ⏳ Tenure Information")
    st.sidebar.caption("Customer’s total months with the company.")
    tenure = st.sidebar.selectbox("Tenure Group (Months)", ["1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"])
    for group in ["1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"]:
        inputs[f"tenure_group_{group}"] = 1 if tenure == group else 0

    return inputs

# ------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------
def predict_churn(model, inputs, feature_names, metadata):
    df_input = pd.DataFrame([inputs])
    for f in feature_names:
        if f not in df_input.columns:
            df_input[f] = 0
    df_input = df_input[feature_names]
    
    # Get probability
    proba = model.predict_proba(df_input)[0]
    
    # Use optimized threshold if available, otherwise use default 0.5
    threshold = metadata.get('best_threshold', 0.5)
    prediction = 1 if proba[1] >= threshold else 0
    
    return prediction, proba

# ------------------------------------------------------------
# Dashboard Metrics
# ------------------------------------------------------------
def overview_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")
    col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    col4.metric("Avg Total Charges", f"${df['TotalCharges'].mean():.0f}")

# ------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------
def visualize_data(df):
    st.subheader("📈 Customer Insights")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names=df["Churn"].map({0: "No Churn", 1: "Churn"}), title="Churn Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x="Churn", y="MonthlyCharges", title="Monthly Charges vs Churn")
        fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=["No Churn", "Churn"]))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Main App
# ------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    df = load_data()
    model, meta = load_model()
    if df is None or model is None:
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Model Info")
    if meta:
        metrics = meta["performance_metrics"]
        threshold = meta.get('best_threshold', 0.5)
        st.sidebar.info(f"""
        **Model:** {meta['model_name']}  
        **Type:** {meta['model_type']}  
        **AUC-ROC:** {metrics['auc_roc']:.3f}  
        **Accuracy:** {metrics['accuracy']:.3f}  
        **Recall:** {metrics['recall']:.3f}  
        **Threshold:** {threshold:.3f}
        """)

    tab1, tab2, tab3 = st.tabs(["🏠 Overview", "🔍 Predict Churn", "📊 Insights"])

    with tab1:
        st.header("📊 Dataset Overview")
        overview_metrics(df)
        visualize_data(df)
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.header("🔮 Predict Customer Churn")
        inputs = customer_input_form()
        if inputs and st.sidebar.button("🚀 Predict Churn"):
            pred, proba = predict_churn(model, inputs, meta["features"], meta)
            churn_prob = proba[1] * 100
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box danger-prediction">
                    <h2>⚠️ HIGH CHURN RISK</h2>
                    <h1>{churn_prob:.1f}%</h1>
                    <p>This customer is likely to churn.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box success-prediction">
                    <h2>✅ LOW CHURN RISK</h2>
                    <h1>{100 - churn_prob:.1f}%</h1>
                    <p>This customer is likely to stay.</p>
                </div>""", unsafe_allow_html=True)

    with tab3:
        st.header("📈 Model Insights")
        if meta:
            metrics = meta["performance_metrics"]
            fig = go.Figure(go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color="#667eea"))
            fig.update_layout(title="Model Performance Metrics", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
