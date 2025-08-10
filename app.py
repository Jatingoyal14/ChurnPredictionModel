import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔍 Customer Churn Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c62828;
    }
    .prediction-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


class ChurnPredictionApp:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self.models = self.load_models()

    @st.cache_resource
    def load_models(_self=None):
        """Load all trained models"""
        models = {}
        try:
            if os.path.exists('models/logistic_regression.pkl'):
                models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
            if os.path.exists('models/random_forest.pkl'):
                models['Random Forest'] = joblib.load('models/random_forest.pkl')
            if os.path.exists('models/deep_nn.h5'):
                models['Deep Neural Network'] = load_model('models/deep_nn.h5')
            if os.path.exists('models/preprocessor.pkl'):
                if _self is not None:
                    _self.preprocessor = joblib.load('models/preprocessor.pkl')
            if os.path.exists('models/feature_names.pkl'):
                if _self is not None:
                    _self.feature_names = joblib.load('models/feature_names.pkl')

            # Debug output (visible in UI and terminal)
            st.write("DEBUG: Models found:", list(models.keys()))
            st.write("DEBUG: Preprocessor loaded:", bool(_self and _self.preprocessor))
            st.write("DEBUG: Feature names loaded:", len(_self.feature_names) if _self else '-')
            return models
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return {}

    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        try:
            # Create engineered features
            df = data.copy()
            # Age groups
            df['Age_Group'] = pd.cut(df['Age'], 
                                     bins=[0, 30, 45, 60, 120], 
                                     labels=['Young', 'Middle', 'Senior', 'Elder'])
            # Credit score categories  
            df['Credit_Category'] = pd.cut(df['Credit Score'], 
                                           bins=[0, 500, 650, 750, 900], 
                                           labels=['Poor', 'Fair', 'Good', 'Excellent'])
            # Income categories
            df['Income_Category'] = pd.cut(df['Estimated Yearly Income'], 
                                           bins=[0, 100000, 300000, 500000, 999999999], 
                                           labels=['Low', 'Medium', 'High', 'Premium'])
            # Loyalty score
            df['Loyalty_Score'] = df['Customer Since'] / df['Age']
            df['Loyalty_Score'] = df['Loyalty_Score'].fillna(0)
            # High value customer
            df['High_Value_Customer'] = (df['Estimated Yearly Income'] > 400000).astype(int)
            # Has current account
            df['Has_Current_Account'] = (df['Current Account'] > 0).astype(int)
            # Remove target column if present
            if 'Closed' in df.columns:
                df = df.drop('Closed', axis=1)
            # Apply preprocessing
            if self.preprocessor is not None:
                processed_data = self.preprocessor.transform(df)
                return pd.DataFrame(processed_data, columns=self.feature_names)
            else:
                return df
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return None

    def make_prediction(self, processed_data, model_name):
        """Make prediction using specified model"""
        try:
            model = self.models[model_name]
            if model_name == 'Deep Neural Network':
                # Neural network prediction
                prediction_prob = model.predict(processed_data, verbose=0)[0][0]
                prediction = int(prediction_prob > 0.5)
            else:
                # Scikit-learn models
                prediction = model.predict(processed_data)[0]
                prediction_prob = model.predict_proba(processed_data)[0][1]
            return prediction, prediction_prob
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None

# Initialize the app
app = ChurnPredictionApp()

# Add sidebar model refresh button
if st.sidebar.button("🔄 Refresh Models"):
    app.models = app.load_models(app)
    st.sidebar.success("✅ Models refreshed!")

# Main title
st.markdown('<h1 class="main-header">🔍 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Single Prediction", "Batch Prediction", "Model Information"])


# =========== Single Prediction ===========
if page == "Single Prediction":
    st.markdown('<h2 class="sub-header">📊 Single Customer Prediction</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        geography = st.selectbox("Geography", ["Delhi", "Bengaluru", "Mumbai"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)

    with col2:
        customer_since = st.number_input("Customer Since (years)", min_value=0, max_value=20, value=3)
        current_account = st.number_input("Current Account Balance", min_value=0.0, value=100000.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
        upi_enabled = st.selectbox("UPI Enabled", [0, 1])

    with col3:
        yearly_income = st.number_input("Estimated Yearly Income", min_value=0.0, value=300000.0)
        # --- FIXED MODEL DROPDOWN ---
        if app.models and len(app.models) > 0:
            model_choice = st.selectbox("Choose Model", list(app.models.keys()))
        else:
            st.error("❌ No models loaded. Please train the model first, then click '🔄 Refresh Models'.")
            model_choice = None

    # Prediction button
    if st.button("🚀 Predict Churn", type="primary"):
        if app.models and (model_choice in app.models):
            input_data = pd.DataFrame({
                'Credit Score': [credit_score],
                'Geography': [geography],
                'Gender': [gender],
                'Age': [age],
                'Customer Since': [customer_since],
                'Current Account': [current_account],
                'Num of products': [num_products],
                'UPI Enabled': [upi_enabled],
                'Estimated Yearly Income': [yearly_income]
            })
            processed_data = app.preprocess_input(input_data)
            if processed_data is not None:
                prediction, probability = app.make_prediction(processed_data, model_choice)
                if prediction is not None:
                    col_result1, col_result2 = st.columns(2)
                    with col_result1:
                        if prediction == 1:
                            st.markdown(f'''
                            <div class="prediction-high">
                                <h3>⚠️ HIGH CHURN RISK</h3>
                                <p>This customer is likely to churn</p>
                                <p><strong>Churn Probability: {probability:.2%}</strong></p>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="prediction-low">
                                <h3>✅ LOW CHURN RISK</h3>
                                <p>This customer is likely to stay</p>
                                <p><strong>Retention Probability: {(1-probability):.2%}</strong></p>
                            </div>
                            ''', unsafe_allow_html=True)
                    with col_result2:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Risk %"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "red" if probability > 0.5 else "green"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('<h3 class="sub-header">💡 Business Recommendations</h3>', unsafe_allow_html=True)
                    if probability > 0.7:
                        st.warning("**Immediate Action Required:** Contact customer within 24 hours with retention offer")
                    elif probability > 0.5:
                        st.info("**Moderate Risk:** Schedule follow-up and consider targeted promotions")
                    else:
                        st.success("**Low Risk:** Continue regular service, customer likely satisfied")
        else:
            st.error("No models available. Please train the model first and then click '🔄 Refresh Models'.")

# =========== Batch Prediction ===========
elif page == "Batch Prediction":
    st.markdown('<h2 class="sub-header">📂 Batch Prediction</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("**Data Preview:**")
        st.dataframe(data.head())
        # --- FIXED MODEL DROPDOWN ---
        if app.models and len(app.models) > 0:
            model_choice = st.selectbox("Choose Model for Batch Prediction", list(app.models.keys()))
        else:
            st.error("❌ No models loaded. Please train the model first, then click '🔄 Refresh Models'.")
            model_choice = None

        if st.button("🚀 Run Batch Prediction", type="primary"):
            if app.models and model_choice in app.models:
                with st.spinner("Processing predictions..."):
                    processed_data = app.preprocess_input(data)
                    if processed_data is not None:
                        predictions, probabilities = [], []
                        for i in range(len(processed_data)):
                            row_data = processed_data.iloc[i:i+1]
                            pred, prob = app.make_prediction(row_data, model_choice)
                            predictions.append(pred)
                            probabilities.append(prob)
                        results_df = data.copy()
                        results_df['Churn_Prediction'] = predictions
                        results_df['Churn_Probability'] = probabilities
                        results_df['Risk_Level'] = pd.cut(results_df['Churn_Probability'],
                                                         bins=[0, 0.3, 0.7, 1.0],
                                                         labels=['Low', 'Medium', 'High'])
                        st.success(f"✅ Processed {len(results_df)} customers")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("High Risk Customers",
                                      len(results_df[results_df['Risk_Level'] == 'High']))
                        with col2:
                            st.metric("Medium Risk Customers",
                                      len(results_df[results_df['Risk_Level'] == 'Medium']))
                        with col3:
                            st.metric("Low Risk Customers",
                                      len(results_df[results_df['Risk_Level'] == 'Low']))
                        fig = px.histogram(results_df, x='Risk_Level',
                                           title="Customer Risk Distribution",
                                           color='Risk_Level',
                                           color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(results_df)
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download Results",
                                           data=csv,
                                           file_name="churn_predictions.csv",
                                           mime="text/csv")
            else:
                st.error("No models available. Please train the model first and then click '🔄 Refresh Models'.")

# =========== Model Information ===========
elif page == "Model Information":
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    if app.models and len(app.models) > 0:
        st.success(f"✅ {len(app.models)} models loaded successfully")
        for model_name in app.models.keys():
            with st.expander(f"📊 {model_name} Details"):
                if model_name == "Logistic Regression":
                    st.write("**Type:** Linear Classification Model")
                    st.write("**Advantages:** Fast, interpretable, good baseline")
                    st.write("**Best for:** Understanding feature importance")
                elif model_name == "Random Forest":
                    st.write("**Type:** Ensemble Tree-based Model")
                    st.write("**Advantages:** Handles non-linear relationships, feature importance")
                    st.write("**Best for:** Robust predictions with feature insights")
                elif model_name == "Deep Neural Network":
                    st.write("**Type:** Deep Learning Model")
                    st.write("**Architecture:** Multi-layer neural network with dropout")
                    st.write("**Best for:** Complex pattern recognition")
        st.markdown('<h3 class="sub-header">📋 Feature Engineering</h3>', unsafe_allow_html=True)
        engineered_features = [
            "Age_Group: Categorical grouping of customer age",
            "Credit_Category: Credit score classification",
            "Income_Category: Income level segmentation",
            "Loyalty_Score: Customer tenure relative to age",
            "High_Value_Customer: Binary flag for high-income customers",
            "Has_Current_Account: Binary flag for account holders"
        ]
        for feature in engineered_features:
            st.write(f"• {feature}")
    else:
        st.error("❌ No models loaded. Please train the model first and then click '🔄 Refresh Models'.")
        st.markdown("### 🔧 How to train models:")
        st.code("""
# Run this in your terminal:
python churn_prediction_model.py

# Or run the training pipeline:
from churn_prediction_model import ChurnPredictionSystem
system = ChurnPredictionSystem()
system.run_complete_pipeline('data/churn_prediction_data.csv')
        """)

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit and TensorFlow**")
st.markdown("*For questions or issues, please contact the development team.*")
