#  Prediction System

A complete machine learning system for predicting customer churn with an interactive Streamlit web interface.

## 🚀 Quick Start Guide

### Step 1: Download Project Files
1. Save all the provided files to a folder called `ChurnPredictionApp`
2. Your folder structure should look like this:
```
ChurnPredictionApp/
├── app.py                          # Streamlit UI application
├── churn_prediction_model.py       # ML training pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── setup_and_run.bat              # Windows setup script
├── setup_and_run.sh               # Mac/Linux setup script
└── data/
    └── Churn-Prediction-Data.csv   # Your dataset
```

### Step 2: Install Requirements

**Option A: Using VS Code (Recommended)**
1. Open VS Code
2. Open the `ChurnPredictionApp` folder (`File > Open Folder...`)
3. Open Terminal in VS Code (`View > Terminal`)
4. Run the following commands:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

**Option B: Using Command Line**
1. Navigate to the project folder
2. Run the same commands as above

### Step 3: Prepare Your Data
1. Create a `data` folder in your project directory
2. Place your `Churn-Prediction-Data.csv` file inside the `data` folder

### Step 4: Train the Model
Run the training pipeline first:
```bash
python churn_prediction_model.py
```

This will:
- Load and preprocess your data
- Train multiple ML models (Neural Network, Random Forest, Logistic Regression)
- Save trained models in the `models/` folder
- Display performance metrics

### Step 5: Launch the Web Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 Features

### 📊 Interactive Dashboard
- **Single Prediction**: Enter customer details and get instant churn prediction
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Model Comparison**: Compare different ML models

### 🤖 Multiple ML Models
- **Deep Neural Network**: Advanced pattern recognition
- **Random Forest**: Ensemble method with feature importance
- **Logistic Regression**: Interpretable baseline model

### 📈 Advanced Features
- **Feature Engineering**: Automatic creation of derived features
- **Class Imbalance Handling**: SMOTE oversampling
- **Interactive Visualizations**: Plotly charts and gauges
- **Business Recommendations**: Actionable insights based on predictions

## 📋 Data Format

Your CSV file should contain these columns:
- `Credit Score`: Customer credit score (300-850)
- `Geography`: Location (Delhi, Bengaluru, Mumbai)
- `Gender`: Male/Female
- `Age`: Customer age
- `Customer Since`: Years as customer
- `Current Account`: Account balance
- `Num of products`: Number of products used
- `UPI Enabled`: UPI usage (0/1)
- `Estimated Yearly Income`: Annual income
- `Closed`: Target variable (0=retained, 1=churned)

## 🛠 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **Model not found error**
   - Run `python churn_prediction_model.py` first to train models

3. **Port already in use**
   - Use `streamlit run app.py --server.port 8502`

4. **TensorFlow warnings**
   - These are normal and can be ignored

### Performance Tips:
- Use smaller batch sizes if running into memory issues
- Close other applications to free up RAM
- Use CPU-only TensorFlow if GPU causes issues

## 🎤 Interview Presentation Tips

1. **Start with Business Problem**: Explain what churn is and why it matters
2. **Show Data Exploration**: Demonstrate data understanding
3. **Explain Feature Engineering**: Highlight your domain knowledge
4. **Model Comparison**: Show why you chose specific models
5. **Live Demo**: Use the Streamlit app for interactive demonstration
6. **Business Impact**: Discuss ROI and actionable insights

## 📞 Support

If you encounter any issues:
1. Check that all files are in the correct locations
2. Ensure Python 3.8+ is installed
3. Verify virtual environment is activated
4. Check error messages in terminal

## 🏆 Project Highlights

- **Production-Ready Code**: Clean, modular, well-documented
- **Multiple ML Models**: Compare different approaches
- **Interactive UI**: Professional Streamlit application
- **Feature Engineering**: Business domain knowledge applied
- **Class Imbalance Handling**: SMOTE for better predictions
- **Model Evaluation**: Comprehensive metrics and visualizations
