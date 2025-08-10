@echo off
echo 🚀 Setting up Customer Churn Prediction System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created!
echo.

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated!
echo.

REM Install requirements
echo 📥 Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install requirements
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!
echo.

REM Create data directory
if not exist "data" (
    echo 📁 Creating data directory...
    mkdir data
    echo ✅ Data directory created!
    echo ⚠️  Please place your Churn-Prediction-Data.csv file in the data folder
    echo.
)

REM Train the model
echo 🤖 Training the machine learning model...
python churn_prediction_model.py
if errorlevel 1 (
    echo ❌ Model training failed. Please check that your CSV file is in the data folder.
    pause
    exit /b 1
)

echo ✅ Model trained successfully!
echo.

REM Launch Streamlit app
echo 🌟 Launching the web application...
echo 🔗 The app will open in your browser at: http://localhost:8501
echo.
echo ⏹️  Press Ctrl+C to stop the application
echo.
streamlit run app.py

pause