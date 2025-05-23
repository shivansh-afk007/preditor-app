@echo off
REM Credit Risk Prediction App Setup Script (Windows)

REM 1. Create virtual environment
python -m venv .venv

REM 2. Activate virtual environment
call .venv\Scripts\activate

REM 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM 4. Train the model (if not already trained)
if not exist models\credit_risk_model.joblib (
    echo Training model...
    python credit_risk_model.py
) else (
    echo Model already trained.
)

REM 5. Run the Flask app
echo Starting the Flask app...
python app.py
