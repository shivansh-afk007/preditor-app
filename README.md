# Credit Risk Prediction App

A modern web application for predicting credit risk using machine learning. Users can input their financial and personal details to receive a risk assessment and alternative credit score, visualized with a beautiful dashboard UI.

## Features
- Predicts credit risk (High/Low) using a trained ML model
- Interactive dashboard with animated credit score gauge
- Clean, responsive UI built with Flask and Bootstrap
- Visualizes credit score, risk probabilities, and key factors
- Easy to use: just input your details and get instant results

## Tech Stack
- Python 3.10+
- Flask
- scikit-learn, pandas, numpy
- Bootstrap 5, HTML/CSS, JavaScript
- Joblib (for model serialization)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/shivansh-afk007/preditor-app.git
cd preditor-app
```

### 2. Create and activate a virtual environment
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (if not already trained)
If you don't have `models/credit_risk_model.joblib`, run:
```bash
python credit_risk_model.py
```

### 5. Run the web application
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Usage
1. Fill in your details in the form (age, income, loan amount, etc.).
2. Click **Predict Risk**.
3. View your alternative credit score, risk assessment, and dashboard.

## Project Structure
```
credit_ml/
├── app.py                  # Flask web app
├── credit_risk_model.py    # Model training script
├── predict.py              # CLI prediction script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── setup.bat               # Windows setup script
├── templates/
│   └── index.html          # Main web UI
└── models/
    └── credit_risk_model.joblib  # Trained model
```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
MIT 