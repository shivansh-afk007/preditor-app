# AltFiScore AI - Credit Risk Prediction System

AltFiScore AI is a modern credit risk assessment platform that leverages machine learning to provide accurate credit risk predictions using alternative data sources. The system offers separate dashboards for consumers and lenders, with a focus on user experience and data-driven insights.

## Features

- **Consumer Dashboard**
  - Credit score visualization with interactive gauge
  - Personalized risk assessment
  - Spending breakdown analysis
  - Credit utilization tracking
  - Recent activity monitoring
  - Customized recommendations

- **Lender Dashboard**
  - Client overview and risk distribution
  - Application status tracking
  - Loan performance analytics
  - Geographic distribution insights
  - Risk assessment tools

- **Machine Learning Model**
  - Random Forest-based credit risk prediction
  - Feature preprocessing pipeline
  - Model evaluation metrics
  - Probability-based risk assessment

## Project Structure

```
credit_ml/
├── app.py                 # Main Flask application
├── credit_risk_model.py   # ML model implementation
├── download_data.py       # Sample data generation
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── setup.bat             # Windows setup script
├── data/                 # Data directory
├── models/               # Trained model storage
├── static/              # Static assets (CSS, images)
└── templates/           # HTML templates
    ├── base.html
    ├── landing.html
    ├── dashboard_consumer.html
    └── dashboard_lender.html
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd credit_ml
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate Sample Data and Train Model**
   ```bash
   python train_model.py
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Application**
   - Open your browser and navigate to `http://localhost:5000`
   - Register as either a consumer or lender
   - Log in to access your dashboard

## Development Workflow

1. **Data Generation**
   - Use `download_data.py` to generate synthetic credit risk data
   - Modify parameters in `generate_sample_data()` for custom datasets

2. **Model Training**
   - Run `train_model.py` to train the credit risk model
   - Model and preprocessor are saved in `models/credit_risk_model.joblib`

3. **Making Predictions**
   - Use the consumer dashboard to input credit data
   - View risk predictions and personalized recommendations
   - Track credit score changes over time

## API Endpoints

- `GET /` - Landing page
- `GET /login` - Login page
- `GET /register` - Registration page
- `GET /dashboard/consumer` - Consumer dashboard
- `GET /dashboard/lender` - Lender dashboard
- `POST /predict` - Credit risk prediction endpoint

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask framework for web application
- scikit-learn for machine learning implementation
- Bootstrap for UI components
- Chart.js for data visualization 