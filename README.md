# Safaricom Churn Intelligence
ML-powered customer churn prediction with Kenya-specific market features — deployed as a live REST API.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.125+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live API](https://img.shields.io/badge/API-Live-brightgreen)](https://safaricom-churn-intelligence.onrender.com)

---

## 🚀 Live Deployment

| | URL |
|--|--|
| **Landing Page** | https://safaricom-churn-intelligence.onrender.com |
| **Interactive Predictor** | https://safaricom-churn-intelligence.onrender.com/app |
| **API Docs (Swagger)** | https://safaricom-churn-intelligence.onrender.com/docs |
| **Health Check** | https://safaricom-churn-intelligence.onrender.com/health |

---

## Project Overview

This project tackles customer churn prediction for Kenya's telco market by enhancing a standard churn dataset with **Kenya-specific features** including M-Pesa integration, Bonga loyalty points, Safaricom Home adoption, and county-level demographics.

The model is served via a **production-grade FastAPI** with a custom interactive frontend — no Jupyter notebook required to use it.

**Key Achievement:** ROC-AUC of **0.8420** (strong model performance in distinguishing churners from non-churners)

---

## Business Impact

- **Customers Analyzed:** 7,043
- **Model Accuracy:** 84.2% (ROC-AUC)
- **At-Risk Customers Identified:** ~3,200 (45% of base)
- **Projected ROI:** 567% from targeted retention campaigns
- **Estimated Annual Savings:** KES 800M+ (if scaled to Safaricom's 40M+ customers)

---

## 🇰🇪 Kenya-Specific Features

Unlike generic churn models, this project incorporates:

| Feature                 | Description                         | Impact                       |

| **M-Pesa Usage**        | Mobile money transaction patterns   | Higher usage = 32% lower churn |
| **Bonga Points**        | Loyalty program engagement          | Active users = 28% lower churn |
| **Safaricom Home**      | Home fiber adoption                 | Subscribers = 41% lower churn |
| **County Demographics** | Geographic & urban/rural split     | Rural + poor network = 2.3x higher risk |
| **Network Quality**     | Perceived service quality by region | Low quality = primary churn driver |
| **Competitor Exposure** | Airtel/Telkom market pressure       | High exposure = 1.8x higher churn |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/app` | Interactive prediction UI |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions via CSV upload |
| `GET` | `/feature-importance` | Top churn drivers from the model |
| `GET` | `/health` | Service health check |

### Quick Example

```bash
curl -X POST "https://safaricom-churn-intelligence.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 8,
    "monthly_charges": 3200,
    "total_charges": 25600,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "mpesa_usage_score": 3.5,
    "bonga_points_active": false,
    "network_quality_score": 4.0,
    "competitor_exposure": 7.0,
    "county": "Nakuru",
    "rural": true,
    "internet_service": "Fiber optic",
    "phone_service": true,
    "paperless_billing": true,
    "multiple_lines": "No",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "senior_citizen": false,
    "partner": false,
    "dependents": false,
    "safaricom_home": false
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7823,
  "risk_level": "High",
  "risk_score": 3,
  "top_churn_drivers": [
    "Month-to-month contract (no long-term commitment)",
    "Short tenure (8 months — high early churn risk)",
    "Low M-Pesa usage score (3.5/10)",
    "Poor network quality perception (4.0/10)"
  ],
  "recommended_actions": [
    " Personal outreach within 24 hours",
    "Offer 2x Bonga points for 3 months",
    "Propose annual contract with 2 months free",
    "Escalate to retention specialist team"
  ]
}
```

---

## 🏗️ Project Structure

```
safaricom-churn-intelligence/
├── data/
│   ├── raw/                          # Original telco dataset
│   └── processed/                    # Enhanced with Kenyan features
├── notebooks/
│   ├── 01_data_preparation.ipynb     # Load & clean data
│   ├── 02_exploratory_analysis.ipynb # EDA with visualizations
│   ├── 03_feature_engineering.ipynb  # Add Kenyan features
│   ├── 04_model_training.ipynb       # Train ML model (ROC-AUC: 0.8420)
│   └── 05_business_insights.ipynb    # ROI & retention strategies
├── src/
│   ├── data_processing.py            # Data loading & cleaning
│   ├── feature_engineering.py        # Kenya-specific features
│   └── model.py                      # ML model training & prediction
├── models/
│   └── churn_model.pkl               # Trained Random Forest model
├── config/
│   └── config.yaml                   # Project configuration
├── main.py                           # FastAPI application
├── landing_page.html                 # Landing page
├── app.html                          # Interactive prediction UI
├── render.yaml                       # Render.com deployment config
├── Dockerfile                        # Docker deployment config
└── requirements.txt
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/bennedictbett/Safaricom-churn-intelligence.git
cd safaricom-churn-intelligence

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/app for the interactive UI or http://127.0.0.1:8000/docs for Swagger.

### Run Notebooks (model training)

```bash
jupyter notebook
# Open notebooks in order: 01 → 02 → 03 → 04 → 05
```

---

## 📈 Key Results

### Model Performance
- **ROC-AUC:** 0.8420
- **Accuracy:** 82%
- **Precision:** 70% (churned class)
- **Recall:** 54% (churned class)

### Top Churn Drivers
1. Contract type (month-to-month)
2. Customer tenure (< 12 months)
3. Monthly charges (high prices)
4. M-Pesa engagement (low usage)
5. Network quality (poor perception)

### Customer Segmentation
- **High Risk (15%):** 70%+ churn probability → Immediate intervention
- **Medium Risk (30%):** 40-70% → Proactive engagement
- **Low Risk (55%):** < 40% → Maintain satisfaction

---

## 💡 Business Recommendations

### 1. Prioritize High-Risk Customers
- Focus on top 15% at-risk customers
- Deploy retention campaigns within 48 hours of risk detection
- **Strategy:** Personal calls + 2x Bonga points + contract discounts

### 2. Boost Digital Ecosystem Engagement
- Increase M-Pesa adoption through incentives
- Gamify Bonga loyalty program
- Bundle Safaricom Home with mobile plans

### 3. Improve Rural Network Quality
- Target infrastructure investment in high-churn counties
- Network quality is the #1 complaint among churners

### 4. Contract Migration Strategy
- Move month-to-month customers to annual contracts
- Offer compelling incentives (3 months free, device upgrades)

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **API Framework:** FastAPI + Uvicorn
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, imbalanced-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Model:** Random Forest Classifier (200 estimators, max_depth=15)
- **Deployment:** Render.com (free tier)
- **Frontend:** Vanilla HTML/CSS/JS (embedded in FastAPI)

---

## 🔮 Enhancements

- [x] Real-time churn scoring API ✅
- [x] Interactive web UI for predictions ✅
- [x] Deployed to production (Render.com) ✅
- [ ] A/B testing framework for retention campaigns
- [ ] Integration with Safaricom CRM systems
- [ ] Expand to predict customer lifetime value (CLV)
- [ ] Add time-series forecasting for churn trends

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Benedict Bett**
- GitHub: [@bennedictbett](https://github.com/bennedictbett)
- LinkedIn: [Benedict Bett](https://www.linkedin.com/in/benedict-bett-a9899038a/)
- Email: benedictbett08@gmail.com

---

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Kenyan telco market insights based on public Safaricom reports and industry research
- Built as a portfolio project demonstrating ML application in East African markets

---

**⭐ If you found this project useful, please star this repository!**