# Retail Analytics with Google Cloud Platform (GCP)

## Project Overview

This project demonstrates the development of a comprehensive retail analytics platform leveraging Google Cloud Platform technologies. It incorporates data management, machine learning, and real-time analytics to provide insights into customer behavior, sales trends, and operational improvements.

### Live Application
Access the application: [Retail Analytics Platform](https://group21cloud.uc.r.appspot.com/)

---

## Features

- **User Authentication:** Secure login system.
- **Interactive Dashboard:** Visual insights into retail data.
- **Real-time Data Search:** Search functionality for household-level data.
- **Machine Learning Analytics:** Predictive models for Customer Lifetime Value (CLV) and churn analysis.
- **Data Upload Capability:** Upload and manage transaction data.

---

## Development Details

- **Database:** MySQL hosted on Google Cloud SQL
- **Framework:** Python Flask
- **Deployment:** Google App Engine
- **Region:** `us-central1`
- **Python Version:** 3.9

---

## Data Sources

- **Households:** 400 entries
- **Transactions:** 85,337 records (2018–2020)
- **Products:** Detailed product metadata
- **Data Files:** `households.csv`, `products.csv`, `transactions.csv`

---

## Machine Learning Models

1. **Linear Regression:** Predict Customer Lifetime Value (CLV) based on purchase behavior.
2. **Random Forest:** Segment customers and predict churn.
3. **Gradient Boosting:** Advanced prediction of CLV with high accuracy.

---

## Key Analytics

- **Demographics & Engagement:**
  - Middle-income households drive the majority of revenue.
  - Larger households spend significantly more on average.
- **Sales Trends:**
  - Stable weekly sales with seasonal peaks during holidays.
- **Basket Analysis:**
  - High-frequency product combinations (e.g., milk-bread, coffee-cream).
  - Recommendations for cross-selling opportunities.
- **Churn Prediction:**
  - High-risk customer identification with personalized retention strategies.

---

## Application Structure

```plaintext
project/
│
├── data/                     # Raw data files
│   ├── households.csv
│   ├── products.csv
│   └── transactions.csv
│
├── static/                   # CSS and static assets
│   └── style.css
│
├── templates/                # HTML templates
│   ├── login.html
│   ├── search.html
│   ├── dashboard.html
│   └── analytics.html
│
├── main.py                   # Application entry point
├── db_utils.py               # Database utilities
├── ml_utils.py               # ML functionalities
├── requirements.txt          # Dependencies
└── app.yaml                  # GCP deployment configuration
```

---

## Deployment

1. **Cloud SQL Setup:** Configured MySQL instance for data storage.
2. **Web App:** Flask-based backend with a responsive frontend.
3. **Deployment:** Hosted on Google App Engine for scalability and reliability.

