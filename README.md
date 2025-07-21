# 🌍 Investigating prominent factor affecting E-commerce development

This project aims to identify and model the most prominent factors affecting e-commerce adoption in developing nations, particularly in rural areas. It uses advanced feature engineering, interpretable machine learning models, and explainability tools to build a robust, scalable, and reproducible pipeline.

---

## 📌 Objective

To predict e-commerce readiness scores for rural communities based on socioeconomic, infrastructural, and behavioral indicators — and to explain those predictions in a way that is useful to policy makers, NGOs, and technology startups.

---

## 🧠 Key Features

- **Data Source:** Simulated dataset resembling rural socioeconomic conditions
- **Models Used:** OLS, Decision Tree (CART), Partial Least Squares Regression (PLSR)
- **Feature Engineering:**
  - `Policy Coherence Score`: Measures digital governance alignment
  - `Trust Index`: Composite indicator via PCA
- **Explainability Tools:** SHAP, LIME, Graphviz
- **Diagnostics:** Residuals, Prediction Error plots, cross-validation
- **Deployment Ready:** Designed for integration into a dashboard (e.g. Streamlit or Dash)

---

## 📁 Project Structure

```
rural-ecommerce-readiness/
├── data/                        # Raw dataset (.csv)
├── notebooks/                   # 10 modular Jupyter notebooks
├── outputs/                     # Saved models and metrics
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
```

---

## 🔟 Pipeline Stages

1. **Setup Environment** - Install dependencies, import libraries
2. **Load Dataset** - Load and inspect raw data
3. **EDA** - Visualize feature distributions and correlations
4. **Feature Engineering** - Create trust/policy indicators and handle missing data
5. **Preprocessing** - Scale features, encode categorical data
6. **Split Dataset** - Stratified train-validation-test split
7. **Model Training** - OLS, CART, PLSR + hyperparameter tuning
8. **Model Evaluation** - RMSE, MAE, MAPE, R², residual analysis
9. **Model Explainability** - SHAP, LIME, and tree diagrams
10. **Export Results** - Save models and preprocessing pipeline

---

## 📊 Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Variance Explained
- **Adjusted R²**: For OLS

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- `matplotlib`, `seaborn`, `yellowbrick`
- `shap`, `lime`, `joblib`, `graphviz`

---

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the notebooks in order from `notebooks/` directory

---

## 📈 Future Work

- Integrate with a Streamlit dashboard for live predictions
- Incorporate time-series trends (growth %)
- Apply model to real-world data from development organizations

---

## 🤝 Contributors

- **You** — as the researcher/developer
- Supervised by academic/research mentors

---

## 📜 License

This project is part of a university dissertation and may be reused with proper attribution for academic purposes.

---
