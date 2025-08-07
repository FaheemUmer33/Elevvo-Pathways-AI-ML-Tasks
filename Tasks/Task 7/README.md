# 🛒 Task 7: Walmart Sales Forecasting

This project focuses on forecasting future sales using historical data from Walmart stores. We perform time-based feature engineering and apply regression models to predict future sales values. The app also visualizes actual vs. predicted sales trends.

---

## 📂 Dataset

- **Source**: [Walmart Sales Forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)
- **Loaded Using**: `kagglehub`

The dataset contains daily sales data for various departments and stores, along with additional features such as holidays and promotions.

---

## 🧠 Objective

- Predict future sales based on historical trends.
- Use time-series related features such as:
  - Date-based features (day, month, year, week)
  - Lag features (previous day/week sales)
- Train regression models to forecast upcoming sales.

---

## 📦 Tools & Libraries

- `Pandas`, `NumPy` – Data manipulation
- `Scikit-learn` – Regression models
- `Matplotlib`, `Seaborn` – Visualization
- `Streamlit` – Web app for interaction
- `Pyngrok` – Expose local app to public via secure tunnel

---

## ⚙️ Implementation Steps

1. **Data Loading**: Load the Walmart dataset using `kagglehub`.
2. **Preprocessing**:
   - Handle missing values (if any)
   - Convert `Date` column to datetime
   - Extract `day`, `month`, `year`, `weekofyear` from the date
   - Add lag features (previous day/week sales)
3. **Modeling**:
   - Train a `RandomForestRegressor` (or any regression model) on training data
   - Evaluate on test set using MAE, RMSE
4. **Visualization**:
   - Plot actual vs. predicted sales for a store/department
5. **Streamlit App**:
   - Allow user to choose store and department
   - Show predicted vs. actual sales
   - Visualize trend over time

---

## 🚀 Streamlit App

You can launch the app using:

```python
!streamlit run app.py &
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
public_url = ngrok.connect(8501, proto="http")
print(public_url)



📊 Sample Visuals
📈 Actual vs Predicted sales line chart

📅 Date-based trends (month-wise sales)

🛍️ Department and Store-wise predictions

📁 File Structure
├── app.py                   # Streamlit app
├── sales_forecasting.ipynb # Full notebook
├── walmart_sales.csv        # Dataset (loaded via kagglehub)
├── models/                  # (optional) saved regression models
├── README.md
✅ Results
Achieved competitive prediction accuracy using tree-based regressors.

Visual insights help identify trends and anomalies.