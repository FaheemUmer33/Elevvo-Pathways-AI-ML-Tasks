# 🏦 Loan Approval Prediction

This project focuses on building a machine learning model and a web-based application to predict loan approval status based on various financial and personal attributes of the applicants. The project uses a Random Forest Classifier and is deployed using Streamlit and Ngrok for live interaction.

---

## 📁 Dataset

- **Source:** [KaggleHub Dataset - Loan Approval Prediction](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **File:** `loan_approval_dataset.csv`
- **Columns:**
  - `loan_id` (Dropped)
  - `no_of_dependents`
  - `education` *(Graduate / Not Graduate)*
  - `self_employed` *(Yes / No)*
  - `income_annum`
  - `loan_amount`
  - `loan_term`
  - `cibil_score`
  - `residential_assets_value`
  - `commercial_assets_value`
  - `luxury_assets_value`
  - `bank_asset_value`
  - `loan_status` *(Target: Approved / Rejected)*

---

## 🔍 Exploratory Data Analysis

- Handled missing values and removed unnecessary ID column.
- Inspected distribution of categorical variables.
- Label encoded `education`, `self_employed`, and `loan_status` features.

---

## 🛠️ Preprocessing & Modeling

- **Scaling:** StandardScaler was applied to numerical features.
- **Split:** Data was split into train and test sets using `train_test_split`.
- **Model:** Trained a `RandomForestClassifier` with `class_weight='balanced'` to address class imbalance.
- **Evaluation:**
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix
  - Feature Importance Plot

---

## 📊 Streamlit Web App

An interactive web application was built using Streamlit for users to input loan application details and get real-time predictions.

### Features:
- User-friendly form for input
- Model evaluation details
- Prediction result (Approved / Rejected)
- Feature importance visualization

---

## 🌐 Deployment with Ngrok

Ngrok is used to expose the local Streamlit app to a public URL, enabling live interaction without deploying to a cloud server.

### Setup:
1. Install Ngrok and set auth token.
2. Launch Streamlit app in the background.
3. Tunnel port 8501 using Ngrok and share the generated public URL.

---

## 📂 Project Structure


📦 loan-approval-prediction
├── loan_approval_app.py # Streamlit app
├── loan_model.pkl # Trained model (optional for saving)
├── loan_approval_dataset.csv # Dataset from KaggleHub
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## ⚙️ Installation & Run Instructions

1. Clone the repo or open in Colab.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt


Download dataset using KaggleHub:

import kagglehub
path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")


Run the Streamlit app:
streamlit run loan_approval_app.py


(Optional) Expose app using Ngrok:

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
public_url = ngrok.connect(8501)
print(public_url)



📌 Key Learnings
Handling categorical data and class imbalance

Model deployment using Streamlit and Ngrok

End-to-end ML pipeline from preprocessing to live prediction

Building interactive ML-powered web apps