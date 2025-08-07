# 🎯 Task 1: Student Score Prediction App

This project is part of my internship at **Elevvo Pathways**, focusing on implementing a regression-based model to predict student exam scores. The application uses **Streamlit** for an interactive web interface and is powered by machine learning via **scikit-learn**.

---

## 📌 Problem Statement

Build a regression model that predicts a student's **math score** based on:
- Demographic attributes
- Test preparation course status
- Reading and writing scores

---

## 🧠 Solution Overview

We used a **linear regression model** trained on the [Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) to predict math scores. An interactive **Streamlit app** allows users to input features and view the predicted outcome in real-time.

---

## 📂 Dataset

- Source: Kaggle ([Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams))
- File: `StudentsPerformance.csv`
- Features used:
  - Gender
  - Race/Ethnicity
  - Parental Level of Education
  - Lunch
  - Test Preparation Course
  - Reading Score
  - Writing Score

---

## ⚙️ Technologies Used

| Tool/Library     | Purpose                         |
|------------------|----------------------------------|
| Python           | Programming Language             |
| Pandas           | Data Handling                    |
| NumPy            | Numerical Computation            |
| Matplotlib       | Data Visualization               |
| Scikit-learn     | Machine Learning (Linear Regression) |
| Streamlit        | Web App Development              |
| pyngrok          | Public URL for Colab-hosted App  |
| KaggleHub        | Download dataset directly in Colab |

---

## 🚀 App Features

### ✅ Data Exploration
- Loads and displays the original dataset
- Simulates study hours and visualizes correlation with average score

### ✅ Regression Modeling
- Trains a Linear Regression model on preprocessed data
- Visualizes actual vs predicted results
- Calculates model performance metrics (MSE, R² Score)

### ✅ Web App (Streamlit)
- User inputs values via dropdowns and sliders
- Encodes user input to match training data format
- Predicts and displays expected math score
- Hosted via `ngrok` to work within Google Colab

---

## 🖥️ How to Run (in Google Colab)

1. Install dependencies:
   ```python
   pip install kagglehub streamlit pyngrok scikit-learn pandas matplotlib



Download dataset:

import kagglehub
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
Save your Streamlit app to a .py file (e.g., student_score_app.py).



Launch Streamlit in the background:

!streamlit run student_score_app.py &>/content/log.txt &



Open app using ngrok:

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🌐 App running at:", public_url)
📊 Model Performance
Metric	Value
MSE	~165.27
R² Score	~0.74



📎 Screenshots

![User Form](/image.png)
