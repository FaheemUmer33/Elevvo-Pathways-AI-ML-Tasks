# 🧠 Task 2: Customer Segmentation using K-Means Clustering

This project aims to segment customers based on key demographic and behavioral attributes using **K-Means clustering**, helping businesses understand customer groups and target them effectively.

## 📁 Dataset

- Source: [Kaggle Dataset - Mall Customer Segmentation](https://www.kaggle.com/datasets/vetrirah/customer)
- Format: CSV (`Train.csv`)
- Attributes include:
  - `Gender`, `Age`, `Spending_Score`, `Family_Size`, `Ever_Married`, `Work_Experience`, etc.

## 🛠️ Tech Stack

- **Python 3**
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn` for preprocessing and clustering
  - `streamlit` for the interactive app
  - `pyngrok` for deploying Streamlit in Google Colab

---

## 📊 Steps Performed

### 1. **Data Loading & Exploration**
- Dataset loaded using `kagglehub` directly in Colab.
- Null values checked and handled using forward fill and row drops where necessary.
- Categorical variables (`Gender`, `Ever_Married`, etc.) encoded using `LabelEncoder`.

### 2. **Feature Selection & Scaling**
- Selected features for clustering:
  - `Age`, `Spending_Score`, `Family_Size`, `Work_Experience`
- Used `StandardScaler` to normalize numerical data for better clustering performance.

### 3. **Clustering with K-Means**
- Used the **Elbow Method** to determine the optimal number of clusters.
- Applied KMeans clustering with the optimal `k` (default = 4).
- Assigned cluster labels to each customer.

### 4. **Visualization**
- 2D PCA projection of the clusters to visualize separation.
- Scatter plot using `Age` vs `Spending Score` colored by cluster ID.

---

## 🌐 Streamlit Web App

An interactive web app is built using **Streamlit**, allowing users to:

- View raw data.
- Visualize Elbow curve to choose cluster count.
- Apply K-Means clustering interactively.
- Visualize customer segments (2D PCA plot).
- Download the clustered dataset.

### 🚀 Hosted on ngrok (for Colab)
```bash
!streamlit run customer_segmentation_app.py & npx localtunnel --port 8501



🧪 How to Run (In Google Colab)
Install dependencies:

!pip install kagglehub streamlit pyngrok pandas matplotlib seaborn scikit-learn
Run all notebook cells in order.



Get public URL to the Streamlit app using ngrok:

from pyngrok import ngrok
ngrok.connect(8501)


📂 File Structure
.
├── customer_segmentation_app.py     # Streamlit app script
├── Train.csv                        # Customer dataset
├── Task2_CustomerSegmentation.ipynb # Complete Colab notebook
└── README.md                        # Project overview



📌 Key Learnings
KMeans is effective for unsupervised grouping based on similarities.

Feature scaling is crucial for distance-based models.

Visual tools (Elbow method, PCA) enhance understanding of clustering quality.

Streamlit allows simple and powerful web app creation with real-time interactivity.