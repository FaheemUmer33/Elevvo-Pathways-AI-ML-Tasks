# 🌲 Forest Cover Type Classification

This project is part of my AI/ML internship, where we classify different forest cover types using the **UCI Covertype Dataset**. We train an XGBoost classifier and build an interactive **Streamlit web app** with real-time evaluation and feature analysis.

---

## 📌 Objective

To classify the type of forest cover (7 types) based on cartographic variables using machine learning, specifically **XGBoost**, and deploy the solution using **Streamlit** with **ngrok** integration for remote access.

---

## 📁 Dataset Information

- **Source**: [UCI Machine Learning Repository - Covertype Data Set](https://archive.ics.uci.edu/ml/datasets/covertype)
- **Features**:
  - Elevation, Slope, Aspect, Horizontal/Vertical Distance to Hydrology
  - Wilderness Areas (4 binary features)
  - Soil Types (40 binary features)
- **Target**: `Cover_Type` (7 forest types labeled 1–7)

---

## ✅ Tasks Performed

### 1. 📥 Data Loading
- Used `ucimlrepo` to fetch the dataset programmatically.
- Combined features and target into a single DataFrame.

### 2. 🧹 Preprocessing
- Separated features (`X`) and target (`y`)
- Normalized the target labels from 1–7 to 0–6 for XGBoost compatibility.
- Applied train-test split.

### 3. 🧠 Model Training
- Trained an `XGBClassifier` with `multi:softmax` objective for multi-class classification.

### 4. 📈 Model Evaluation
- Calculated accuracy and classification report.
- Plotted the confusion matrix to visualize model performance.
- Analyzed top 15 most important features.

### 5. 🌐 Streamlit App
- Built an interactive web app using **Streamlit**.
- App sections:
  - Dataset preview
  - Classification report
  - Confusion matrix
  - Feature importance bar chart

### 6. 🚀 Deployment with ngrok
- Used `pyngrok` to tunnel the app and make it accessible via a public URL.

---

## 🧪 Libraries Used

- `pandas`, `numpy` for data manipulation  
- `xgboost` for model training  
- `scikit-learn` for preprocessing and evaluation  
- `matplotlib`, `seaborn` for visualization  
- `streamlit` for the web app  
- `pyngrok` for deployment  
- `ucimlrepo` for direct UCI dataset fetching  

---

## 🚀 How to Run the App (on Google Colab or Jupyter Notebook)

1. Install dependencies:
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn streamlit pyngrok ucimlrepo



Write the Streamlit app code:

%%writefile forest_cover_classification_app.py
# [Paste the full Streamlit app code here]


Authenticate ngrok:

from pyngrok import conf
conf.get_default().auth_token = "YOUR_NGROK_AUTH_TOKEN"


Launch the Streamlit app:

!streamlit run forest_cover_classification_app.py &> /dev/null &

import time
time.sleep(5)

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🌐 Your Streamlit app is live at:", public_url)
📸 Sample Output
Accuracy: ~94%

Visuals:

Confusion matrix heatmap

Top features like Elevation, Horizontal Distance to Fire Points, etc.

📌 Conclusion
This task demonstrates the application of gradient boosting on high-dimensional geospatial data and deploying it through a modern ML interface. The model provides solid accuracy and interpretability using XGBoost’s feature importance capabilities.

