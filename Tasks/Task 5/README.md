# 🎬 Task 5: Movie Recommendation System

This project is part of the AI/ML Internship at Elevvo Pathways. It implements a **user-based collaborative filtering** recommender system using the **MovieLens 100K dataset**. A Streamlit web app is also included to interactively recommend movies to users.

---

## 📌 Objective

Build a recommender system that suggests top-rated unseen movies to a user based on similar users' preferences using a **User-Item matrix** and **Cosine Similarity**.

---

## 📁 Dataset

- **Source**: [MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- **Files Used**:
  - `u.data`: Contains user ratings.
  - `u.item`: Contains movie information.

---

## 🚀 Features

- Preprocessing and loading of MovieLens dataset.
- Construction of the User-Item rating matrix.
- Similarity-based movie recommendation using **Cosine Similarity**.
- Evaluation using **Precision@K**.
- Real-time recommendation app using **Streamlit** and **Ngrok** for public access.

---

## 📊 Model Workflow

### 1. Load and Preprocess Dataset
- Data is loaded from `KaggleHub` using the `kagglehub` Python package.
- Ratings and movie titles are parsed into Pandas DataFrames.

### 2. Build User-Item Matrix
- Construct a matrix where rows represent users and columns represent movies.
- Missing ratings are filled with zero.

### 3. Train-Test Split
- Each user's ratings are split to simulate train and test behavior.
- 80% for training, 20% for evaluation.

### 4. Recommendation Logic
- Compute **Cosine Similarity** between users.
- Filter out weak similarities.
- Recommend top K movies the user hasn’t rated before.

### 5. Evaluation
- **Precision@5** is computed using 100 random test users to evaluate recommendation relevance.

---

## 📈 Sample Output

```bash
📈 Average Precision@5: 0.12
🔍 Recommendations for User 59:
✔️ Star Wars (1977)
✔️ Fargo (1996)
✔️ Return of the Jedi (1983)
✔️ Empire Strikes Back, The (1980)
✔️ Silence of the Lambs, The (1991)




🌐 Streamlit Web App
A user-friendly UI built with Streamlit for selecting a user and getting recommended movies interactively.


✅ Features
Select any user ID.

Choose number of recommendations.

Get top K movies the user is likely to enjoy.

🛠 Setup Instructions
1. Install Dependencies

pip install streamlit kagglehub pyngrok


2. Authenticate Ngrok
ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>


3. Run the Streamlit App
streamlit run movie_recommender_app.py


4. Start Ngrok Tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)


📂 Project Structure
movie-recommender/
├── movie_recommender_app.py     # Streamlit application
├── Task5_Movie_Recommendation.ipynb
├── README.md                    # Project documentation
└── data/                        # Downloaded via KaggleHub



📌 Technologies Used
Python, Pandas, NumPy

Scikit-learn (Cosine Similarity, Train-Test Split)

Streamlit (Web App)

PyNgrok (Tunneling)

KaggleHub (Dataset Access)

🧠 Learnings
Hands-on experience with collaborative filtering.

Matrix operations and cosine similarity for recommendation.

Evaluation metrics like Precision@K.

Real-time deployment with Streamlit and Ngrok.