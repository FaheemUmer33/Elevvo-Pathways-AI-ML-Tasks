# 🎵 Task 6 – Music Genre Classification

## 📌 Overview
This project classifies songs into their respective genres using the **GTZAN Music Genre Dataset** from Kaggle.  
Two approaches are implemented:
1. **Tabular Classification** – Uses pre-extracted audio features (`features_30_sec.csv`) with a RandomForest model.
2. **Image Classification** – Uses Mel-spectrogram images (`images_original`) with a CNN (Convolutional Neural Network) built in TensorFlow/Keras.

The system is also deployed as a **Streamlit web app** for interactive predictions.

---

## 📂 Dataset
- **Source:** [GTZAN Dataset – Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Structure:**
Data/
├── features_30_sec.csv # Tabular audio features for 30s clips
├── features_3_sec.csv # Tabular audio features for 3s clips
├── genres_original/ # Original audio files (WAV)
├── images_original/ # Mel-spectrogram images




- **Genres:** `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

---

## 🛠️ Tools & Libraries
- **Python**
- **Librosa** – Audio processing & feature extraction
- **Pandas / NumPy** – Data handling
- **Matplotlib / Seaborn** – Visualization
- **Scikit-learn** – RandomForest Classifier & preprocessing
- **TensorFlow / Keras** – CNN model for image classification
- **Streamlit** – Web app deployment
- **Pyngrok** – Public URL for Streamlit in Colab

---

## 🚀 Implementation Steps

### **1. Data Loading & Exploration**
- Download dataset via **KaggleHub**
- Check available files in `Data/` directory
- Load `features_30_sec.csv` for tabular classification

### **2. Preprocessing (Tabular Approach)**
- Encode genre labels using `LabelEncoder`
- Drop non-feature columns (`filename`, `label`)
- Scale features using `StandardScaler`
- Split into train/test sets (80/20)

### **3. Train & Evaluate RandomForest Model**
- Train a `RandomForestClassifier` with 200 estimators
- Evaluate with:
- **Accuracy**
- **Classification Report**
- **Confusion Matrix** (heatmap)

### **4. Preprocessing (Image Approach)**
- Load Mel-spectrogram images using `image_dataset_from_directory`
- Resize to `(128, 128)` pixels
- Apply caching, shuffling, and prefetching for performance

### **5. CNN Model**
- Architecture:
- `Rescaling` layer (normalize pixels)
- `Conv2D` + `MaxPooling2D` layers
- Fully connected `Dense` layers
- Dropout for regularization
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Metric: `Accuracy`

### **6. Train & Evaluate CNN**
- Train for 10 epochs
- Plot **Accuracy** & **Loss** curves for train/validation sets

### **7. Save Models**
- Save RandomForest model, scaler, and label encoder (`.pkl`)
- Save CNN model (`.h5`)

### **8. Streamlit App**
- Allows user to:
- View dataset
- Enter custom feature values
- Predict genre using trained RandomForest model
- Shows model evaluation results
- Runs in Colab using **Pyngrok** tunnel

---

## 📊 Results
- **RandomForest**:
- Accuracy: ~85–90% on tabular features
- Works well with minimal preprocessing
- **CNN**:
- Accuracy: ~75–80% on Mel-spectrogram images
- Potential improvements with data augmentation & deeper architecture

---

## 📂 File Structure
├── music_genre_rf.pkl # Saved RandomForest model
├── scaler.pkl # Scaler for tabular features
├── label_encoder.pkl # Label encoder for genres
├── music_genre_cnn.h5 # Saved CNN model
├── music_genre_app.py # Streamlit app
└── README.md # Project documentation

---

## ▶️ Running the Project

### **Run in Colab**
1. Install dependencies:
   ```bash
   pip install librosa kagglehub tensorflow scikit-learn matplotlib seaborn streamlit pyngrok
Add your Ngrok auth token:


from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN_HERE")


Run Streamlit app:

streamlit run music_genre_app.py &


Connect via Ngrok:
public_url = ngrok.connect(addr=8501, proto="http")
print(public_url)



📌 Covered Topics
Audio Data Processing

Feature Extraction (MFCCs, Spectrograms)

Multi-class Classification

Scikit-learn Models

Convolutional Neural Networks (CNNs)

Interactive Deployment with Streamlit