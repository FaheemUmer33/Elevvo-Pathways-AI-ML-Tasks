# 🚦 Task 8: Traffic Sign Recognition

This project focuses on classifying traffic signs using deep learning models. It uses the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset from Kaggle and provides an interactive **Streamlit web app** for testing predictions.

---

## 📁 Dataset

**Source**: [Kaggle - Traffic Signs Preprocessed](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed)

**Path After Mounting KaggleHub**:
/root/.cache/kagglehub/datasets/valentynsichkar/traffic-signs-preprocessed/versions/2


---

## 🧠 Models Used

1. **Custom CNN** - A Convolutional Neural Network built from scratch using Keras.
2. **Pretrained MobileNetV2** - A transfer learning model using Keras applications.

---

## 🧰 Tools & Libraries

- Python
- TensorFlow / Keras
- OpenCV
- scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Pyngrok

---

## 🛠️ Steps Performed

### ✅ Step 1: Load Preprocessed Dataset
- Used `.pickle` files for train, validation, and test sets.

### ✅ Step 2: Preprocess the Data
- Normalized image pixels to range [0, 1].
- One-hot encoded target labels.

### ✅ Step 3: Build Custom CNN
- 3 convolutional layers with ReLU + MaxPooling.
- Fully connected Dense layer with softmax output.
- Used `Adam` optimizer and `categorical_crossentropy` loss.

### ✅ Step 4: Train the CNN
- Trained for 15 epochs with validation data.
- Observed high accuracy (above 95%).

### ✅ Step 5: Evaluate and Analyze
- Evaluated on test set.
- Visualized confusion matrix using Seaborn heatmap.

---

## 🧪 Step 6: Predict on Custom Images
- Uploaded custom traffic sign image.
- Used OpenCV to preprocess and predict label.
- Prediction: Displayed top predicted class with label.

---

## 💾 Step 7: Save & Load Model
- Saved trained model as `traffic_sign_cnn_model.h5`.

---

## 🌟 Bonus Tasks

### 🎯 Bonus 1: Data Augmentation
- Added `ImageDataGenerator` for augmenting training images.
- Improved generalization and reduced overfitting.

### 🧠 Bonus 2: MobileNetV2 Transfer Learning
- Imported `MobileNetV2` with `imagenet` weights.
- Added custom Dense layers for 43-class output.
- Fine-tuned and compared results with CNN.

---

## 🌐 Streamlit Web App

### Features:
- Upload image of a traffic sign.
- See predicted label instantly.
- Real-time traffic sign recognition.

### Deployment:
Used **Streamlit + Pyngrok** for secure public access.

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("Streamlit app running at:", public_url)
!streamlit run app.py &> /dev/null &


📊 Results
Model	Accuracy	Notes
Custom CNN	~96%	Fast, good for smaller devices
MobileNetV2	~98%	Better generalization, transfer learning



📌 Folder Structure
├── traffic_sign_cnn_model.h5
├── traffic_sign_mobilenet_model.h5
├── app.py                     # Streamlit App
├── utils.py                   # Utility functions
├── README.md
└── /images                    # Custom test images



🚀 How to Run


1. Install dependencies
pip install tensorflow opencv-python streamlit pyngrok matplotlib scikit-learn


2. Start the app with Ngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8501)
print(public_url)
!streamlit run app.py &> /dev/null &