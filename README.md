# ❤️‍🩹 Heart Disease Prediction using Machine Learning

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" />
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy" />
<img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn" />
<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit" />
<img src="https://img.shields.io/badge/Joblib-Model%20Serialization-4B8BBE?style=for-the-badge" />

</p>

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to predict the likelihood of heart disease using patient medical attributes.

Multiple classification algorithms were trained and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Decision Tree  
- Support Vector Machine (SVM)  

Since this is a medical diagnosis problem, minimizing **False Negatives** is critical. Therefore, special importance was given to **Recall** during model evaluation.

After comparing all models, **KNN was selected** as the final model due to:

- High Accuracy  
- Strong Recall performance  
- Balanced overall metrics  

The final model is deployed using **Streamlit** for real-time predictions.


## 🎯 Problem Statement

Given patient health parameters such as age, cholesterol, ECG results, and heart rate, predict whether the person is likely to have heart disease.

⚠️ In medical classification:

- False Positive → Additional testing (manageable)  
- False Negative → Missed diagnosis (dangerous)  

Hence, Recall is prioritized along with accuracy to reduce the risk of misclassifying heart patients as healthy.


## 🧠 Machine Learning Workflow

1. Data Loading (`heart.csv`)
2. Exploratory Data Analysis
3. One-Hot Encoding for categorical features
4. Feature Scaling using `StandardScaler`
5. Training multiple classification models
6. Model Evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
7. Model Selection (KNN chosen)
8. Model Serialization using Joblib
9. Deployment with Streamlit (`app.py`)


## 📊 Final Model

- Algorithm: **K-Nearest Neighbors (KNN)**
- Scaler: `StandardScaler`
- Saved Model: `KNN_heart_model.pkl`
- Saved Scaler: `sc.pkl`
- Saved Feature Columns: `columns.pkl`


## 🖥️ Streamlit Web Application

The web application allows users to:

- Enter medical parameters
- Click **Predict**
- Receive instant prediction results

If predicted positive:
> ⚠️ The model predicts likelihood of heart disease. Please consult a healthcare professional.

If predicted negative:
> 💚 The model predicts lower likelihood. Maintain regular health check-ups.

## 📷 Demo Screenshots
<img width="1919" height="860" alt="image" src="https://github.com/user-attachments/assets/4bf8956b-aa90-43fb-ac1b-ddb958dddf03" />
<img width="1919" height="860" alt="image" src="https://github.com/user-attachments/assets/f0bcb30d-0af5-4179-872b-004fbc63433e" />


## 🗂️ Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                  # Streamlit web application
├── heart.ipynb             # Model training & experimentation notebook
├── heart.csv               # Dataset
├── KNN_heart_model.pkl     # Trained KNN model
├── sc.pkl                  # Saved StandardScaler
├── columns.pkl             # Feature column order reference
├── requirements.txt
└── README.md
```


## ⚙️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Pandas | Data Handling |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning Models |
| Streamlit | Web App Deployment |
| Joblib | Model Serialization |


## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mrunmayee3108/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📌 Key Learning Outcomes

- Compared multiple classification algorithms
- Understood importance of Recall in medical datasets
- Implemented preprocessing pipeline properly
- Maintained feature consistency during deployment
- Built and deployed a real-world ML application


## ⚠️ Disclaimer

This project is not a substitute for professional medical diagnosis.

## 💚 Support

If you liked the project, consider giving it a star ⭐ on github! 


## 👩‍💻 Author

**Mrunmayee Potdar**   

