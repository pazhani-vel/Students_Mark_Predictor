🎓 Student Exam Score Predictor

A machine learning application that predicts a student's exam score based on key lifestyle and academic habits. Built with Python, scikit-learn, and deployed as an interactive web app using Streamlit.

---

📌 Overview

This project explores how daily student behaviors — such as study hours, attendance, sleep, mental health, and part-time employment — influence academic performance. Using a structured ML pipeline, multiple regression models are trained, evaluated, and compared to select the best-performing model, which is then served through a user-friendly Streamlit interface.

---

## 🗂️ Project Structure

```
Student_Score_Predictor/
├── Dataset/
│   └── student_habits_performance.csv   # Raw dataset (909 records, 16 features)
└── Score_predictor/
    ├── predictor.ipynb                  # EDA, feature engineering & model training notebook
    ├── best_model.pkl                   # Serialized best-performing model
    └── app.py                           # Streamlit web application
```

---

## 📊 Dataset

The dataset contains **909 student records** with **16 features** covering academic, behavioral, and lifestyle attributes:

| Feature | Description |
|---|---|
| `study_hours_per_day` | Daily study hours |
| `attendance_percentage` | Class attendance rate (%) |
| `mental_health_rating` | Self-reported mental health score (1–10) |
| `sleep_hours` | Average nightly sleep hours |
| `part_time_job` | Whether the student holds a part-time job (Yes/No) |
| `exam_score` | Final exam score (target variable, 0–100) |
| + 10 other features | Age, gender, social media hours, Netflix hours, diet quality, exercise frequency, parental education level, internet quality, extracurricular participation |

---

## ⚙️ ML Pipeline

### 1. Data Preprocessing
- Removed missing values and duplicate records
- Label-encoded categorical features (e.g., `part_time_job`)
- Selected the 5 most relevant features based on correlation analysis

### 2. Exploratory Data Analysis (EDA)
- Distribution histograms and count plots for all features
- Correlation heatmap to identify relationships with exam score
- Scatter plots and box plots to visualize feature-target relationships

### 3. Model Training & Selection
Three regression models were trained and tuned using **5-fold cross-validated GridSearchCV**:

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| Decision Tree Regressor | `max_depth`, `min_samples_split` |
| Random Forest Regressor | `n_estimators`, `max_depth` |

Models were evaluated on **RMSE** and **R² Score**. The best-performing model was serialized using `joblib` as `best_model.pkl`.

---

## 🚀 Web Application

The Streamlit app (`app.py`) loads the saved model and provides an interactive interface for real-time predictions.

**Input Features (via sliders & dropdowns):**
- Study Hours per Day (0–12)
- Attendance Percentage (0–100%)
- Mental Health Rating (1–10)
- Sleep Hours per Night (0–12)
- Part-Time Job (Yes / No)

**Output:** Predicted exam score (clamped to a valid 0–100 range)

### Run the App

```bash
# Install dependencies
pip install streamlit scikit-learn numpy joblib

# Launch the app
cd Score_predictor
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn (Linear Regression, Decision Tree, Random Forest, GridSearchCV)
- **Model Serialization:** Joblib
- **Web App:** Streamlit

---

## 📁 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Results

The best model was selected based on the lowest RMSE across the three candidates. The final model is trained on the full dataset before deployment to maximize predictive accuracy.

---

## 🙌 Acknowledgements

- Dataset sourced for academic and educational purposes
- Built as a learning project to demonstrate an end-to-end ML workflow: data cleaning → EDA → feature engineering → model selection → deployment
