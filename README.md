# Student Performance Analysis & Score Prediction 📊🤖

A small end-to-end **Data Analysis + Machine Learning** project built with Python.

The project analyzes how academic and lifestyle factors relate to student performance and uses **Linear Regression** to predict a student's final exam score.

## 🎯 Project Objective

Predict `final_exam_score` using:

- Study hours per day
- Attendance percentage
- Sleep hours
- Assignment completion percentage
- Previous exam score

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📁 Project Structure

```text
student-performance-analysis-ml/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── student_analysis.ipynb
│
├── src/
│   └── train_model.py
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   └── study_vs_marks.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## 🔍 Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Linear Regression
   ↓
Model Evaluation
   ↓
Prediction & Insights
```

## 📈 Analysis Performed

- Dataset inspection
- Missing-value detection and median imputation
- Descriptive statistics
- Correlation analysis
- Study-hours vs score analysis
- Actual vs predicted score visualization

## 🤖 Machine Learning

### Algorithm
**Linear Regression**

### Evaluation Metrics

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — R-squared

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-analysis-ml.git
cd student-performance-analysis-ml
```

### 2. Create a virtual environment (recommended)

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the ML model

```bash
python src/train_model.py
```

### 5. Open the notebook

```bash
jupyter notebook
```

Then open:

`notebooks/student_analysis.ipynb`

## 💡 Key Takeaway

Student performance is influenced by multiple academic and lifestyle factors. In this project, the model combines these variables instead of relying on a single factor to estimate final performance.

## 🚀 Future Improvements

- Compare Linear Regression with Random Forest and Gradient Boosting
- Add classification: Pass/Fail prediction
- Build an interactive Streamlit dashboard
- Perform cross-validation and hyperparameter tuning
- Deploy the model as a web application

## 👨‍💻 Author

**Ayush**  
BTech CSE — IIIT Una

---

⭐ If you found this project useful, consider starring the repository.
