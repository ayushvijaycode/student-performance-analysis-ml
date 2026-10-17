from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
FEATURES = ["study_hours_per_day", "attendance_percent", "sleep_hours",
            "assignment_completion_percent", "previous_exam_score"]
TARGET = "final_exam_score"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data[FEATURES + [TARGET]] = data[FEATURES + [TARGET]].apply(lambda col: col.fillna(col.median()))
    return data

@st.cache_resource
def fit_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R²": r2_score(y_test, pred),
    }
    return model, X_test, y_test, pred, scores

df = load_data()
model, X_test, y_test, predictions, scores = fit_model()

st.title("🎓 Student Performance Predictor")
st.write("Explore student data and estimate final exam scores with a Linear Regression model.")
st.warning("Educational demo only. This app uses synthetic sample data. Predictions are illustrative and must not be used for real academic decisions.")

tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict a Score", "📊 Explore Data", "🧪 Model Performance"])

with tab_predict:
    st.subheader("Enter student details")
    left, right = st.columns(2)
    with left:
        study = st.slider("Study hours per day", 1.0, 10.0, 5.0, 0.5)
        attendance = st.slider("Attendance (%)", 45, 100, 80)
        sleep = st.slider("Sleep hours per day", 4.5, 9.5, 7.0, 0.5)
    with right:
        assignments = st.slider("Assignment completion (%)", 30, 100, 80)
        previous = st.slider("Previous exam score", 30, 98, 70)
    if st.button("Predict Final Exam Score", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "study_hours_per_day": study,
            "attendance_percent": attendance,
            "sleep_hours": sleep,
            "assignment_completion_percent": assignments,
            "previous_exam_score": previous,
        }], columns=FEATURES)
        result = float(model.predict(row)[0])
        st.metric("Estimated final exam score", f"{max(0, min(100, result)):.1f} / 100")
        st.caption("This is a model estimate, not a guaranteed exam result.")

with tab_data:
    a, b, c = st.columns(3)
    a.metric("Student records", len(df))
    b.metric("Input features", len(FEATURES))
    c.metric("Missing values after cleaning", int(df[FEATURES + [TARGET]].isna().sum().sum()))
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Study hours vs final exam score")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["study_hours_per_day"], df[TARGET], alpha=0.75)
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Final exam score")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Feature correlation")
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    img = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_model:
    cols = st.columns(3)
    for col, (name, value) in zip(cols, scores.items()):
        col.metric(name, f"{value:.3f}" if name == "R²" else f"{value:.2f}")
    st.write("Metrics are calculated on a held-out 20% test split. Synthetic data means these are demonstration results.")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions, alpha=0.75)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], "--")
    ax.set_xlabel("Actual score")
    ax.set_ylabel("Predicted score")
    ax.set_title("Actual vs predicted")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("About this project"):
    st.markdown("**Algorithm:** Linear Regression  \n**Dataset:** 150 synthetic records  \n**Features:** study hours, attendance, sleep, assignment completion, previous exam score  \n**Stack:** Python, Pandas, Matplotlib, Scikit-learn, Streamlit")
    st.markdown("[View GitHub repository](https://github.com/ayushvijaycode/student-performance-analysis-ml)")
