import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    .main-title { font-size: 2.6rem; font-weight: 800; color: #14213d; }
    .sub-title { color: #65748b; font-size: 1.05rem; }
    div[data-testid="stMetric"] {
        background: white; padding: 18px; border-radius: 14px;
        border: 1px solid #e6eaf1; box-shadow: 0 3px 12px rgba(30,50,80,.04);
    }
    .result-card { padding: 22px; border-radius: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------- Load and train model ----------
@st.cache_resource
def train_model():
    df = pd.read_csv("loan_approval.csv")

    # Clean data similar to the notebook
    df = df.drop_duplicates().copy()

    df["income"] = (
        df["income"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["income"] = pd.to_numeric(df["income"], errors="coerce")

    # Treat credit scores above 900 as invalid, then fill missing values
    df.loc[df["credit_score"] > 900, "credit_score"] = None
    df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median())
    df["years_working"] = df["years_working"].fillna(df["years_working"].median())

    df["employment"] = df["employment"].astype(str).str.strip().str.capitalize()
    df["employment"] = df["employment"].replace({
        "Self-employed": "Self employment",
        "Self employed": "Self employment"
    })
    df["employment"] = df["employment"].map({
        "Salaried": 1,
        "Self employment": 0
    })

    df["loan_status"] = df["loan_status"].astype(str).str.strip().str.capitalize()

    features = ["income", "credit_score", "loan_amount", "years_working", "employment"]
    X = df[features]
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=1)
    model.fit(X_train_scaled, y_train)

    test_prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, test_prediction)

    return model, scaler, accuracy, df, features

model, scaler, accuracy, data, features = train_model()

# ---------- Header ----------
st.markdown('<div class="main-title">🏦 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An ML-powered demo that predicts a loan decision using your financial details.</div>', unsafe_allow_html=True)
st.write("")

m1, m2, m3 = st.columns(3)
m1.metric("Model", "SVM (RBF)")
m2.metric("Test Accuracy", f"{accuracy:.1%}")
m3.metric("Dataset Records", f"{len(data):,}")

st.divider()

# ---------- Inputs ----------
st.subheader("📝 Applicant Details")
st.caption("Fill in the details below and click Predict to see the model's result.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income (₹)",
            min_value=0, max_value=10_000_000,
            value=800_000, step=10_000,
            help="Enter annual income in rupees."
        )
        credit_score = st.slider(
            "Credit Score", min_value=300, max_value=900, value=760,
            help="Typical credit-score range shown for this demo."
        )
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0, max_value=10_000_000,
            value=300_000, step=10_000
        )

    with col2:
        years_working = st.slider(
            "Years of Work Experience", min_value=0, max_value=50, value=6
        )
        employment_label = st.selectbox(
            "Employment Type", ["Salaried", "Self-employed"]
        )

    predict_button = st.form_submit_button("🔍 Predict Loan Decision", use_container_width=True)

if predict_button:
    employment_value = 1 if employment_label == "Salaried" else 0

    new_person = pd.DataFrame([{
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "years_working": years_working,
        "employment": employment_value
    }])

    new_scaled = scaler.transform(new_person[features])
    prediction = model.predict(new_scaled)[0]

    st.subheader("📌 Prediction Result")
    if prediction == "Approved":
        st.success("✅ Predicted: LOAN APPROVED")
        st.markdown("The model predicts that this application may be approved.")
    else:
        st.error("❌ Predicted: LOAN REJECTED")
        st.markdown("The model predicts that this application may be rejected.")

    st.info("This is an educational ML prediction—not an actual bank decision or financial advice.")

# ---------- Model info ----------
with st.expander("ℹ️ About this ML model"):
    st.write("""
    - **Algorithm:** Support Vector Machine (SVC)
    - **Kernel:** RBF
    - **C parameter:** 1
    - **Preprocessing:** StandardScaler
    - **Train/test split:** 80% / 20%
    - **Random state:** 42
    - **Input features:** Income, Credit Score, Loan Amount, Years Working, Employment
    """)
    st.write("Employment is encoded as Salaried = 1 and Self-employed = 0.")

with st.expander("📊 Explore the dataset"):
    st.write("Loan status counts")
    st.bar_chart(data["loan_status"].value_counts())
    st.write("Preview of cleaned dataset")
    st.dataframe(data.head(10), use_container_width=True)

st.caption("Made with Python, Scikit-learn and Streamlit | ML learning project")
