import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="LIAR Dataset Explorer & Model Visualizer",
    page_icon="🤥",
    layout="wide"
)

# App Title & Header
st.title("🤥 LIAR Dataset Explorer & Fake News Detector")
st.markdown(
    """
    Welcome to the interactive dashboard for the **LIAR Dataset** (Pants on Fire fake news detection benchmark).
    Use the sidebar to navigate between dataset exploration, model evaluation, and live predictions.
    """
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview & Data", "Model Evaluation", "Live Prediction"])

# Mock/Sample Data Loading
@st.cache_data
def load_sample_data():
    columns = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affinity", 
               "barely_true_cnt", "false_cnt", "half_true_cnt", "mostly_true_cnt", "pants_on_fire_cnt", "context"]
    
    sample_rows = [
        ["1001.json", "half-true", "Says the Bush administration ignored warnings before 9/11.", "history", "Barack Obama", "President", "Illinois", "democrat", 0, 1, 2, 0, 0, "a speech in Ohio"],
        ["1002.json", "pants-fire", "The Health Care law will create death panels to decide if seniors live or die.", "health-care", "Sarah Palin", "Governor", "Alaska", "republican", 1, 3, 0, 1, 4, "a Facebook post"],
        ["1003.json", "mostly-true", "Unemployment has decreased continuously over the last three years.", "economy", "Joe Biden", "Vice President", "Delaware", "democrat", 1, 0, 1, 5, 0, "a TV debate"],
        ["1004.json", "false", "Tax rates under current legislation are the highest they have ever been in history.", "taxes", "Donald Trump", "Businessman", "New York", "republican", 5, 8, 2, 1, 3, "a rally speech"],
        ["1005.json", "barely-true", "We have imported more oil this month than during any previous period.", "energy", "Mitt Romney", "Governor", "Massachusetts", "republican", 2, 2, 1, 1, 0, "a press conference"]
    ]
    return pd.DataFrame(sample_rows, columns=columns)

df = load_sample_data()

# ---------------------------------------------------------
# PAGE 1: Dataset Overview & Data Exploration
# ---------------------------------------------------------
if page == "Overview & Data":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.st_columns = st.columns(4)
    col1.metric("Total Samples", "12,836")
    col2.metric("Classes", "6 Labels")
    col3.metric("Train Split", "10,269")
    col4.metric("Test Split", "1,283")

    st.subheader("Sample Statements")
    st.dataframe(df[["id", "label", "statement", "speaker", "party_affinity", "context"]], use_container_width=True)

    st.subheader("Label Distribution")
    label_counts = pd.DataFrame({
        'Label': ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true'],
        'Count': [1047, 2507, 2103, 2627, 2454, 2053]
    })
    st.bar_chart(label_counts.set_index('Label'))

# ---------------------------------------------------------
# PAGE 2: Model Evaluation
# ---------------------------------------------------------
elif page == "Model Evaluation":
    st.header("📈 Model Performance Metrics")

    model_choice = st.selectbox("Select Model Architecture", ["BERT Fine-Tuned", "RoBERTa Classifier", "TF-IDF + Logistic Regression"])

    col1, col2, col3 = st.columns(3)
    if model_choice == "RoBERTa Classifier":
        col1.metric("Accuracy", "45.2%", "+2.1%")
        col2.metric("F1-Score (Macro)", "0.44", "+0.03")
        col3.metric("Loss", "1.32", "-0.12")
    elif model_choice == "BERT Fine-Tuned":
        col1.metric("Accuracy", "43.1%", "+0.5%")
        col2.metric("F1-Score (Macro)", "0.41", "+0.01")
        col3.metric("Loss", "1.41", "-0.05")
    else:
        col1.metric("Accuracy", "26.4%", "Baseline")
        col2.metric("F1-Score (Macro)", "0.25", "Baseline")
        col3.metric("Loss", "1.78", "Baseline")

    st.subheader("Confusion Matrix")
    st.markdown("Visualizing classification spread across the 6 truthfulness fine-grained levels.")
    
    labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    dummy_cm = np.random.randint(10, 100, size=(6, 6))
    np.fill_diagonal(dummy_cm, np.random.randint(150, 300, size=6))
    cm_df = pd.DataFrame(dummy_cm, index=labels, columns=labels)
    
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: Live Prediction Sandbox
# ---------------------------------------------------------
elif page == "Live Prediction":
    st.header("🔍 Live Truthfulness Predictor")
    st.markdown("Test statements manually to see predicted truthfulness probabilities.")

    with st.form("prediction_form"):
        statement_input = st.text_area("Enter Statement:", "The national debt has decreased significantly over the last quarter.")
        speaker_input = st.text_input("Speaker:", "John Doe")
        party_input = st.selectbox("Party Identification:", ["democrat", "republican", "independent", "none"])
        context_input = st.text_input("Context/Venue:", "Television Interview")
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        st.subheader("Prediction Results")
        
        # Placeholder prediction logic
        probs = {
            'pants-fire': 0.05,
            'false': 0.15,
            'barely-true': 0.35,
            'half-true': 0.25,
            'mostly-true': 0.15,
            'true': 0.05
        }
        
        predicted_label = max(probs, key=probs.get)
        st.success(f"Predicted Class: **{predicted_label.upper()}**")

        st.subheader("Confidence Spread")
        prob_df = pd.DataFrame(list(probs.items()), columns=['Label', 'Probability'])
        st.bar_chart(prob_df.set_index('Label'))import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="LIAR Dataset Explorer & Model Visualizer",
    page_icon="🤥",
    layout="wide"
)

# App Title & Header
st.title("🤥 LIAR Dataset Explorer & Fake News Detector")
st.markdown(
    """
    Welcome to the interactive dashboard for the **LIAR Dataset** (Pants on Fire fake news detection benchmark).
    Use the sidebar to navigate between dataset exploration, model evaluation, and live predictions.
    """
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview & Data", "Model Evaluation", "Live Prediction"])

# Mock/Sample Data Loading
@st.cache_data
def load_sample_data():
    columns = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affinity", 
               "barely_true_cnt", "false_cnt", "half_true_cnt", "mostly_true_cnt", "pants_on_fire_cnt", "context"]
    
    sample_rows = [
        ["1001.json", "half-true", "Says the Bush administration ignored warnings before 9/11.", "history", "Barack Obama", "President", "Illinois", "democrat", 0, 1, 2, 0, 0, "a speech in Ohio"],
        ["1002.json", "pants-fire", "The Health Care law will create death panels to decide if seniors live or die.", "health-care", "Sarah Palin", "Governor", "Alaska", "republican", 1, 3, 0, 1, 4, "a Facebook post"],
        ["1003.json", "mostly-true", "Unemployment has decreased continuously over the last three years.", "economy", "Joe Biden", "Vice President", "Delaware", "democrat", 1, 0, 1, 5, 0, "a TV debate"],
        ["1004.json", "false", "Tax rates under current legislation are the highest they have ever been in history.", "taxes", "Donald Trump", "Businessman", "New York", "republican", 5, 8, 2, 1, 3, "a rally speech"],
        ["1005.json", "barely-true", "We have imported more oil this month than during any previous period.", "energy", "Mitt Romney", "Governor", "Massachusetts", "republican", 2, 2, 1, 1, 0, "a press conference"]
    ]
    return pd.DataFrame(sample_rows, columns=columns)

df = load_sample_data()

# ---------------------------------------------------------
# PAGE 1: Dataset Overview & Data Exploration
# ---------------------------------------------------------
if page == "Overview & Data":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.st_columns = st.columns(4)
    col1.metric("Total Samples", "12,836")
    col2.metric("Classes", "6 Labels")
    col3.metric("Train Split", "10,269")
    col4.metric("Test Split", "1,283")

    st.subheader("Sample Statements")
    st.dataframe(df[["id", "label", "statement", "speaker", "party_affinity", "context"]], use_container_width=True)

    st.subheader("Label Distribution")
    label_counts = pd.DataFrame({
        'Label': ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true'],
        'Count': [1047, 2507, 2103, 2627, 2454, 2053]
    })
    st.bar_chart(label_counts.set_index('Label'))

# ---------------------------------------------------------
# PAGE 2: Model Evaluation
# ---------------------------------------------------------
elif page == "Model Evaluation":
    st.header("📈 Model Performance Metrics")

    model_choice = st.selectbox("Select Model Architecture", ["BERT Fine-Tuned", "RoBERTa Classifier", "TF-IDF + Logistic Regression"])

    col1, col2, col3 = st.columns(3)
    if model_choice == "RoBERTa Classifier":
        col1.metric("Accuracy", "45.2%", "+2.1%")
        col2.metric("F1-Score (Macro)", "0.44", "+0.03")
        col3.metric("Loss", "1.32", "-0.12")
    elif model_choice == "BERT Fine-Tuned":
        col1.metric("Accuracy", "43.1%", "+0.5%")
        col2.metric("F1-Score (Macro)", "0.41", "+0.01")
        col3.metric("Loss", "1.41", "-0.05")
    else:
        col1.metric("Accuracy", "26.4%", "Baseline")
        col2.metric("F1-Score (Macro)", "0.25", "Baseline")
        col3.metric("Loss", "1.78", "Baseline")

    st.subheader("Confusion Matrix")
    st.markdown("Visualizing classification spread across the 6 truthfulness fine-grained levels.")
    
    labels = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    dummy_cm = np.random.randint(10, 100, size=(6, 6))
    np.fill_diagonal(dummy_cm, np.random.randint(150, 300, size=6))
    cm_df = pd.DataFrame(dummy_cm, index=labels, columns=labels)
    
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: Live Prediction Sandbox
# ---------------------------------------------------------
elif page == "Live Prediction":
    st.header("🔍 Live Truthfulness Predictor")
    st.markdown("Test statements manually to see predicted truthfulness probabilities.")

    with st.form("prediction_form"):
        statement_input = st.text_area("Enter Statement:", "The national debt has decreased significantly over the last quarter.")
        speaker_input = st.text_input("Speaker:", "John Doe")
        party_input = st.selectbox("Party Identification:", ["democrat", "republican", "independent", "none"])
        context_input = st.text_input("Context/Venue:", "Television Interview")
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        st.subheader("Prediction Results")
        
        # Placeholder prediction logic
        probs = {
            'pants-fire': 0.05,
            'false': 0.15,
            'barely-true': 0.35,
            'half-true': 0.25,
            'mostly-true': 0.15,
            'true': 0.05
        }
        
        predicted_label = max(probs, key=probs.get)
        st.success(f"Predicted Class: **{predicted_label.upper()}**")

        st.subheader("Confidence Spread")
        prob_df = pd.DataFrame(list(probs.items()), columns=['Label', 'Probability'])
        st.bar_chart(prob_df.set_index('Label'))
