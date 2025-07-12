import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and vectorizer
model_nb = joblib.load('model_nb.pkl')
model_lr = joblib.load('model_lr.pkl')
model_svm = joblib.load('model_svm.pkl')
tfidf = joblib.load('tfidf.pkl')

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

st.title("🧠 Emotion Detection from Text")
st.write("Enter a sentence and choose a model to detect its emotion.")

text_input = st.text_area("Enter text:")
model_choice = st.selectbox("Select Model", ["Naive Bayes", "Logistic Regression", "SVM"])

if st.button("Detect Emotion"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        vectorized = tfidf.transform([cleaned])
        if model_choice == "Naive Bayes":
            prediction = model_nb.predict(vectorized)[0]
        elif model_choice == "Logistic Regression":
            prediction = model_lr.predict(vectorized)[0]
        else:
            prediction = model_svm.predict(vectorized)[0]
        st.success(f"Predicted Emotion: **{prediction}**")

# Accuracy comparison (static values from your script)
accuracies = {
    'Naive Bayes': joblib.load('acc_nb.pkl') if 'acc_nb.pkl' in joblib.os.listdir() else 0.85,
    'Logistic Regression': joblib.load('acc_lr.pkl') if 'acc_lr.pkl' in joblib.os.listdir() else 0.88,
    'SVM': joblib.load('acc_svm.pkl') if 'acc_svm.pkl' in joblib.os.listdir() else 0.90
}

if st.checkbox("Show Model Accuracy Comparison"):
    fig, ax = plt.subplots()
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    st.pyplot(fig)