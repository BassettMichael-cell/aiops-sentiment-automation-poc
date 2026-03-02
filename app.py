import streamlit as st
import joblib
import os

# Path to the model file created by the training pipeline
MODEL_PATH = "models/sentiment_model.joblib"

# Load model & vectorizer (cached so it only loads once)
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Make sure the training pipeline has run successfully (check the Actions tab).")
        st.stop()
    vectorizer, model = joblib.load(MODEL_PATH)
    return vectorizer, model

# Page title and description
st.title("Live Sentiment Analysis Demo")
st.markdown("""
This app uses the model automatically trained in the GitHub Actions pipeline.  
Paste any text (movie review, tweet, product feedback, etc.) and get an instant prediction.
""")

# Load the model
vectorizer, model = load_model()

# User input
user_text = st.text_area(
    "Enter text to analyze:",
    height=120,
    placeholder="This movie was absolutely fantastic! Loved every minute."
)

# Prediction button
if st.button("Predict Sentiment"):
    if user_text.strip():
        with st.spinner("Analyzing..."):
            # Vectorize the input text
            text_vec = vectorizer.transform([user_text])
            
            # Make prediction
            prediction = model.predict(text_vec)[0]
            prob = model.predict_proba(text_vec)[0]
            
            # Interpret result
            label = "Positive 😊" if prediction == 1 else "Negative 😔"
            confidence = max(prob) * 100  # highest probability as confidence
            
            # Display result
            st.subheader(f"Prediction: **{label}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            if prediction == 1:
                st.success("Positive sentiment detected!")
            else:
                st.warning("Negative sentiment detected.")
    else:
        st.warning("Please enter some text to analyze!")

# Footer
st.markdown("---")
st.caption(
    "Powered by scikit-learn Logistic Regression + TF-IDF | "
    "Trained on 1,000 IMDb samples | "
    "Automatically built & updated via GitHub Actions"
)
