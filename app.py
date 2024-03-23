import streamlit as st
import numpy as np

from helper_functions.classifier import classify_text

# Define the Streamlit app
def main():
    st.title("Scam Text Classifier")
    st.markdown("<h2 style='font-size: 18px;'>Scam or not scam?</h2>", unsafe_allow_html=True)

    # Create a text input field for the user
    user_input = st.text_input("Enter your text here:")

    # Add a button to run the classification
    if st.button("Classify"):
        if user_input:
            classification_result = classify_text(user_input, model_path="./models/classfication/best_lstm.h5")
            # Display the classification result
            st.markdown("<p style='font-size: 24px; margin-top: 20px;'>Classification Result:</p>", unsafe_allow_html=True)
            if classification_result == 1:
                st.markdown("<p style='font-size: 36px; color: red;'>Scam!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 36px; color: green;'>Not scam!</p>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text.")



# Run the Streamlit app
if __name__ == "__main__":
    main()
