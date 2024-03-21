import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Define the Streamlit app
def main():
    st.title("Text Classification App")

    # Create a text input field for the user
    user_input = st.text_input("Enter your text here:")

    # Add a button to run the classification
    if st.button("Classify"):
        if user_input:
            # Load LSTM model
            model_path = './models/classification/best_lstm.h5'
            model = load_model(model_path)
            
            # Make predictions using the loaded LSTM model
            classification_result = classify_text(user_input, model)
            
            # Display the classification result
            st.write("Classification Result:", classification_result)
        else:
            st.warning("Please enter some text.")

# Function to classify text
def classify_text(text, model):
    # Convert text to numpy array
    text_array = np.array([text])
    # Make predictions using the loaded LSTM model
    predictions = model.predict(text_array)
    return predictions

# Run the Streamlit app
if __name__ == "__main__":
    main()
