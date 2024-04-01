import streamlit as st
import numpy as np

from helper_functions.classifier import classify_text
from helper_functions.topic_model import get_topic, generate_wordcloud_per_topic

# Define the Streamlit app
def main():
    st.title("Scam Text Classifier")
    st.markdown("<h2 style='font-size: 18px;'>Scam or not scam?</h2>", unsafe_allow_html=True)

    # Create a text input field for the user
    user_input = st.text_input("Enter your text here:")

    # Add a button to run the classification and topic modeling
    if st.button("Classify"):
        if user_input:
            classification_result = classify_text(user_input, model_path="./models/classfication/best_lstm.h5")
            # Display the classification result
            st.markdown("<p style='font-size: 24px; margin-top: 20px;'>Classification Result:</p>", unsafe_allow_html=True)
            if classification_result == 1:
                st.markdown("<p style='font-size: 36px; color: red;'>Scam!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 36px; color: green;'>Not scam!</p>", unsafe_allow_html=True)

            # Call the get_topic function to extract the topic and coherence score
            path = '../../data/train_data.csv'
            topic_chosen_dict, topic_label, topic_coher_score = get_topic(path, user_input)

            # Display the coherence score
            st.write(f"The coherence score of your text with the topic chosen is {round(topic_coher_score, 4)}")

            # Generate and display the word cloud
            st.markdown("<p style='font-size: 24px; margin-top: 20px;'>Word Cloud for Chosen Topic:</p>", unsafe_allow_html=True)
            generate_wordcloud_per_topic(topic_chosen_dict, topic_label)
        else:
            st.warning("Please enter some text.")

# Run the Streamlit app
if __name__ == "__main__":
    main()