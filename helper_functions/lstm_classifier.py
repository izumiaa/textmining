import pandas as pd
import re 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from spellchecker import SpellChecker
from empath import Empath
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, concatenate, Dense, SpatialDropout1D, Bidirectional, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

stop_list = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.porter.PorterStemmer()
emotion_labels = ['deception', 'money','payment','celebration','achievement']

#################################################### 
# convert the text input to dataframe 
####################################################

def convert_to_dataframe(text):
    df = pd.DataFrame({'text': [text]})
    return df

#################################################### 
# Feature eng functions 
####################################################

def remove_unknown_ch(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s' + re.escape(punctuation) + ']', '', str(text))
    return cleaned_text

def preprocess_text(text):
    text = re.sub(r'[^a-z\s]', '', str(text))
    text = text.lower()
    tokens = text.split()
    stop_list = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_list]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def calculate_punctuation_percentage(text):
    #percentage of punctuations to characters
    total_chars = len(text)
    punctuation_chars = sum([1 for char in text if char in punctuation])
    return (punctuation_chars / total_chars) * 100 if total_chars > 0 else 0

def calculate_num_chars(text):
    return len(text)

def calculate_num_words(text):
    words = word_tokenize(text)
    num_words = 0
    for word in words:
        if word in punctuation:
            pass
        else:
            num_words += 1
    return num_words

def calculate_num_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def count_misspellings(text):
    words = word_tokenize(text)
    spell = SpellChecker()
    num_mispelled = len(list(spell.unknown(words)))
    return(num_mispelled)

def calculate_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def determine_polarity(score): 
    return 'positive' if score > 0 else ('negative' if score < 0 else 'neutral')

def extract_emotions(text):
    """
    Each value epresents the degree to which the input text 
    expresses or relates to each predefined category
    0 - No relation
    1 - Strong relation
    """
    lexicon = Empath()
    emotions = lexicon.analyze(str(text), normalize=True)
    return emotions

def check_url_presence(text):
    if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text) != None:
        return 1
    else:
        return 0

def check_phone_number_presence(text):
    #may need to add more patterns for phone numbers
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) != None:
        return 1
    else:
        return 0

# Function to count number of POS verb tags
def count_pos_verbs(text):
    tokens = nltk.pos_tag(word_tokenize(text))
    verb_tags = [tag for word, tag in tokens if tag.startswith('VB')]
    return len(verb_tags)

def calculate_spaces_percentage(text):
    total_chars = len(text)
    num_spaces = text.count(' ')
    return (num_spaces / total_chars) * 100 if total_chars > 0 else 0

# def encode(text):
#     if text == 'spam':
#         return 1
#     else:
#         return 0
    
def generate_features(df):
    df['cleaned_text'] = df['text'].apply(remove_unknown_ch)
    df['preprocessed_tokens'] = df['text'].apply(preprocess_text)

    df['punctuation_percentage'] = df['cleaned_text'].apply(calculate_punctuation_percentage)
    df['num_chars'] = df['cleaned_text'].apply(calculate_num_chars)
    df['num_sentences'] = df['cleaned_text'].apply(calculate_num_sentences)
    df['num_words'] = df['cleaned_text'].apply(calculate_num_words)
    df['num_misspellings'] = df['cleaned_text'].apply(count_misspellings)
    df['misspelling_percentage'] = df['num_misspellings']/df['num_words'] * 100
    df['num_pos_verbs'] = df['cleaned_text'].apply(count_pos_verbs)
    df['pos_verbs_percentage'] = df['num_pos_verbs']/df['num_words'] * 100
    df['spaces_percentage'] = df['cleaned_text'].apply(calculate_spaces_percentage)

    df['sentiment_score'] = df['cleaned_text'].apply(calculate_sentiment_score)
    df['polarity'] = df['sentiment_score'].apply(determine_polarity)
    df['emotions'] = df['cleaned_text'].apply(extract_emotions)
    for label in emotion_labels:
        col_name = label.lower() + '_score'
        df[col_name] = df['emotions'].apply(lambda x: x[label] if x is not None and label in x else 0)

    df['url_presence'] = df['cleaned_text'].apply(check_url_presence)
    df['phone_number_presence'] = df['cleaned_text'].apply(check_phone_number_presence)
    # df['binary_label'] = df['label'].apply(encode)
    return df

#################################################### 
# Preprocess text 
####################################################

def preprocess_text2(df):
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    mask = df['num_pos_verbs'] > df['num_words']
    df.loc[mask, 'pos_verbs_percentage'] = 100
    return df

#################################################### 
# Prep and tokenize
####################################################
def prep_and_tokenize(df): 
    # Define features
    features = ['num_sentences', 'misspelling_percentage', 'pos_verbs_percentage',
                'spaces_percentage', 'sentiment_score', 'money_score', 'payment_score',
                'celebration_score', 'achievement_score', 'url_presence',
                'phone_number_presence']
    
    input_text_data = df['cleaned_text'].astype(str)
    input_numerical_features = df[features].values
    # input_labels = df['binary_label']

    # Text data preprocessing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(input_text_data)
    X_input_text = tokenizer.texts_to_sequences(input_text_data)
    max_length = 200 # majority of sequences have less than 200 tokens
    X_input_text = pad_sequences(X_input_text, maxlen=max_length)
    return X_input_text, input_numerical_features

#################################################### 
# Return the classification 
####################################################

def get_classification(X_input_text, input_numerical_features, checkpoint_filepath):
    # Callback
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode='min',
                                        verbose=1)
    
    # Predict labels for test data
    best_model = load_model(checkpoint_filepath)

    y_pred = best_model.predict([X_input_text, input_numerical_features])
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    return y_pred

#################################################### 
# Main function to classify text 
####################################################

def classify_text(text, model_path): 
    ##### this is the final one #####
    # Convert to dataframe 
    df = convert_to_dataframe(text)
    
    # Feature engineering
    df_features = generate_features(df)

    # Preprocess text
    df_features = preprocess_text2(df_features)

    # Prep input data
    X_input_text, input_numerical_features = prep_and_tokenize(df_features)

    # Get classification
    class_result = get_classification(X_input_text, input_numerical_features, model_path)

    return class_result

# model_path = "../models/classfication/best_lstm.h5"
# output1 = classify_text("Congratulations! You've won a free trip to the Bahamas! Claim your prize now by clicking on the link below.", model_path); 
# output2 = classify_text("Your account has been compromised. Please transfer $5000 to unlock your prize winnings.", model_path); 
# output3 = classify_text("The weather forecast predicts sunny skies and warm temperatures for the weekend. It's a great time to plan outdoor activities with friends and family", model_path); 
# output4 = classify_text("hi", model_path)
# output5 = classify_text("See you later", model_path)
# print("Category 1: ", output1)
# print("Category 2: ", output2)
# print("Category 3: ", output3)
# print("Category 4: ", output4)
# print("Category 5: ", output5)

# if output1 == 1:
#     print("Category 1: Positive")
