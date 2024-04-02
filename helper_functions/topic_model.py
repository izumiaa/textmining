import os
import re
import ast
import pandas as pd
import numpy as np
import tqdm
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from gensim.models import Phrases
from gensim import corpora
from gensim.models import CoherenceModel

from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from gensim.models.nmf import Nmf


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import joblib


def save_data_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def preprocess_text(text):

    # Define custom stopwords
    custom_stopwords = set(stopwords.words('english'))
    # custom_stopwords = set(stopwords.words('english')) | {'enron'}

    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    if not isinstance(text, str):
        return ""
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize into words, remove stopwords, lemmatize, and filter out short words
    tokens = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and len(word) > 1]
    return filtered_words

def get_dict(path): 
    # Read the CSV file
    df = pd.read_csv(path)

    # Preprocess text data
    df['tokens'] = df['text'].apply(preprocess_text)

    # Separate spam and ham text - Unigram
    uni_spam_df = df[df['label'] == 'spam']
    uni_ham_df = df[df['label'] == 'ham']

    # Create dictionary and corpus for spam text - Unigram
    uni_spam_dictionary = corpora.Dictionary(uni_spam_df['tokens'])
    uni_spam_corpus = [uni_spam_dictionary.doc2bow(text) for text in uni_spam_df['tokens']]

    # Create dictionary and corpus for ham text - Unigram
    uni_ham_dictionary = corpora.Dictionary(uni_ham_df['tokens'])
    uni_ham_corpus = [uni_ham_dictionary.doc2bow(text) for text in uni_ham_df['tokens']]

    # Text Col
    uni_col = "tokens"

    return uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary

# Build NMF model for spam text
def spamNMFModel(spamDict, hamDict, spamCorp, hamCorp, spamIdDict, hamIdDict, textCol, numTopics):
    
    spam_nmf_model = Nmf(
        corpus=spamCorp, 
        num_topics=numTopics, 
        id2word=spamIdDict,
        chunksize=100,
        passes=2,
        random_state=42
    )

    # Build LDA model for ham text
    ham_nmf_model = Nmf(corpus=hamCorp, num_topics=numTopics, id2word=hamIdDict)
        
    # Calculate coherence score for spam topics
    spam_coherence_model = CoherenceModel(model=spam_nmf_model, texts=spamDict[textCol], dictionary=spamIdDict, coherence='c_v')
    spam_coherence_score = spam_coherence_model.get_coherence()
#     print("Coherence score for spam topics:", spam_coherence_score)

    # Calculate coherence score for ham topics
    ham_coherence_model = CoherenceModel(model=ham_nmf_model, texts=hamDict[textCol], dictionary=hamIdDict, coherence='c_v')
    ham_coherence_score = ham_coherence_model.get_coherence()
#     print("Coherence score for ham topics:", ham_coherence_score)
    
    return spam_nmf_model, ham_nmf_model, spam_coherence_score, ham_coherence_score

# def get_gensim_nmf_model(uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary): 

#     # Gensim NMF Model - Uni - Spam
#     uni_spam_nmf_num_topics1 = 7
#     uni_spam_nmf_model1, uni_ham_nmf_model1, uni_spam_nmf_coher1, uni_ham_nmf_coher1 = spamNMFModel(uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary, "tokens", uni_spam_nmf_num_topics1)

#     new_topics=[]

#     for topic_id, topic in uni_spam_nmf_model1.print_topics(num_topics=uni_spam_nmf_num_topics1, num_words=20):
#         # print(f'Topic {topic_id}: {topic}')
#         indiv_topic_dict = {}
#         indiv_topic = topic.split(" + ")
#         for ind in indiv_topic:
#             sep = ind.split("*")
#             word = sep[1][1:-1]
#             val = float(sep[0])
#             indiv_topic_dict[word] = val
#         new_topics.append(indiv_topic_dict)

#     print("Coherence score for spam topic model:", uni_spam_nmf_coher1)
#     return uni_spam_nmf_model1, new_topics



def get_topic(path, input_text, load_saved_data=True): 
    if load_saved_data:
        # Load previously saved data
        uni_spam_df = pd.DataFrame.from_dict(load_data_from_json('models/topic_model/uni_spam_df.json'))
        uni_ham_df = pd.DataFrame.from_dict(load_data_from_json('models/topic_model/uni_ham_df.json'))
        uni_spam_corpus = load_data_from_json('models/topic_model/uni_spam_corpus.json')
        uni_ham_corpus = load_data_from_json('models/topic_model/uni_ham_corpus.json')
        uni_spam_dictionary = corpora.Dictionary.load('models/topic_model/uni_spam_dictionary.pkl')
        uni_ham_dictionary = corpora.Dictionary.load('models/topic_model/uni_ham_dictionary.pkl')
        uni_spam_nmf_model1 = joblib.load('models/topic_model/uni_spam_nmf_model1.pkl')
        uni_ham_nmf_model1 = joblib.load('models/topic_model/uni_ham_nmf_model1.pkl')
        uni_spam_nmf_coher1 = joblib.load('models/topic_model/uni_spam_nmf_coher1.pkl')
        uni_ham_nmf_coher1 = joblib.load('models/topic_model/uni_ham_nmf_coher1.pkl')
    else:
        print('else block was printed.')
        uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary = get_dict(path) 
        
        # Save the data to JSON files
        save_data_to_json(uni_spam_df.to_dict(), 'models/topic_model/uni_spam_df.json')
        save_data_to_json(uni_ham_df.to_dict(), 'models/topic_model/uni_ham_df.json')
        save_data_to_json(uni_spam_corpus, 'models/topic_model/uni_spam_corpus.json')
        save_data_to_json(uni_ham_corpus, 'models/topic_model/uni_ham_corpus.json')
        
        # Define the number of topics for NMF
        uni_spam_nmf_num_topics1 = 7

        # Call the function to generate NMF models and coherence values
        uni_spam_nmf_model1, uni_ham_nmf_model1, uni_spam_nmf_coher1, uni_ham_nmf_coher1 = spamNMFModel(uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary, "tokens", uni_spam_nmf_num_topics1)

        # Save the NMF models and coherence values
        joblib.dump(uni_spam_nmf_model1, 'models/topic_model/uni_spam_nmf_model1.pkl')
        joblib.dump(uni_ham_nmf_model1, 'models/topic_model/uni_ham_nmf_model1.pkl')
        joblib.dump(uni_spam_nmf_coher1, 'models/topic_model/uni_spam_nmf_coher1.pkl')
        joblib.dump(uni_ham_nmf_coher1, 'models/topic_model/uni_ham_nmf_coher1.pkl')
        
        # Save the dictionaries only if load_saved_data is False
        if not load_saved_data:
            uni_spam_dictionary.save('models/topic_model/uni_spam_dictionary.pkl')
            uni_ham_dictionary.save('models/topic_model/uni_ham_dictionary.pkl')

    # Rest of your code...
    inputText = preprocess_text(input_text)
    inputTextDf = pd.DataFrame({"text": inputText})

    min_topics = 2
    max_topics = 11
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    topic_labels_spam = [
        "HTML/Website Spams",
        "Email/Phishing Scams",
        "Investment/Financial Scams",
        "Foreign Language Spam", # was spanish spam
        "Housing/Urgent Account Scams",
        "Travel/Promotion Advertising Scams",
        "Company Information Scams",
    ]

    # Gensim NMF Model - Uni - Spam
    uni_spam_nmf_num_topics1 = 7

    new_topics=[]

    for topic_id, topic in uni_spam_nmf_model1.print_topics(num_topics=uni_spam_nmf_num_topics1, num_words=20):
        # print(f'Topic {topic_id}: {topic}')
        indiv_topic_dict = {}
        indiv_topic = topic.split(" + ")
        for ind in indiv_topic:
            sep = ind.split("*")
            word = sep[1][1:-1]
            val = float(sep[0])
            indiv_topic_dict[word] = val
        new_topics.append(indiv_topic_dict)

    print("Coherence score for spam topic model:", uni_spam_nmf_coher1)

    new_text_corpus =  uni_spam_dictionary.doc2bow(inputTextDf['text'][0].split())
    # predicted topic distribution
    pred_topic_distri = uni_spam_nmf_model1[new_text_corpus]

    max_coher = max(pred_topic_distri, key=lambda x: x[1])
    topic_chosen_id = max_coher[0]
    topic_coher_score = max_coher[1]
    topic_chosen_dict = new_topics[topic_chosen_id]
    topic_label = topic_labels_spam[topic_chosen_id]

    return topic_chosen_dict, topic_label, topic_coher_score




# Main function
# the below functions should be called in the main function
# rmb to add generate_wordcloud_per_topic to the main function 

def generate_wordcloud_per_topic(topic, topic_label):
    # Create a figure with multiple subplots, one for each topic
    fig, axes = plt.subplots(figsize=(10, 5))

    # Generate a word cloud for the current topic
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(topic)

    # Plot the word cloud for the current topic
    axes.imshow(wordcloud, interpolation='bilinear')
    axes.set_title(topic_label)
    axes.axis('off')
    
    plt.tight_layout()
    plt.show()




# import os
# import re
# import ast
# import pandas as pd
# import numpy as np
# import tqdm
# import gensim
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import sent_tokenize

# from gensim.models import Phrases
# from gensim import corpora
# from gensim.models import CoherenceModel

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt 
# from gensim.models.nmf import Nmf


# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# def preprocess_text(text):

#     # Define custom stopwords
#     custom_stopwords = set(stopwords.words('english'))
#     # custom_stopwords = set(stopwords.words('english')) | {'enron'}

#     # Initialize WordNet lemmatizer
#     lemmatizer = WordNetLemmatizer()

#     if not isinstance(text, str):
#         return ""
#     # Remove non-alphabetical characters
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     # Tokenize into words, remove stopwords, lemmatize, and filter out short words
#     tokens = word_tokenize(text.lower())
#     filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and len(word) > 1]
#     return filtered_words

# def get_dict(path): 
#     # Read the CSV file
#     df = pd.read_csv(path)

#     # Preprocess text data
#     df['tokens'] = df['text'].apply(preprocess_text)

#     # Separate spam and ham text - Unigram
#     uni_spam_df = df[df['label'] == 'spam']
#     uni_ham_df = df[df['label'] == 'ham']

#     # Create dictionary and corpus for spam text - Unigram
#     uni_spam_dictionary = corpora.Dictionary(uni_spam_df['tokens'])
#     uni_spam_corpus = [uni_spam_dictionary.doc2bow(text) for text in uni_spam_df['tokens']]

#     # Create dictionary and corpus for ham text - Unigram
#     uni_ham_dictionary = corpora.Dictionary(uni_ham_df['tokens'])
#     uni_ham_corpus = [uni_ham_dictionary.doc2bow(text) for text in uni_ham_df['tokens']]

#     # Text Col
#     uni_col = "tokens"

#     return uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary

# # Build NMF model for spam text
# def spamNMFModel(spamDict, hamDict, spamCorp, hamCorp, spamIdDict, hamIdDict, textCol, numTopics):
    
#     spam_nmf_model = Nmf(
#         corpus=spamCorp, 
#         num_topics=numTopics, 
#         id2word=spamIdDict,
#         chunksize=100,
#         passes=2,
#         random_state=42
#     )

#     # Build LDA model for ham text
#     ham_nmf_model = Nmf(corpus=hamCorp, num_topics=numTopics, id2word=hamIdDict)
        
#     # Calculate coherence score for spam topics
#     spam_coherence_model = CoherenceModel(model=spam_nmf_model, texts=spamDict[textCol], dictionary=spamIdDict, coherence='c_v')
#     spam_coherence_score = spam_coherence_model.get_coherence()
# #     print("Coherence score for spam topics:", spam_coherence_score)

#     # Calculate coherence score for ham topics
#     ham_coherence_model = CoherenceModel(model=ham_nmf_model, texts=hamDict[textCol], dictionary=hamIdDict, coherence='c_v')
#     ham_coherence_score = ham_coherence_model.get_coherence()
# #     print("Coherence score for ham topics:", ham_coherence_score)
    
#     return spam_nmf_model, ham_nmf_model, spam_coherence_score, ham_coherence_score

# # def get_gensim_nmf_model(uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary): 

# #     # Gensim NMF Model - Uni - Spam
# #     uni_spam_nmf_num_topics1 = 7
# #     uni_spam_nmf_model1, uni_ham_nmf_model1, uni_spam_nmf_coher1, uni_ham_nmf_coher1 = spamNMFModel(uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary, "tokens", uni_spam_nmf_num_topics1)

# #     new_topics=[]

# #     for topic_id, topic in uni_spam_nmf_model1.print_topics(num_topics=uni_spam_nmf_num_topics1, num_words=20):
# #         # print(f'Topic {topic_id}: {topic}')
# #         indiv_topic_dict = {}
# #         indiv_topic = topic.split(" + ")
# #         for ind in indiv_topic:
# #             sep = ind.split("*")
# #             word = sep[1][1:-1]
# #             val = float(sep[0])
# #             indiv_topic_dict[word] = val
# #         new_topics.append(indiv_topic_dict)

# #     print("Coherence score for spam topic model:", uni_spam_nmf_coher1)
# #     return uni_spam_nmf_model1, new_topics



# def get_topic(path, input_text): 
#     inputText = preprocess_text(input_text)
#     uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary = get_dict(path) 
#     inputTextDf = pd.DataFrame({"text": inputText})

#     min_topics = 2
#     max_topics = 11
#     step_size = 1
#     topics_range = range(min_topics, max_topics, step_size)

#     topic_labels_spam = [
#         "HTML/Website Spams",
#         "Email/Phishing Scams",
#         "Investment/Financial Scams",
#         "Foreign Language Spam", # was spanish spam
#         "Housing/Urgent Account Scams",
#         "Travel/Promotion Advertising Scams",
#         "Company Information Scams",
#     ]

#     # Gensim NMF Model - Uni - Spam
#     uni_spam_nmf_num_topics1 = 7
#     uni_spam_nmf_model1, uni_ham_nmf_model1, uni_spam_nmf_coher1, uni_ham_nmf_coher1 = spamNMFModel(uni_spam_df, uni_ham_df, uni_spam_corpus, uni_ham_corpus, uni_spam_dictionary, uni_ham_dictionary, "tokens", uni_spam_nmf_num_topics1)

#     new_topics=[]

#     for topic_id, topic in uni_spam_nmf_model1.print_topics(num_topics=uni_spam_nmf_num_topics1, num_words=20):
#         # print(f'Topic {topic_id}: {topic}')
#         indiv_topic_dict = {}
#         indiv_topic = topic.split(" + ")
#         for ind in indiv_topic:
#             sep = ind.split("*")
#             word = sep[1][1:-1]
#             val = float(sep[0])
#             indiv_topic_dict[word] = val
#         new_topics.append(indiv_topic_dict)

#     print("Coherence score for spam topic model:", uni_spam_nmf_coher1)

#     new_text_corpus =  uni_spam_dictionary.doc2bow(inputTextDf['text'][0].split())
#     # predicted topic distribution
#     pred_topic_distri = uni_spam_nmf_model1[new_text_corpus]

#     max_coher = max(pred_topic_distri, key=lambda x: x[1])
#     topic_chosen_id = max_coher[0]
#     topic_coher_score = max_coher[1]
#     topic_chosen_dict = new_topics[topic_chosen_id]
#     topic_label = topic_labels_spam[topic_chosen_id]

#     return topic_chosen_dict, topic_label, topic_coher_score

# def generate_wordcloud_per_topic(topic, topic_label):
#     # Create a figure with multiple subplots, one for each topic
#     fig, axes = plt.subplots(figsize=(10, 5))

#     # Generate a word cloud for the current topic
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(topic)

#     # Plot the word cloud for the current topic
#     axes.imshow(wordcloud, interpolation='bilinear')
#     axes.set_title(topic_label)
#     axes.axis('off')
    
#     plt.tight_layout()
#     plt.show()