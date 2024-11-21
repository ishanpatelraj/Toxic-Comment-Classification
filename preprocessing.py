import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.iloc[:60000].reset_index(drop=True)
    df['comment_text'] = df['comment_text'].str.lower()
    df['comment_text'] = df['comment_text'].apply(remove_newline)
    df['comment_text'] = df['comment_text'].apply(remove_urls)
    df['comment_text'] = df['comment_text'].apply(remove_numbers)
    df['comment_text'] = df['comment_text'].apply(remove_chatwords)
    df['comment_text'] = df['comment_text'].apply(remove_punctuations)
    tokenizer = Tokenizer(oov_token='<oov>')
    tokenizer.fit_on_texts(df['comment_text'])
    input_sequences = tokenizer.texts_to_sequences(df['comment_text'])
    labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    return input_sequences, labels, tokenizer

def remove_newline(text):
    return text.replace('\\n', ' ')

def remove_urls(text):
    import re
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def remove_chatwords(text):
    chatwords = ['u', 'ur', 'omg', 'idk']
    return ' '.join([word for word in text.split() if word not in chatwords])

def remove_punctuations(text):
    import string
    return text.translate(str.maketrans('', '', string.punctuation))
