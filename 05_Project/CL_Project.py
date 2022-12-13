import numpy as np
import pandas
from keras.preprocessing.text import Tokenizer
from collections import Counter
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense , Conv1D, GlobalAveragePooling2D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# Helper Functions
def counter_word(text):
    count = Counter()
    for sentence in text:
        for word in sentence.split():
            count[word] += 1
    return count
def remove_punct(sentence):
  temp = str.maketrans("", "", string.punctuation)
  return sentence.translate(temp)

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_html(text):
    html = re.compile(r"<.*?>")
    return html.sub(r"", text)

def remove_emoji(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", string)


# fetch comments and labels from dataset
train = pandas.read_csv('dataset.csv')
train = train.dropna().reset_index(drop=True)
train = np.asarray(train)
x_set = []
y_set = []

for data in train:
  x_set.append(data[0])
  y_set.append(data[1:].astype(np.float32))


comments_raw = np.asarray(x_set)
labels = np.asarray(y_set)

# Extract Indonesian stop words
stop = open('stop_words_indonesia.txt').readlines()

# Perform basic pre-processing steps
comments = []
for comment in comments_raw:
  temp = remove_punct(comment)
  temp = remove_URL(temp)
  temp = remove_html(temp)
  temp = remove_emoji(temp)
  new = []
  for t in temp.split(" "):
    if t not in stop:
      new.append(t)
      new = [n for n in new if n not in ['USER','RT']] # some unnecesarry characters are present in comments which are removed
  comments.append(" ".join(new))

# get vocabulary count and perform tokenization and padding
counter = counter_word(comments)
max_length = 104
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
padded = pad_sequences(
    sequences, maxlen=max_length, padding="post", truncating="post"
)

# Get count vector for bag of words representation
vectorizer = CountVectorizer(max_features = 2000)
X = vectorizer.fit_transform(comments)
bow = X.toarray()

# get train and test data for word embedding representation

x_tr_e, x_te_e, y_tr_e, y_te_e = train_test_split(padded,labels , 
                                   test_size=0.1, 
                                   random_state=104,
                                   shuffle=True)

# get train and test data for bag of words representation
x_tr_b, x_te_b, y_tr_b, y_te_b = train_test_split(bow,labels , 
                                   test_size=0.1, 
                                   random_state=104,
                                   shuffle=True)

# convert train and test data to tensors
x_train_e = tf.convert_to_tensor(x_tr_e)
y_train_e = tf.convert_to_tensor(y_tr_e)
x_test_e = tf.convert_to_tensor(x_te_e)
y_test_e = tf.convert_to_tensor(y_te_e)

x_train_b = tf.convert_to_tensor(x_tr_b)
y_train_b = tf.convert_to_tensor(y_tr_b)
x_test_b = tf.convert_to_tensor(x_te_b)
y_test_b = tf.convert_to_tensor(y_te_b)

# Calculate parameter for the embedding layer
embedding_param = len(tokenizer.word_index.items()) + 1

#LSTM Model
model_lstm = Sequential()
model_lstm.add(Embedding(embedding_param,32,input_length=max_length))
model_lstm.add(LSTM(128,dropout=0.1,return_sequences = True))
model_lstm.add(LSTM(64,dropout=0.1,return_sequences = True))
model_lstm.add(LSTM(32,dropout=0.1))
model_lstm.add(Dense(16, activation="relu"))
model_lstm.add(Dense(12, activation="sigmoid"))

model_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model_lstm.fit( x_train_e, y_train_e, epochs= 5 , validation_data=(x_test_e, y_test_e),)

#CNN Model
model_cnn = Sequential()
model_cnn.add(Embedding(embedding_param,32,input_length=max_length))
model_cnn.add(Conv1D(128,3,padding='same'))
model_cnn.add(MaxPooling1D(2))
model_cnn.add(Conv1D(64,3,padding='same'))
model_cnn.add(MaxPooling1D(2))
model_cnn.add(Conv1D(32,3,padding='same'))
model_cnn.add(MaxPooling1D(2))
model_cnn.add(Flatten())
model_cnn.add(Dense(16,activation='relu'))
model_cnn.add(Dense(12,activation='sigmoid'))

model_cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history1 = model_cnn.fit(x_train_e, y_train_e, epochs= 4 , validation_data=(x_test_e, y_test_e))

# MLP Model for word embedding representation
model_ann = Sequential()
model_ann.add(Embedding(embedding_param,32,input_length=max_length))
model_ann.add(Dense(64))
model_ann.add(Dense(32))
model_ann.add(Flatten())
model_ann.add(Dense(16,activation="relu"))
model_ann.add(Dense(12,activation="sigmoid"))

model_ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history2 = model_ann.fit(x_train_e, y_train_e, batch_size = 100, epochs= 4 , validation_data=(x_test_e, y_test_e))

#MLP Model for Bag of Words Representation
model_ann1 = Sequential()
model_ann1.add(Dense(64))
model_ann1.add(Dense(32))
model_ann1.add(Flatten())
model_ann1.add(Dense(16,activation="relu"))
model_ann1.add(Dense(12,activation="sigmoid"))

model_ann1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history21 = model_ann1.fit(x_train_b, y_train_b,batch_size = 100, epochs= 4 , validation_data=(x_test_b, y_test_b))

# Get model summaries
model_lstm.summary()
model_cnn.summary()
model_ann.summary()
model_ann1.summary()

# save models
model_lstm.save('model_lstm.h5')
model_cnn.save('model_cnn1.h5')
model_ann.save('model_ann.h5')
model_ann1.save('model_ann2.h5')

# test on test set
new_model1 = tf.keras.models.load_model('model_lstm.h5')
results = new_model1.evaluate(x_test_e, y_test_e, batch_size=128)

new_model2 = tf.keras.models.load_model('model_cnn1.h5')
results = new_model2.evaluate(x_test_e, y_test_e, batch_size=128)

new_model3 = tf.keras.models.load_model('model_ann.h5')
results = new_model3.evaluate(x_test_e, y_test_e, batch_size=128)

new_model4 = tf.keras.models.load_model('model_ann2.h5')
results = new_model4.evaluate(x_test_b, y_test_b, batch_size=128)


