#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import numpy as  np
import os, random
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd
from nltk.stem import SnowballStemmer

model = r"SBW-vectors-300-min5.txt"
train=r"./data/haha_2019_train_preprocessed_lemmatized.csv"
test=r"./data/haha_2019_test_preprocessed_lemmatized.csv"


# In[2]:


#text preprocessing
stemmer = SnowballStemmer('spanish')

def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) 
    text = re.sub('[¡¿.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    words = text.split()
    words = [w for w in words if len(w)>=3]
    stop_words = set(stopwords.words('spanish'))
    words = [w for w in words if not w in stop_words]
    text=' '.join(words)
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(i) for i in tokens]
    return stemmed


# In[3]:


#read csv

values= pd.read_csv(train, sep=',', header=None, encoding = 'utf-8-sig').values
np.random.seed(42)
np.random.shuffle(values)
#df=pd.DataFrame(values)

m = len(values)

train_length = int(0.9 * m)
train_data, test_data = values[:train_length], values[train_length:]

df=pd.DataFrame(train_data)

texts_train=df[1].tolist()
scores_train=df[9].tolist()
categories_train=[1 if str(s)!='nan' else 0 for s in scores_train]

df=pd.DataFrame(test_data)

texts_test=df[1].tolist()
scores_test=df[9].tolist()
categories_test=[1 if str(s)!='nan' else 0 for s in scores_test]
'''
texts=df[1].tolist()
scores=df[9].tolist()

categories=[1 if str(s)!='nan' else 0 for s in scores]

df = pd.read_csv(test, sep=',', header=None)
texts_test=df[1].tolist()
'''


# In[4]:


words=set()#set of all words
for text in texts_train:
    words_text=clean_text(text)
    words.update(words_text)
print("number of words: {0}".format(len(words)))


# In[5]:


embdict=dict()#dictionary for words and emb
index=0

with open(model,'r',encoding = 'utf-8-sig')as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size):
        word=str(f.readline()).replace('b','').replace('\'','').replace('\\n','').lower().split()
        w = stemmer.stem(word[0])
        if w in words:
            word.remove(word[0])
            emb = [float(x) for x in word]
            embdict[str(w)]=emb
        index+=1
        if index%100000==0:
            print("iteration "+str(index))

print("size of dictionary: {0}".format(len(embdict)))


# <h1>сеть

# In[10]:


import tensorflow.keras as keras
import sklearn
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, GRU, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import copy


# In[17]:


num_classes = 2

prep_texts_train =[]
prep_texts_test =[]

print("предобработка текста")
for t in texts_train:
    prep_texts_train.append(' '.join(clean_text(t)))
for t in texts_test:
    prep_texts_test.append(' '.join(clean_text(t)))
    
descriptions = prep_texts_train
    
x_train = prep_texts_train
y_train = categories_train
    
x_test = prep_texts_test
y_test = categories_test


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

max_words = 0
for desc in descriptions:
    try:
        words = len(desc)
        if words > max_words:
            max_words = words
    except:
        pass
print('Максимальное количество слов в самом длинном тексте: {} слов'.format(max_words))

maxSequenceLength = max_words

t = Tokenizer()
    
t.fit_on_texts(descriptions)
vocab_size = len(t.word_index) + 1

encoded_docs_train = t.texts_to_sequences(x_train)
encoded_docs_test = t.texts_to_sequences(x_test)
padded_docs_train = sequence.pad_sequences(encoded_docs_train, maxlen=maxSequenceLength)
padded_docs_test = sequence.pad_sequences(encoded_docs_test, maxlen=maxSequenceLength)

total_unique_words = len(t.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))


# In[18]:


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in t.word_index.items():
    try:
        embedding_vector = embdict[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass
        #print(i)
        #print(word)


# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, GRU, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxSequenceLength, trainable=False))
#model.add(e)
#e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxSequenceLength, trainable=False)
#model.add(e)
#model.add(Flatten())
#model.add(Dense(20, activation='sigmoid'))
#model.add(Dropout=0.5)
#model.add(Embedding(300, maxSequenceLength))
#model.add(Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)))
#model.add(Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.2)))
model.add(Dense(num_classes, activation='softmax'))
# compile the model
#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#model.compile(optimizer = rmsprop, loss = 'mean_squared_error', metrics=['mean_squared_error', 'mae'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())


# In[23]:


history = model.fit(padded_docs_train, y_train, epochs = 20, verbose=2, validation_data=(padded_docs_test, y_test))


# In[ ]:


predict = np.argmax(model.predict(x_test), axis=1)
answer = np.argmax(y_test, axis=1)
print('Accuracy: %f' % (accuracy_score(predict, answer)*100))
print('F1-score: %f' % (f1_score(predict, answer, average="macro")*100))
print('Precision: %f' % (precision_score(predict, answer, average="macro")*100))
print('Recall: %f' % (recall_score(predict, answer, average="macro")*100))  

