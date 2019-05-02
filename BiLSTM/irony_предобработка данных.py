#!/usr/bin/env python
# coding: utf-8

# In[2]:


from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
import numpy as  np
import os
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd

path_model = r"./word2vec_twitter_model.bin"
path_data=r"./data"


# In[3]:


def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) 
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    words = text.split()
    words = [w for w in words if len(w)>3]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    text=' '.join(words)
    tokens = word_tokenize(text)
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    return stemmed
    
words=set()#множество слов по всем текстам

for f in os.listdir(path_data):
    print(f)
    full_path = path_data+'/'+f
    try:
        f_text = open(full_path, "r").readlines()
        for line in f_text:
            words_text=clean_text(line)
            words.update(words_text)
    except Exception as  err:
        print(err)
        
        pass
        '''
        lines=[]
        with open(full_path) as file:
            while True:
                try:
                    line = file.readline()
                    lines.append(line)
                except:
                    pass
        for line in lines:
            words_text=clean_text(line)
            words.update(words_text)
        '''
    
print("Всего слов: {}".format(len(words)))


# In[3]:


embdict=dict()#словарь эмбеддингов и слов
index=0
porter = PorterStemmer()

with open(path_model,'rb')as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size):
        word = []
        while True:
            ch = f.read(1).decode(errors='ignore')
            if ch ==' ':
                word = ''.join(word)
                break
            if ch != '\n':
                word.append(ch)
        if len(word) != 0:
            tp= np.fromstring(f.read(binary_len), dtype='float32')
            word = porter.stem(word.lower())
            if word in words:
                embdict[str(word)]=tp.tolist()

        else:
            f.read(binary_len)
        index+=1
        if index%1000000==0:
            print("iteration "+str(index))
print(embdict)
print("Слов в словаре:"+str(len(embdict)))


# In[4]:


for f in os.listdir(path_data):
    print(f)
    full_path = path_data+'/'+f
    try:
        f_text = open(full_path, "r").readlines()
        vectors=[]
        lines=[]
        for line in f_text:
            words_text=clean_text(line)
            emb_vector=np.zeros(400)
            for word in words_text:
                try:
                    emb_vector+=embdict[word]
                except:
                    pass
            lines.append(line)
            vectors.append(emb_vector)
        df=pd.DataFrame(vectors)
        df.insert(loc=0, column='texts', value=lines)
        df.to_csv(f+'.csv',sep=';')
    except:
        pass
    
print("Всего слов: "+str(len(words)))


# <h1>дальше код про сеть

# In[5]:


import numpy as np
import tensorflow.keras as keras
import pandas as pd
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
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import copy


# In[6]:


name_train1=r"none-class-1200.txt.csv"
name_train2=r"i_train.txt.csv"
name_test=r"i_test.txt.csv"
name_val=r"p_val.txt.csv"


# In[7]:


with open(name_train1, encoding='utf-8') as f:
        df1 = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)
with open(name_train2, encoding='utf-8') as f:
        df2 = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)       

train_data1 = df1.values
train_data2 = df2.values

train_data1 = np.column_stack((np.zeros(len(train_data1)),train_data1))
train_data2 = np.column_stack((np.ones(len(train_data2)),train_data2))

train_data=np.vstack((train_data1,train_data2)) 
np.random.shuffle(train_data)

with open(name_test, encoding='utf-8') as f:
        df1 = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)
with open(name_val, encoding='utf-8') as f:
        df2 = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)
test_data = df1.values
test_data = np.column_stack((np.ones(len(test_data)),test_data))
val_data = df2.values
val_data = np.column_stack((np.ones(len(val_data)),val_data))


# In[8]:


def remove_floats(texts, categories, vectors):
    _texts=[]
    _categories=[]
    _vectors=[]
    for i in range(len(texts)):
        if type(texts[i]) is str:
            _texts.append(clean_text(texts[i]))
            _categories.append(categories[i])
            _vectors.append(vectors[i])
    return _texts,_categories,_vectors

df = pd.DataFrame(train_data)
texts = df[1].tolist()
categories = df[0].tolist()
vectors = df.drop([0,1], axis=1).values
texts,categories,vectors=remove_floats(texts,categories,vectors)

df = pd.DataFrame(test_data)
texts_test = df[1].tolist()
categories_test = df[0].tolist()
vectors_test = df.drop([0,1], axis=1).values
texts_test,categories_test,vectors_test=remove_floats(texts_test,categories_test,vectors_test)

df = pd.DataFrame(val_data)
texts_val = df[1].tolist()
categories_val = df[0].tolist()
vectors_val = df.drop([0,1], axis=1).values
texts_val,categories_val,vectors_val=remove_floats(texts_val,categories_val,vectors_val)


# In[12]:


num_classes = 2

descriptions = texts+texts_test
    
x_train = texts
y_train = categories
    
x_test = texts_test
y_test = categories_test

x_val = texts_val
y_val = categories_val


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

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
encoded_docs_val = t.texts_to_sequences(x_val)
padded_docs_train = sequence.pad_sequences(encoded_docs_train, maxlen=maxSequenceLength)
padded_docs_test = sequence.pad_sequences(encoded_docs_test, maxlen=maxSequenceLength)
padded_docs_val= sequence.pad_sequences(encoded_docs_val, maxlen=maxSequenceLength)

total_unique_words = len(t.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))


# In[10]:


embedding_matrix = np.zeros((vocab_size, 400))
for word, i in t.word_index.items():
    try:
        embedding_vector = embdict[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        print(word)


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, GRU, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Embedding(vocab_size, 400, weights=[embedding_matrix], input_length=maxSequenceLength, trainable=False))
#model.add(e)
#e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxSequenceLength, trainable=False)
#model.add(e)
#model.add(Flatten())
#model.add(Dense(200, activation='sigmoid'))
#model.add(Dropout=0.5)
#model.add(Embedding(300, maxSequenceLength))
model.add(Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.2)))
model.add(Dense(num_classes, activation='softmax'))
# compile the model
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#model.compile(optimizer = rmsprop, loss = 'mean_squared_error', metrics=['mean_squared_error', 'mae'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())


# In[18]:


history = model.fit(padded_docs_train, y_train, epochs = 10, verbose=2, validation_data=(padded_docs_test, y_test))
predict = np.argmax(model.predict(padded_docs_test), axis=1)
answer = np.argmax(y_test, axis=1)
print('Accuracy: %f' % (accuracy_score(predict, answer)*100))
print('F1-score: %f' % (f1_score(predict, answer, average="macro")*100))
print('Precision: %f' % (precision_score(predict, answer, average="macro")*100))
print('Recall: %f' % (recall_score(predict, answer, average="macro")*100)) 


# # 10 epochs:
# - ## test:
# Accuracy: 87.000000<br>
# F1-score: 46.524064<br>
# Precision: 43.500000<br>
# Recall: 50.000000<br>
# 
# - ## val:
# Accuracy: 35.000000<br>
# F1-score: 25.925926<br>
# Precision: 17.500000<br>
# Recall: 50.000000<br>
# 
# # 20 epochs:
# - ## test:
# Accuracy: 90.000000<br>
# F1-score: 47.368421<br>
# Precision: 45.000000<br>
# Recall: 50.000000<br>
# 
# - ## val:
# Accuracy: 28.000000<br>
# F1-score: 21.875000<br>
# Precision: 14.000000<br>
# Recall: 50.000000<br>

# In[19]:


predict = np.argmax(model.predict(padded_docs_val), axis=1)
answer = np.argmax(y_val, axis=1)
print('Accuracy: %f' % (accuracy_score(predict, answer)*100))
print('F1-score: %f' % (f1_score(predict, answer, average="macro")*100))
print('Precision: %f' % (precision_score(predict, answer, average="macro")*100))
print('Recall: %f' % (recall_score(predict, answer, average="macro")*100))  


# In[ ]:




