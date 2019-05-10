#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python --version')


# In[5]:


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
path_data=r"./data_2"


# In[7]:


import tensorflow as tf
h = tf.constant("GPU?")
s = tf.Session()


# In[8]:


print(s.run(h))


# In[6]:


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


# In[9]:


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
        if index%500000==0:
            print("iteration "+str(index))
print(embdict)
print("Слов в словаре:"+str(len(embdict)))


# In[10]:


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

# In[11]:


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


# In[12]:


name_train_none=r"none_train.csv"
name_train=r"m_train.csv"

name_test_none=r"none_test.csv"
name_test_irony=r"i_test.csv"
name_test_puns=r"p_test.csv"
name_test_met=r"m_test.csv"


# In[13]:


def create_data(path, data_none, flag=False):
    with open(path, encoding='utf-8') as f:
        data = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)    
    data1 = data.values
    data2 = data_none.values
    data1 = np.column_stack((np.ones(len(data1)),data1))
    data2 = np.column_stack((np.zeros(len(data2)),data2))
    data_res=np.vstack((data1,data2)) 
    if flag:
        np.random.shuffle(data_res)
    return data_res

with open(name_train_none, encoding='utf-8') as f:
        df_train_none = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)
with open(name_test_none, encoding='utf-8') as f:
        df_test_none = pd.read_csv(f, sep=';', header=0, decimal = '.', index_col=0)

train = create_data(name_train, df_train_none, True)
test_irony = create_data(name_test_irony, df_test_none)
test_puns = create_data(name_test_puns, df_test_none)
test_met = create_data(name_test_met, df_test_none)


# In[14]:


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

def split_data(d):
    df = pd.DataFrame(d)
    t = df[1].tolist()
    cat = df[0].tolist()
    vec= df.drop([0,1], axis=1).values
    t,cat,vec=remove_floats(t,cat,vec)
    return t,cat,vec

texts,categories,vectors=split_data(train)
texts_test_irony ,categories_test_irony ,vectors_test_irony =split_data(test_irony)
texts_test_puns ,categories_test_puns ,vectors_test_puns =split_data(test_puns)
texts_test_met ,categories_test_met ,vectors_test_met =split_data(test_met)


# In[15]:


num_classes = 2

descriptions = texts
    
x_train = texts
y_train = categories
    
x_test_irony = texts_test_irony
y_test_irony = categories_test_irony

x_test_puns = texts_test_puns
y_test_puns = categories_test_puns

x_test_met = texts_test_met
y_test_met = categories_test_met


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test_irony = keras.utils.to_categorical(y_test_irony, num_classes)
y_test_puns = keras.utils.to_categorical(y_test_puns, num_classes)
y_test_met = keras.utils.to_categorical(y_test_met, num_classes)

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
encoded_docs_test_irony= t.texts_to_sequences(x_test_irony)
encoded_docs_test_puns= t.texts_to_sequences(x_test_puns)
encoded_docs_test_met= t.texts_to_sequences(x_test_met)
padded_docs_train = sequence.pad_sequences(encoded_docs_train, maxlen=maxSequenceLength)
padded_docs_test_irony = sequence.pad_sequences(encoded_docs_test_irony, maxlen=maxSequenceLength)
padded_docs_test_puns = sequence.pad_sequences(encoded_docs_test_puns, maxlen=maxSequenceLength)
padded_docs_test_met = sequence.pad_sequences(encoded_docs_test_met, maxlen=maxSequenceLength)

total_unique_words = len(t.word_counts)
print('Всего уникальных слов в словаре: {}'.format(total_unique_words))


# In[16]:


embedding_matrix = np.zeros((vocab_size, 400))
for word, i in t.word_index.items():
    try:
        embedding_vector = embdict[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass
        #print(word)


# In[17]:


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


def predict(padded_docs_test, y_test, name):
    predict = np.argmax(model.predict(padded_docs_test), axis=1)
    answer = np.argmax(y_test, axis=1)
    f=open(name, 'w')
    st= 'Precision: %f' % (precision_score(predict, answer, average="macro")*100)
    print(st)
    f.write(st+'\n')
    st= 'Recall: %f' % (recall_score(predict, answer, average="macro")*100)
    print(st)
    f.write(st+'\n')
    st= 'F1-score: %f' % (f1_score(predict, answer, average="macro")*100)
    print(st)
    f.write(st+'\n')
    st= 'Accuracy: %f' % (accuracy_score(predict, answer)*100)
    print(st)
    f.write(st+'\n')

    for p in predict:
        f.write(str(p)+'\n')
    f.close()

ep=10
for i in range(7):
    history = model.fit(padded_docs_train, y_train, epochs = ep, verbose=2, validation_data=(padded_docs_test_met, y_test_met))
    predict(padded_docs_test_irony, y_test_irony, 'met_irony_'+str((i+1)*ep)+'.txt')
    predict(padded_docs_test_puns, y_test_puns, 'met_puns_'+str((i+1)*ep)+'.txt')
    predict(padded_docs_test_met, y_test_met, 'met_met_'+str((i+1)*ep)+'.txt')


# # 60 epochs:
# Precision: 77.000000<br>
# Recall: 78.957529<br>
# F1-score: 76.604618<br>
# Accuracy: 77.000000<br><br>
# Precision: 50.250000<br>
# Recall: 50.679394<br>
# F1-score: 40.914051<br>
# Accuracy: 50.250000<br><br>
# Precision: 53.250000<br>
# Recall: 57.068675<br>
# F1-score: 45.950242<br>
# Accuracy: 53.250000<br>

# In[54]:



def predict(padded_docs_test, y_test, name):
    predict = np.argmax(model.predict(padded_docs_test), axis=1)
    answer = np.argmax(y_test, axis=1)
    f=open(name, 'w')
    st= 'Precision: %f' % (precision_score(predict, answer, average="macro")*100)
    print(st)
    f.write(st+'\n')
    st= 'Recall: %f' % (recall_score(predict, answer, average="macro")*100)
    print(st)
    f.write(st+'\n')
    st= 'F1-score: %f' % (f1_score(predict, answer, average="macro")*100)
    print(st)
    f.write(st+'\n')
    st= 'Accuracy: %f' % (accuracy_score(predict, answer)*100)
    print(st)
    f.write(st+'\n')

    for p in predict:
        f.write(str(p)+'\n')
    f.close()
    

predict(padded_docs_test_irony, y_test_irony, 'irony_irony_90.txt')
predict(padded_docs_test_puns, y_test_puns, 'irony_puns_90.txt')
predict(padded_docs_test_met, y_test_met, 'irony_met_90.txt')


# In[ ]:




