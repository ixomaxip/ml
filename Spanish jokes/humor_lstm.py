#!/usr/bin/env python
# coding: utf-8

# In[133]:


import tensorflow as tf
import sklearn, tensorflow.keras
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, GRU, SimpleRNN, Conv1D,MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer
import copy


# In[156]:


import numpy as  np
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import SnowballStemmer

word2vec = r"SBW-vectors-300-min5.txt"
train=r"haha_2019_train_preprocessed_lemmatized.csv"
test=r"haha_2019_test_preprocessed_lemmatized.csv"
val1=r"train.csv"
val2=r"test.csv"
val3=r"ev.csv"
val_ov=r"values1_oversampled.csv"
texts_ov=r"texts_train_oversampled.csv"
categories_ov=r"categories_train_oversampled.csv"
#train_sent1=r"C:\Users\Annie\Documents\Working\Spanish jokes\data\haha_2019_sent_politics_preprocessed_lemmatized.csv"
#train_sent2=r"C:\Users\Annie\Documents\Working\Spanish jokes\data\haha_2019_sent_preprocessed_lemmatized.csv"


# In[157]:


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
    #stemmed = [stemmer.stem(i) for i in tokens]
    return tokens


# In[158]:


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
categories_train_raw = [1 if str(s)!='nan' else 0 for s in scores_train]

df=pd.DataFrame(test_data)

texts_test=df[1].tolist()
texts_test_original=df[1].tolist()
scores_test=df[9].tolist()
categories_test_raw = [1 if str(s)!='nan' else 0 for s in scores_test]

#texts_train= pd.read_csv(texts_ov, sep=';', header=None, encoding = 'utf-8-sig')[0].tolist()
#categories_train_row= pd.read_csv(texts_ov, sep=';', header=None, encoding = 'utf-8-sig')[0].tolist()


# In[159]:


df=pd.read_csv(test, sep=',', header=None, encoding = 'utf-8-sig')
texts_ev=df[1].tolist()


# In[160]:


values1= pd.read_csv(val1, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
values1=scaler.fit_transform(values1)
#values1=scaler.transform(values1)


# In[161]:


values2= pd.read_csv(val2, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
values2=scaler.transform(values2)


# In[162]:


values3= pd.read_csv(val3, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
values3=scaler.transform(values3)


# In[111]:


#oversampling
count_maj=0
for c in categories_train_raw:
    if c==0:
        count_maj+=1
print('мажоритарный класс '+str(count_maj))

categories_jokes=[]
values_jokes=[]
texts_jokes=[]

for i in range(len(texts_train)):
        if categories_train_raw[i]==1:
            categories_jokes.append(categories_train_raw[i])
            values_jokes.append(values1[i])
            texts_jokes.append(texts_train[i])
        

print('миноритарный класс '+str(len(texts_jokes)))
values_jokes=np.array(values_jokes)

random.seed(42)
initial_size=len(texts_jokes)
new_size=len(texts_jokes)
while new_size<count_maj:
    r=random.randint(0,initial_size-1)
    values1=np.vstack((values1, values_jokes[r]))
    texts_train=np.append(texts_train, texts_jokes[r])
    categories_train_raw=np.append(categories_train_raw, categories_jokes[r])
    new_size+=1
    a=count_maj-new_size
    if a%500==0:
        print(a)


# In[163]:


sw_list=stopwords.words('spanish')
vectorizer = TfidfVectorizer(max_features=5000)

for i in range(len(texts_train)):
    texts_train[i]=' '.join(clean_text(texts_train[i]))
for i in range(len(texts_test)):
    texts_test[i]=' '.join(clean_text(texts_test[i]))
for i in range(len(texts_ev)):
    texts_ev[i]=' '.join(clean_text(texts_ev[i]))
    

tfidf_train = vectorizer.fit_transform(texts_train).toarray()
tfidf_test = vectorizer.transform(texts_test).toarray()
tfidf_ev = vectorizer.transform(texts_ev).toarray()


# In[113]:


import pickle

with open('categories_train_raw.dat','wb') as file:
    pickle.dump(categories_train_raw, file)
with open('texts_train.dat','wb') as file:
    pickle.dump(texts_train, file)
with open('values1.dat','wb') as file:
    pickle.dump(values1, file)


# In[155]:


with open('categories_train_raw.dat','rb') as file:
    categories_train=pickle.load(file)
with open('texts_train.dat','rb') as file:
    texts_train=pickle.load(file)
with open('values1.dat','rb') as file:
    values1=pickle.load(file)


# In[114]:


values1=np.hstack((tfidf_train,values1))
values2=np.hstack((tfidf_test,values2))
values3=np.hstack((tfidf_ev,values3))


# In[164]:


words=set()#set of all words
for text in texts_train:
    words_text=text.split();
    words.update(words_text)
for text in texts_test:
    words_text=text.split();
    words.update(words_text)
for text in texts_ev:
    words_text=text.split();
    words.update(words_text)
print("number of words: {0}".format(len(words)))


# In[165]:


embdict=dict()
index=0

with open(word2vec,'r',encoding = 'utf-8-sig')as f:
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
del(words)


# In[171]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 300

tokenizer=Tokenizer()
#tt=texts_train.tolist()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#tokenizer.fit_on_texts(tt+texts_test+texts_ev)
tokenizer.fit_on_texts(texts_train+texts_test+texts_ev)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[172]:


embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    try:
        embedding_vector = embdict[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass
        #print(i)
        #print(word)
del(embdict)


# In[173]:


texts_train = tokenizer.texts_to_sequences(texts_train)
texts_train = sequence.pad_sequences(texts_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', texts_train.shape)
texts_test = tokenizer.texts_to_sequences(texts_test)
texts_test = sequence.pad_sequences(texts_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', texts_test.shape)
texts_ev = tokenizer.texts_to_sequences(texts_ev)
texts_ev = sequence.pad_sequences(texts_ev, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', texts_ev.shape)


# In[174]:


print(len(texts_train[0]))
print(texts_train.shape)
print(values1.shape)
print(vocab_size)
print(embedding_matrix.shape)


# In[175]:


num_classes=2
categories_train = tf.keras.utils.to_categorical(categories_train_raw, num_classes)
categories_test = tf.keras.utils.to_categorical(categories_test_raw, num_classes)


# In[176]:


print(texts_train.shape)
print(texts_test.shape)
print(categories_train.shape)
print(categories_test.shape)


# In[ ]:


import tensorflow.keras.backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[183]:


inputA=Input(shape=(MAX_SEQUENCE_LENGTH,))
inputB=Input(shape=(len(values1[0]),))
inputC=Input(shape=(len(tfidf_train[0]),))


x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputA)
    # x = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional(LSTM(64, dropout=0.6, recurrent_dropout=0.2))(x)

#x = Bidirectional(LSTM(64, return_sequences=True))(x)
x=Conv1D(64,5,padding='same')(x)
x=MaxPooling1D(pool_size = (20), strides=(10))(x)
x=Conv1D(64,5,padding='same')(x)
x=MaxPooling1D(pool_size = (20), strides=(10))(x)
x=Flatten()(x)
#x = Bidirectional(LSTM(64))(x)

x = Model(inputs=inputA, outputs=x)

y = Dense(1024, activation='relu')(inputC)
y = Dropout(0.1)(y)
y = Dense(256, activation='relu')(y)
y = Dropout(0.1)(y)
y = Dense(64, activation='relu')(y)
y = Dropout(0.1)(y)
y = Model(inputs=inputC, outputs=y)


y1 = Dense(64, activation='relu')(inputB)
y1 = Dropout(0.1)(y1)
y1 = Dense(64, activation='relu')(y1)
y1 = Dropout(0.1)(y1)
y1 = Model(inputs=inputB, outputs=y1)

combined=concatenate([x.output, y.output, y1.output])
z=Dense(64, activation='relu')(combined)
z = Dropout(0.1)(z)
z=Dense(64, activation='relu')(z)
z = Dropout(0.8)(z)
z=Dense(2, activation='softmax')(z)

model = tensorflow.keras.models.Model(inputs=[inputA, inputB, inputC], outputs=z)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1],
                  )

model.summary()


# In[145]:


inputA=Input(shape=(MAX_SEQUENCE_LENGTH,))
inputB=Input(shape=(len(values1[0]),))

x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputA)
    # x = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional(LSTM(64, dropout=0.6, recurrent_dropout=0.2))(x)

#x = Bidirectional(LSTM(64, return_sequences=True))(x)
x=Conv1D(64,5,padding='same')(x)
x=MaxPooling1D(pool_size = (20), strides=(10))(x)
x=Conv1D(64,5,padding='same')(x)
x=MaxPooling1D(pool_size = (20), strides=(10))(x)
x=Flatten()(x)
x=Flatten()(x)
#x = Bidirectional(LSTM(64))(x)

x = Model(inputs=inputA, outputs=x)

y = Dense(1024, activation='relu')(inputB)
y = Dropout(0.1)(y)
y = Dense(256, activation='relu')(y)
y = Dropout(0.1)(y)
y = Dense(64, activation='relu')(y)
y = Dropout(0.1)(y)
y = Model(inputs=inputB, outputs=y)

combined=concatenate([x.output, y.output])
z=Dense(64, activation='relu')(combined)
z = Dropout(0.1)(z)
z=Dense(64, activation='relu')(z)
z = Dropout(0.8)(z)
z=Dense(2, activation='softmax')(z)

model = tensorflow.keras.models.Model(inputs=[inputA, inputB], outputs=z)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1],
                  )

model.summary()


# In[19]:


def create_model(Dy : list = [0.1, 0.1, 0.1], Dz : list = [0.1, 0.1]):
    inputA=Input(shape=(MAX_SEQUENCE_LENGTH,))
    inputB=Input(shape=(len(values1[0]),))

    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputA)
    # x = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional(LSTM(64, dropout=0.6, recurrent_dropout=0.2))(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)

    x = Model(inputs=inputA, outputs=x)

    y = Dense(1024, activation='relu')(inputB)
    y = Dropout(Dy[0])(y)
    y = Dense(256, activation='relu')(y)
    y = Dropout(Dy[1])(y)
    y = Dense(64, activation='relu')(y)
    y = Dropout(Dy[2])(y)
    y = Model(inputs=inputB, outputs=y)

    combined=concatenate([x.output, y.output])
    z=Dense(64, activation='relu')(combined)
    z = Dropout(Dz[0])(z)
    z=Dense(64, activation='relu')(z)
    z = Dropout(Dz[1])(z)
    z=Dense(2, activation='softmax')(z)

    model = tensorflow.keras.models.Model(inputs=[inputA, inputB], outputs=z)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1],
                  )

    model.summary()
    return model


# In[20]:


def create_model2(Dy = 0.0, Dz = 0.0):
    inputA=Input(shape=(MAX_SEQUENCE_LENGTH,))
    inputB=Input(shape=(len(values1[0]),))

    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputA)
    # x = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional(LSTM(64, dropout=0.6, recurrent_dropout=0.2))(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)

    x = Model(inputs=inputA, outputs=x)

    y = Dense(1024, activation='relu')(inputB)
#     y = Dropout(Dy[0])(y)
    y = Dense(256, activation='relu')(y)
#     y = Dropout(Dy[1])(y)
    y = Dense(64, activation='relu')(y)
    y = Dropout(Dy)(y)
    y = Model(inputs=inputB, outputs=y)

    combined=concatenate([x.output, y.output])
    z=Dense(64, activation='relu')(combined)
#     z = Dropout(Dz[0])(z)
    z=Dense(64, activation='relu')(z)
    z = Dropout(Dz)(z)
    z=Dense(2, activation='softmax')(z)

    model = tensorflow.keras.models.Model(inputs=[inputA, inputB], outputs=z)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1],
                  )

    model.summary()
    return model


# In[43]:


np.arange(0.1, 0.9, 0.1)


# In[21]:


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model2)

dropout = np.arange(0.1, 0.9, 0.4).tolist()
param_grid = dict(Dy=dropout, Dz=dropout)
print(param_grid)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4)


# In[23]:


grid_result = grid.fit(texts_train, np.array(values1), 
          categories_train, epochs=1, 
          verbose=0, 
          validation_data=([texts_test, np.array(values2)], categories_test)
         )
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[26]:


model = create_model([0.2, 0.5, 0.8], [0.2, 0.8])


# In[184]:


model.fit([texts_train, np.array(values1), np.array(tfidf_train)], 
          categories_train, epochs=1, 
          verbose=1, 
          validation_data=([texts_test, np.array(values2), np.array(tfidf_test)], categories_test)
         )
#           callbacks=callbacks)


# In[185]:


predict = np.argmax(model.predict([np.array(texts_test),np.array(values2), np.array(tfidf_test)]), axis=1)
answer = np.argmax(categories_test, axis=1)
print('F1-score: %f' % (f1_score(predict, answer, average="macro")*100))


# In[96]:


model


# In[187]:


predict = np.argmax(model.predict([np.array(texts_ev),np.array(values3),np.array(tfidf_ev)]), axis=1)
print(predict)
with open('prediction_cnn1.txt', 'w', encoding='utf-8') as file:
    for p in predict:
        print(str(p),file=file)


# In[91]:


from tensorflow.keras.models import load_model
model.save('lstm.h5')


# In[ ]:


from tensorflow.keras.models import load_model
model = load_model('lstm.h5')


# In[ ]:




