import tensorflow as tf
import numpy as np
import nltk
import tflearn
from nltk.stem.lancaster import LancasterStemmer
import json
import random
from tkit import tokenize, stem_lower, encode
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

stemmer = LancasterStemmer()

with open('Artificial_Inteligent\Chatbot\intents.json') as file_data:
    data = json.load(file_data)


words_list=[]
tags_list=[]
doc_x=[]
doc_y=[]
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word=tokenize(pattern)
        # print('\n',word)
        words_list.extend(word)
        doc_x.append(word)
        doc_y.append(intent['tag'])
    
    if (intent['tag'] not in tags_list):
        tags_list.append(intent['tag'])

character_skip=['?',',','.','!']

words_list=[stem_lower(word.lower()) for word in words_list if word not in character_skip]

words_list=sorted(set(words_list)) ### sắp xếp theo Alphabet và bỏ những từ trùng nhau 
tags_list= sorted(tags_list)


x_train=[]  
y_train=[]

data_empty=[0 for i in range (len(tags_list))]

for x, sentence in enumerate(doc_x):

    data_train=encode(sentence, words_list)

    output = data_empty[:]
    output[tags_list.index(doc_y[x])]=1

    x_train.append(data_train)     
    y_train.append(output)

x_train= np.array(x_train)
y_train=np.array(y_train)

with open("training_data", "wb") as f:
    pickle.dump({'words':words_list,'topics': tags_list,'x_train': x_train ,'y_train': y_train},f)



model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))
 
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)
model.save('D:\AI\Final_Project\model_train.h5')





