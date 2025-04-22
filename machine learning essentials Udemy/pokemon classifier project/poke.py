import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras_preprocessing import image
from keras.layers import Dense 
from keras import Sequential

classes =os.listdir("images/Train")
print(classes)
print("training Data:")
for f in classes:
    path="images/Train/"+f
    length=len(os.listdir(path))
    print(f"{f}-{length}")
train_data=[] #x
train_labels=[] #y
for category in classes:
    folder =f"images/Train/{category}"
    for img_name in os.listdir(folder):
        img_path=f"{folder}/{img_name}"
        
        img =image.load_img(img_path, target_size=(100,100))
         
        img =image.img_to_array(img)
        train_data.append(img)
        train_labels.append(category)
       
print(len(train_data))
len(train_labels)

train_data=np.array(train_data)
train_labels=np.array(train_labels)
train_data=train_data.reshape(len(train_data),30000)

category2label={'Pikachu':0,'Charmander':1,'Bulbasaur':2}
label2category={0:'Pikachu',1:'Charmander',2:'Bulbasaur'}
train_labels=np.array([category2label[label] for label in train_labels])
print(train_labels.shape)
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)


print(train_labels[:5])
print(train_labels[-5:])
features =train_data.shape[1]
model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(features,)))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer ="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
model.fit(train_data, train_labels, batch_size=32, epochs=50)

test_data=[]
test_labels=[]
for category in classes:
    folder =f"images/Test/{category}"
    for imag_name in os.listdir(folder):
        img_path=f"{folder}/{category}"
        img=image.load_img(img_path,target_size(100,100))
        img=image.img_to_array(img)
        test_data.append(img)
        test_labels.append(category)
test_data=np.array(test_data)
test_labels=np.array(test_labels)

test_labels=np.array([category2label[label] for label in test_labels])
test_labels=to_categorical(test_labels)
pred= model.predict(test_data).argmax(axis=1)
pred[:10]

[label2category[p] for p in pred]
