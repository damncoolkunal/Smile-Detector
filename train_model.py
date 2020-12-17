#!/usr/bin/env python
# coding: utf-8

# In[4]:


#smile detector
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os


#creating Argument Parser

ap = argparse.ArgumentParser()
ap.add_argument("-d" ,"--dataset" , required =True , help = "path to the dataset file")
ap.add_argument("-m", "--model" , required =True , help = "path to the model file")
args = vars(ap.parse_args())

#inputing data and labels

data =[]
labels =[]

#loop over the input images

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image =cv2.imread(imagePath)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width= 28)
    image =img_to_array(image)
    data.append(image)
    
    #extract the class labels
    
    label = imagePath.split(os.path.sep)[-3]
    label ="smiling" if label =="positives" else "not_smiling"
    labels.append(label)
    
    
data =np.array(data , dtype ="float")/255.0
labels = np.array(labels)


#convert the labels from integers to vectors

le = LabelEncoder().fit(labels)
labels =np_utils.to_categorical(le.transform(labels),2)


classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals



(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


#start training LENET

print("info initializing....")

model =LeNet.build(width =28 , height =28 , depth =1, classes =2)
model.compile(loss ="binary_crossentropy" , optimizer ="adam" , metrics =["accuracy"])

#train the network

print("INFO start training network")

H = model.fit(trainX, trainY, validation_data = (testX, testY),
    class_weight = classWeight, batch_size = 64, epochs = 15, verbose = 1)


#evaluate the network

print("evaluating the network....")

predictions = model.predict(testX , batch_size =64)

print (classification_report(testY.argmax(axis =1), predictions.argmax(axis =1) , target_names= le.classes_))

#save the model to disk

print("INFO serializing the network...")

model.save(args["model"])

#plotting the training curves

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])


    




















  


# In[ ]:





# In[ ]:




