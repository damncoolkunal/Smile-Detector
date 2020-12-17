#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Running the smile detector in real time
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2



#parsing the arguments for that

ap= argparse.ArgumentParser()

ap.add_argument("-c" ,"--cascade" , required =True , help = "path to the cascade file for the faces")
ap.add_argument("-m" ,"--model" , required =True , help = "path to the model file")
#ap.add_argument("-v" ,"--video" , required =True , help = "path to the video file")

args = vars(ap.parse_args())


#load the face detector in cascade and CNN

detector = cv2.CascadeClassifier(args["cascade"])

model = load_model(args["model"])

if not args.get("video" , False):
    camera = cv2.VideoCapture(0)
    
else:
    camera = cv2.VideoCapture(args["video"])
    
    

while True:
    #grab the current frame
    
    (grabbed ,frame) = camera.read()
    
    if args.get("video") and not grabbed:
        break
        
    
    frame =imutils.resize(frame ,width=600)
    gray =cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    frameClone = frame.copy()
    
    
    #lets draw bounding boxes around it
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),
     flags=cv2.CASCADE_SCALE_IMAGE)
    
    
    for (fX, fY, fH, fW) in rects:
        
        roi = gray[fY:fY+ fH, fX: fX+ fW]
        roi = cv2.resize(roi, (28,28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        
        roi =np.expand_dims(roi , axis =0)
        
        
        (notSmiling , smiling ) = model.predict(roi)[0]
        label = "Smiling"  if smiling >= notSmiling else "Not Smiling"
 
        
        
        cv2.putText(frameClone , label ,(fX ,fY- 10), cv2.FONT_HERSHEY_COMPLEX , 0.45 ,(0,255,0) , 2)
        
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0,255,0), 2)
    
    cv2.imshow("Face" , frameClone)
    
    
    if cv2.waitKey(1) &0XFF == ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()


            
    

























# In[ ]:





# In[ ]:




