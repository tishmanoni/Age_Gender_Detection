#GitHub. 2022. Gender-Detection/detect_gender_webcam.py at master · balajisrinivas/Gender-Detection. [online] Available at: < https: // github.com/balajisrinivas/Gender-Detection/blob/master/detect_gender_webcam.py > [Accessed 02 August 2022].
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os



import os
import cvlib as cv

# load model
# model = load_model('gender_detection.model')
model_gen = keras.models.load_model('new_gender_model.h5', compile = False)
model_age = keras.models.load_model('age_model.h5', compile=False)

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['male', 'female']
# age_class = ['(0-24)', '(25-49)', '(50-74)', '(75-99)', '(100-124)']

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (198, 198))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model_gen.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        # age detection on face
        max = -1
        count = 0
        message = ""
        conf_age = model_age.predict(face_crop)
        for i in conf_age[0]:
            if i > max:
                max = i
                temp = count
            count += 1

            if temp==0:
                message = '0-3 yrs old'
            if temp==1:
                message='4 -12yrs old'
            if temp==2:
                message = '13-19 yrs old'
            if temp==3:
                message= '20-30 yrs old'
            if temp==4:
                message = '31-45 yrs old'
            if temp==5:
                message = '40-60 yrs old'
            if temp==6:
                message='60+ yrs old'

        
        # idx_age = np.argmax(conf_age)
        label_age = message
        

        # conf[idx] * 100
      
        # {: .2f}%

        label = "{} : {} ".format(label, label_age)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender and age detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()