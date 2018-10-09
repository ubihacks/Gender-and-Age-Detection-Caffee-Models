import cv2
import imutils
import time
import numpy as np

import DisplayVideo

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(11, 13)','(14,16)','(17, 20)','(21, 23)', '(23 to 26)','(27, 30)', '(31, 33)','(34, 37)', '(38, 40)','(41, 50)', '(51, 60)', '(61, 70)', '(71, 80)', '(81, 100)']
#age_list = ['(1, 10)', '(11,15)', '(16,20)', '(21, 25)', '(26, 30)', '(31,35)', '(36, 40)', '(41,45)', '(46, 50)',
#            '(51,60)']
# age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def initialize_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe(
        'Model/deploy_age.prototxt',
        'Model/age_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(
        'Model/deploy_gender.prototxt',
        'Model/gender_net.caffemodel')

    return (age_net, gender_net)


# def read_from_camera(age_net, gender_net):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     while True:
#
#         ret, image = cap.read()
#
#         face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
#
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 5)
#
#         if (len(faces) > 0):
#             print("Found {} faces".format(str(len(faces))))
#
#         for (x, y, w, h) in faces:
#             cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
#
#             # Get Face
#             face_img = image[y:y + h, h:h + w].copy()
#             blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#
#             # Predict Gender
#             gender_net.setInput(blob)
#             gender_preds = gender_net.forward()
#
#             gender = gender_list[gender_preds[0].argmax()]
#
#             print("Gender : " + gender + gender_preds[0].argmax())
#
#             # Predict Age
#             age_net.setInput(blob)
#             age_preds = age_net.forward()
#             age = age_list[age_preds[0].argmax()]
#             print("Age Range: " + age)
#
#             overlay_text = "%s %s" % (gender, age)
#             cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#         cv2.imshow('frame', image)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break


def read_from_camera(self):
    font = cv2.FONT_HERSHEY_SIMPLEX
    age_net, gender_net = initialize_caffe_models()
    ret, self.video = self.cap.read()
    face_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(self.video, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if (len(faces) > 0):
        print("Found {} faces".format(str(len(faces))))

    for (x, y, w, h) in faces:
        cv2.rectangle(self.video, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Get Face
        face_img = self.video[y:y + h + 50, x:x + w + 50].copy()

        blob = cv2.dnn.blobFromImage(gray, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)
        overlay_text = "%s %s" % (gender, age)
        cv2.putText(self.video, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    DisplayVideo.DisplayVideo(self, self.video, 1)




def ComputeImage(self,ImageName):
    age_net, gender_net = initialize_caffe_models()
    font = cv2.FONT_HERSHEY_SIMPLEX

    image = cv2.imread(ImageName, 1)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    face_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_alt.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    if (len(faces) > 0):
        print("Found {} faces".format(str(len(faces))))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Get Face
        face_img = image[y:y+ h+50, x:x + w+50].copy()

        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)

        self.showResult.setText("Gender : "+ gender +"\n"+"Age :" + age)








