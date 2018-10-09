import sqlite3
import threading

import cv2
import sys
import os
from PyQt5.QtGui import QImage, QPixmap
from qtconsole.qt import QtGui, QtCore
import keyboard
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QSplashScreen, QProgressBar
from PyQt5.uic import loadUi

import DisplayImage
import DisplayVideo
import GenderAndAge


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.ImageName = ""
        self.video = None
        self.resizeImage = None
        self.resultImage = None
        self.cap = None
        self.timer = QTimer(self)
        loadUi('GenderAndAgeGitUI.ui', self)
        self.getPersonImage.clicked.connect(self.GetImageFromDrive)
        self.computeImage.clicked.connect(self.ComputeImage)


    @pyqtSlot()
    def GetImageFromDrive(self):
        self.showResult.setText("Click Guess For" + "\n" + "Result")
        fname, filter = QFileDialog().getOpenFileName(self, 'Open File', 'c:\\', "Image Files(*.jpg)")
        if fname:
            self.LoadImage(fname)
        else:
            print("invalid Input")

    @pyqtSlot()
    def ComputeImage(self):
        self.showResult.setText("Running...")
        age_net, gender_net = GenderAndAge.initialize_caffe_models()
        font = cv2.FONT_HERSHEY_SIMPLEX

        self.resultImage = cv2.imread(self.ImageName, 1)
        self.resultImage = cv2.resize(self.resizeImage, (300,450), fx=2, fy=2, interpolation=cv2.INTER_AREA)
        cv2.threshold(self.resultImage, 127, 255, cv2.THRESH_BINARY)

        face_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_alt.xml')

        gray = cv2.cvtColor(self.resultImage, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        if (len(faces) > 0):
            print("Found {} faces".format(str(len(faces))))

        for (x, y, w, h) in faces:
            cv2.rectangle(self.resultImage, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Get Face
            face_img = self.resultImage[y:y + h + 50, x:x + w + 50].copy()

            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), GenderAndAge.MODEL_MEAN_VALUES, swapRB=False)

            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GenderAndAge.gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = GenderAndAge.age_list[age_preds[0].argmax()]
            print("Age Range: " + age)

            self.showResult.setText("Gender : " + gender + "\n" + "Age :" + age)

            overlay_text = "%s %s" % (gender,age)

            cv2.putText(self.resultImage, overlay_text, (x-10, y-10), font, .6, (255, 255, 255), 1, cv2.LINE_AA)



            DisplayImage.DisplayResultImage(self)



    def StartLiveCam(self):
        self.LoadVideo(0)

    def LoadVideo(self, vname):
        self.cap = cv2.VideoCapture(vname)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def LoadImage(self, fname):
        self.image = cv2.imread(fname)
        self.resizeImage = cv2.resize(self.image, (300, 450))
        self.ImageName = fname
        DisplayImage.DisplayImage(self)

    def update_frame(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        age_net, gender_net = GenderAndAge.initialize_caffe_models()
        ret, self.video = self.cap.read()
        face_cascade = cv2.CascadeClassifier('Model/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(self.video, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            print("Found {} faces".format(str(len(faces))))

        for (x, y, w, h) in faces:
            cv2.rectangle(self.video, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Get Face
            face_img = self.video[y:y + h + 50, x:x + w + 50].copy()

            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), GenderAndAge.MODEL_MEAN_VALUES, swapRB=False)

            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GenderAndAge.gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)

            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = GenderAndAge.age_list[age_preds[0].argmax()]
            print("Age Range: " + age)
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(self.video, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        DisplayVideo.DisplayVideo(self, self.video, 1)
        # startCapture = threading.Thread(target=GenderAndAge.read_from_camera(self), )
        # startCapture.start()


app = QApplication(sys.argv)
windows = Main()
windows.setWindowTitle('Project Name ')
windows.show()

sys.exit(app.exec_())
