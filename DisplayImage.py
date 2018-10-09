from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore


def DisplayImage(self):
    qformat = QImage.Format_Indexed8
    if len(self.resizeImage.shape) == 3:
        if (self.resizeImage.shape[2]) == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    img = QImage(self.resizeImage, self.resizeImage.shape[1], self.resizeImage.shape[0]
                 , self.resizeImage.strides[0], qformat)

    img = img.rgbSwapped()

    self.inputImage.setPixmap(QPixmap.fromImage(img))
    self.inputImage.setAlignment(QtCore.Qt.AlignCenter)




def DisplayResultImage(self):
    qformat = QImage.Format_Indexed8
    if len(self.resultImage.shape) == 3:
        if (self.resultImage.shape[2]) == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888
    img = QImage(self.resultImage, self.resultImage.shape[1], self.resultImage.shape[0]
                 , self.resultImage.strides[0], qformat)

    img = img.rgbSwapped()

    self.outputImage.setPixmap(QPixmap.fromImage(img))
    self.outputImage.setAlignment(QtCore.Qt.AlignCenter)



class Main(super):
    def __init__(self):
        super(Main, self).__init__()
        DisplayImage(self)
