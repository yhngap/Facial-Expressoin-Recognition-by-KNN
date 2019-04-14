import numpy as np
from numpy.random import randn
import cv2 as cv
import os

def capture_video():

    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow("video", frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


class Nearest_Neighbor(object):

    def __init__(self):
        print("created!")
        self.x, self.y = randn(1, 72*128*3), randn(1, 3)
        self.w1, self.w2 = randn(72*128*3, 100), randn(100,3)
        print("success!")


    def Predict(self,X):

        path="C:\\Users\\yauch\\Desktop\\4901jdataset\\"
        angry=os.listdir(path+"angry")
        happy=os.listdir(path+"happy")
        neutral=os.listdir(path+"neutral")
        correct_loss = np.abs(randn(1, 3))


        print("start angry!")
        for pic in angry:
            img = cv.imread(path+"angry\\"+pic)
            img = cv.resize(img, (128, 72))
            self.Fully_Connected(img,correct_loss*[1,-1,-1])

        print("start happy!")
        for pic in happy:
            img = cv.imread(path+"happy\\"+pic)
            img = cv.resize(img, (128, 72))
            self.Fully_Connected(img, correct_loss*[-1,1,-1])

        print("start neutral!")
        for pic in neutral:
            img = cv.imread(path+"neutral\\"+pic)
            img = cv.resize(img, (128,72))
            self.Fully_Connected(img,correct_loss*[-1,-1,1])

        X=X.reshape(1,X.shape[0]* X.shape[1] * X.shape[2])
        h = np.dot(X, self.w1)
        y_pred = h.dot(self.w2)
        score=np.argmax(y_pred)

        return (score,y_pred)

    def Fully_Connected(self,X,label):
        N, D_in, H, D_out = 1, X.shape[0]* X.shape[1] * X.shape[2], 100, 3
        x, y = X.reshape(1,X.shape[0]* X.shape[1] * X.shape[2]),label
        loss=0
        for t in range(100):
            h = np.maximum(0, np.dot(x, self.w1))
            y_pred = h.dot(self.w2)
            loss = np.square(y_pred - y).sum()

            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 = h.T.dot(grad_y_pred)
            dmiddle = np.dot(grad_y_pred, self.w2.T)
            dmiddle[h <= 0] = 0
            grad_w1 = x.T.dot(dmiddle)
            self.w1 -= 1e-12 * grad_w1
            self.w2 -= 1e-12 * grad_w2
        print("finished! with loss",loss)

nb=Nearest_Neighbor()
img=cv.imread("C:\\Users\\yauch\\Desktop\\4901jdataset\\neutral\\IMG_20190411_144412.jpg")
img=cv.resize(img, (128,72))
print(img.shape)
score,allscore=nb.Predict(img)
print(score,allscore)