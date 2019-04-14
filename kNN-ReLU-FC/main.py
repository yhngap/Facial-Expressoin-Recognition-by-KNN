import numpy as np
from numpy.random import randn
import cv2 as cv
import os
def capture_video():

    capture = cv.VideoCapture(0)  # 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
    while True:
        ret, frame = capture.read()  # 读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
        frame = cv.flip(frame, 1)  # 翻转 0:上下颠倒 大于0水平颠倒   小于180旋转
        cv.imshow("video", frame)
        if cv.waitKey(10) & 0xFF == ord('q'):  # 键盘输入q退出窗口，不按q点击关闭会一直关不掉 也可以设置成其他键。
            break


class Nearest_Neighbor:
    def __init__(self):
        print("created")

    def Train(self,x,y):
        self.Xtr=x
        self.ytr=y

    def Predict(self,X):

        min_score=0
        predict_label=0
        path="C:\\Users\\yauch\\Desktop\\4901jdataset\\"
        angry=os.listdir(path+"angry")
        happy=os.listdir(path+"happy")
        neutral=os.listdir(path+"neutral")
        for pic in angry:
            img = cv.imread(path+"angry\\"+pic)
            cv.imshow("imgage", img)
            cv.waitKey(0)
    def nearestneighbor(self):
        arr = os.listdir("C:\\Users\\yauch\\Desktop\\4901jdataset")

        img = cv.imread("C:\\Users\\yauch\\Desktop\\ml.png")
        cv.imshow("imgage", img)
        cv.waitKey(0)
        print(type(img))
        pass

    def Fully_Connected(self):
        N, D_in, H, D_out = 64, 1000, 100, 10
        x, y = randn(N, D_in), randn(N, D_out)
        w1, w2 = randn(D_in, H), randn(H, D_out)

        for t in range(50):
            h = 1 / (1 + np.exp(-x.dot(w1)))
            y_pred = h.dot(w2)
            loss = np.square(y_pred - y).sum()
            print(t, loss)

            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 = h.T.dot(grad_y_pred)
            grad_h = grad_y_pred.dot(w2.T)
            grad_w1 = x.T.dot(grad_h * h * (1 - h))
            w1 -= 1e-4 * grad_w1
            w2 -= 1e-4 * grad_w2




