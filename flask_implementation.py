import cv2 as cv
from keras.models import load_model
from time import sleep
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
app = Flask(__name__)

@app.route('/')
def flask_imp():
    # return('helloo world!')
    alph=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','none']
    model=load_model('./logs')
    cap=cv.VideoCapture('capstone-img/test_hold.mp4')
    if cap.isOpened()==False:
        print('error reading stream')
    old=-1
    while(cap.isOpened()):
        ret, frame=cap.read()
        if ret==False:
            cap.release()
            break
        # c=plt.imshow(frame)
        # plt.show(block=False)
        b=frame[:,280:1000,:]
        b=resize(b,[72,72,3],preserve_range=True)
        b=np.reshape(b,[1,72,72,3])
        c=np.argmax(model.predict(b))
        if c!=old:
            print('Prediction: '+alph[c])
            # if c==26:
            #     plt.show(block=False)
        old=c
        sleep(.5)
        # plt.close('All')

    cap.release()

if __name__=='__main__':
    flask_imp()
    print('hello world!')
