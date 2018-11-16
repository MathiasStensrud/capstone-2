import cv2 as cv
import numpy as np
import skimage
from skimage.transform import resize

def fill(vid, supp=False):
    i=0
    cap=cv.VideoCapture(f'capstone-img/{vid}.mp4')
    ret=True
    while(ret==True):
        ret, frame=cap.read()
        if ret==False:
            cap.release()
            break
        if i%5 ==0:
            b=frame[:,280:1000,:]
            b=resize(b,[72,72,3],preserve_range=True)
            cv.imwrite(f'data/augment/Frame{i}.png', b)
            if supp==False:
                print('Frame: ',i)
        i+=1
    cap.release()
