import sys
sys.path.append("./yolo_realtime")
import numpy as np
import cv2
from cv2 import CAP_DSHOW

if False:
    cap = cv2.VideoCapture(1)
else:
    cap = cv2.VideoCapture(1, apiPreference=CAP_DSHOW)

from yolo_realtime.yolo_wrapper import Yolo_Wrapper
yolo = Yolo_Wrapper(device='0')
while True:
    _, img = cap.read()
    yolo.detect([img])

# from yolo_realtime.detect import detect
# detect()
