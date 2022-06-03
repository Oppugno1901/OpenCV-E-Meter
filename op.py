import numpy
import cv2

_,contours,hierarchy = cv2.findContours(thresh3,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

1