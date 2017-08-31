import cv2
import os
path = os.path.join('/s/red/a/nobackup/vision/UCF-101', 'v_YoYo_g25_c05.avi')
vid = cv2.VideoCapture(path)
print(vid.isOpened)
ret, frame = vid.read()
print(ret)
print(frame)
