import numpy as np
import cv2

x, y = 280, 170
w, h = 95, 95

cap = cv2.VideoCapture('/home/joshua/FrictionFinger_videos/1.mp4')
# cap = cv2.VideoCapture('/home/joshua/FrictionFinger_videos/video_Hex.mp4')

# First frame of VideoCapture
ret, fram = cap.read()

# Initial Location of window
# x, y, w, h = 300, 170, 75, 75


track_window = (x, y, w, h)

# Setup Region of Interest
roi = fram[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, 0, 255, cv2.NORM_MINMAX)


cv2.imshow('roi', roi)
cv2.waitKey(0)
# Termination Criteria: Either 10 iteration or move bybatleast 1 point
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Using Meanshift
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Plotting on the image

        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)


        # img2 = cv2.polylines(fram, [pts], True, 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break 
    else:
        break


