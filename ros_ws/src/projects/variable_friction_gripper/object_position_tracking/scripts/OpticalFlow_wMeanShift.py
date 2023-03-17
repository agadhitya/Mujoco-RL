import numpy as np
import cv2

x, y = 300, 170
w, h = 75, 75

cap = cv2.VideoCapture('/home/joshua/FrictionFinger_videos/1.mp4')
# cap = cv2.VideoCapture('/home/joshua/FrictionFinger_videos/video_Hex.mp4')

# First frame of VideoCapture
ret, fram = cap.read()


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
old_gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask_opt = np.zeros_like(fram)

track_window = (x, y, w, h)

# Setup Region of Interest
roi = fram[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, 0, 255, cv2.NORM_MINMAX)

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



        # Optical flow
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]


        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
        
            a,b = new.ravel()
            c,d = old.ravel()
            if(a > x and x < x+w and b > y and b < y+w):
                mask_opt = cv2.line(mask_opt, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        
        
    
        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
        
        img2 = cv2.add(frame,mask_opt)
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        # img2 = cv2.polylines(fram, [pts], True, 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break 
    else:
        break


