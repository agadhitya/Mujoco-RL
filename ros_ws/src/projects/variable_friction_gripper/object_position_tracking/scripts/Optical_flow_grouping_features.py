import numpy as np
import cv2
import time


FRAME_TIME_STEP = 0.02
cap = cv2.VideoCapture('/home/joshua/FrictionFinger_videos/2.mp4')

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
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
c = 0
ret = True
while(ret):
    start_time = time.time()
    # while(time.time() - start_time < FRAME_TIME_STEP):
    #     cap.read()
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    # good_new = p1[st==1]
    # good_old = p0[st==1]
    count = 0
    good_new = p1
    good_old = p0
    
    distance = {} 
    angles = {}
    # draw the tracks
    # if (c==0):    
    for i,(new,old) in enumerate(zip(good_new,good_old)):

        distance[i] = np.linalg.norm(new - old)
        angles[i] = np.arctan2(new[0,1]-old[0,1], new[0,0]- old[0,0])
    sorted_distance = sorted(distance.items(), key = lambda distance:distance[1])
    c = 0
    print('max distance', max(sorted_distance))
    print('min distance', min(sorted_distance))
        # if distance > 1:
        #     # print 'i = ', i
        #     # print good_new.shape
        #     # print np.delete(good_new, i, axis = 0).shape
        #     # good_new = np.delete(good_new, i, axis = 0)
        #     # good_old = np.delete(good_old, i, axis = 0)
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), color[count].tolist(), 2)
        #     frame = cv2.circle(frame,(a,b),5,color[count].tolist(),-1)
        #     count = count + 1
    # img = cv2.add(frame,mask)


            
    for k in sorted_distance:
        
        # (point ,d) = sorted_distance[i]
        new = good_new[k[0]]
        old = good_old[k[0]]
        # if count > 5*len(sorted_distance)/6:
        # print k
        if k[1] > 0.002:
            # print 'i = ', i
            # print good_new.shape
            # print np.delete(good_new, i, axis = 0).shape
            # good_new = np.delete(good_new, i, axis = 0)
            # good_old = np.delete(good_old, i, axis = 0)
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), (255,0,0), 2)
            # mask = cv2.line(mask, (a,b),(c,d), color[0].tolist(), 2)
            # frame = cv2.circle(frame,(a,b),1,color[0].tolist(),-1)
        else:
            # print 'i = ', i
            # print good_new.shape
            # print np.delete(good_new, i, axis = 0).shape
            # good_new = np.delete(good_new, i, axis = 0)
            # good_old = np.delete(good_old, i, axis = 0)
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), (0,255,0), 5)
            # mask = cv2.line(mask, (a,b),(c,d), color[5].tolist(), 2)
            # frame = cv2.circle(frame,(a,b),5,color[5].tolist(),-1)
        count = count + 1
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()