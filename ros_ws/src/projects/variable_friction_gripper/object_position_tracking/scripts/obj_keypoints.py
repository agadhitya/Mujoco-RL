import cv2
import numpy as np

# img = cv2.imread('/home/gsathyanarayanan/5.png')
img = cv2.imread('/home/gsathyanarayanan/6.jpg')
print img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canvas = np.zeros((img.shape[0], img.shape[1], 3))
# Filter
kernel = np.ones((5,5), np.float32)
gray = cv2.filter2D(gray, -1, kernel)

# Contours
ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cnt = contours[0]
max_area = cv2.contourArea(cnt)

for cont in contours:
	if cv2.contourArea(cont) > max_area:
		cnt = cont
		max_area = cv2.contourArea(cont)

perimeter = cv2.arcLength(cnt, True)
epsilon = 0.04*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

hull = cv2.convexHull(cnt)

#cv2.drawContours(canvas, cnt, -1, (0,255,0), 3)
cv2.drawContours(canvas, approx, -1, (0,0,255), 3)

cv2.imshow("Contour", canvas)
k = cv2.waitKey(0)

if k==27:
	cv2.destroyAllWindows()




# import cv2
# import numpy as np

# img = cv2.imread('/home/gsathyanarayanan/3.png')

# #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = img
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)

# img = cv2.drawKeypoints(gray, kp, img)
# cv2.imshow('keypoints', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt


# img = cv2.imread('/home/gsathyanarayanan/6.jpg')

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# #kernel = np.ones((15,15),np.uint8)

# #dst = cv2.dilate(gray,kernel)
# dst = cv2.cornerHarris(gray,2,3,0.04)

# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# #kernel = np.ones((1,1),np.uint8)
# #dst = cv2.dilate(gray,kernel,iterations = 1)
# # print dst.shape
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]

# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
# corners = cv2.goodFeaturesToTrack(dst,4,0.01,10)
# #corners = cv2.cornerHarris(gray, 2,3,0.04)
# corners = np.int0(corners)
# print corners
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(dst,(x,y),3,255,-1)

# plt.imshow(img),plt.show()
############################################################################

# import numpy as np
# import cv2

# from matplotlib import pyplot as plt

# img1 = cv2.imread('/home/gsathyanarayanan/3.png',0)          # queryImage
# img2 = cv2.imread('/home/gsathyanarayanan/4.png',0) # trainImage

# # Initiate SIFT detector
# #sift = cv2.SIFT()
# sift = cv2.xfeatures2d.SIFT_create()

# # find the keypoints and descriptors with SIFT
# des1 = cv2.goodFeaturesToTrack(img1,25,0.01,10)
# #print des1
# #kp1, des1 = cv2.goodFeaturesToTrack(img1,25,0.01,10)
# #kp2, des2 = cv2.goodFeaturesToTrack(img2,25,0.01,10)
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# print len(kp1)
# print des1.shape
# #corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img1,flags=2)

# plt.imshow(img3),plt.show() 


# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img = cv2.imread('/home/gsathyanarayanan/5.jpg',0)
# print img.shape
# # Initiate FAST object with default values
# fast = cv2.FastFeatureDetector_create()


# # find and draw the keypoints
# kp = fast.detect(img,None)
# print dir(fast)
# img2 = np.zeros((img.shape[0], img.shape[1], 3))
# img2 = cv2.drawKeypoints(img, kp ,img2, color=(255,0,0))

# # Print all default params
# # print "Threshold: ", fast.getInt('threshold')
# # print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
# # print "neighborhood: ", fast.getInt('type')
# # print "Total Keypoints with nonmaxSuppression: ", len(kp)

# cv2.imwrite('fast_true.png',img2)

# # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img,None)

# print "Total Keypoints without nonmaxSuppression: ", len(kp)

# img3 = np.zeros((img.shape[0], img.shape[1], 3))
# img3 = cv2.drawKeypoints(img, kp, img3, color=(255,0,0))
# cv2.imwrite('fast_false.png',img3)