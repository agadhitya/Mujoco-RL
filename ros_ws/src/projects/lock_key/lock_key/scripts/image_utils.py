# -*- coding: utf-8 -*-
"""
Contains image processing utility functions using cv2.
"""

import copy
import cv2
from math import atan2, cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np
import rospy
from scipy.ndimage import label
from scipy.stats import trim_mean

def get_distance(pt1,pt2):
    '''Returns distance between two points.'''
    return sqrt((pt2[1]-pt1[1])**2+(pt2[0]-pt1[0])**2)

def modify_key_angle(img, angle, key_data, cntr, search_radius):
    '''Adds PI to angle if not pointing towards key tip.'''
    #Identify the 2x pixel locations within search radius (along primary axis)
    p1_max=np.array([cntr[0]+search_radius*sin(angle), cntr[1]+search_radius*cos(angle)],dtype=int)
    p2_max=np.array([cntr[0]-search_radius*sin(angle), cntr[1]-search_radius*cos(angle)],dtype=int)
    p1_max_swap=np.array([p1_max[1],p1_max[0]])
    p2_max_swap=np.array([p2_max[1],p2_max[0]])
    #Find closest key points to extreme points
    p1=np.array(key_data[np.sum(np.square(np.abs(key_data-p1_max_swap)),1).argmin()],dtype=int)
    p2=np.array(key_data[np.sum(np.square(np.abs(key_data-p2_max_swap)),1).argmin()],dtype=int)
    p1=tuple((p1[1],p1[0]))
    p2=tuple((p2[1],p2[0]))
    #Draw points for visualization
    img = cv2.circle(img, tuple(p1_max), 2, (0,0,255), 4)
    img = cv2.circle(img, p1, 2, (255,0,0), 4)
    img = cv2.circle(img, tuple(p2_max), 2, (0,0,255), 4)
    img = cv2.circle(img, p2, 2, (0,255,0), 4)
    #Find distances from key center to extreme datapoints along primary axis
    d1=get_distance(cntr,p1)
    d2=get_distance(cntr,p2)
    #Assume longer distance is towards key tip
    #If the angle is not already pointing away from key tip, then add PI
    if d1<d2:
        # rospy.loginfo('D1<D2, adding PI.')
        return img, angle
    else:
        return img, angle+pi

def get_orientation(img, data, data_center, search_radius=50):
    '''Returns orientation (radians) CCW from horizontal using PCA.'''
    filtered_data=[]
    # Reduce region to pixels within "search_radius" of data_center
    #TODO: USE ARRAY OPERATIONS (np.where) rather than loop
    for ii in range(len(data)):
        if get_distance(data[ii],(data_center[1],data_center[0]))<=search_radius:
            filtered_data.append(data[ii])
    filtered_data=np.array(filtered_data)
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(filtered_data, mean)
    # Calculate angle
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    # Calculate center
    cntr = (int(mean[0,1]), int(mean[0,0]))

    #Rotate key angle by 180 if needed
    img,angle=modify_key_angle(img,angle, filtered_data, data_center, search_radius)

    return img, angle, eigenvectors, eigenvalues, cntr

def drawAxis(img, p_, q_, colour, scale):
    '''Draw Axis from PCA.'''
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    return img

def draw_circles(img,search_box): 
    '''Detects circles in image and plots them.'''     
    # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
      
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
      
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                   param2 = 30, minRadius = 10, maxRadius = 25)
    
    #Define bounds from search_box arg
    xmin=search_box['x']-int(search_box['width']/2)
    xmax=search_box['x']+int(search_box['width']/2)
    ymin=search_box['y']-int(search_box['height']/2)
    ymax=search_box['y']+int(search_box['height']/2)
    
    # Draw circles that are detected. 
    if detected_circles is not None: 
      
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
      
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            if (a>=xmin) & (a<=xmax) & (b>=ymin) & (b<=ymax):        
                # Draw the circumference of the circle. 
                cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 

    return img

def color_change(img,search_box,min_hsv=[150,0,0],max_hsv=[255,150,150],
                 new_pixel=[255,0,0],centroid_pixel=[255,0,255],show_pca=False):
    '''Change pixels within a certain color range to a new color.'''
    #Convert input to arrays
    min_hsv=np.array(min_hsv)
    max_hsv=np.array(max_hsv)
    #Isolate desired pixel indices
    region=np.array(find_color(img,min_hsv,max_hsv))
    #Create boolean filter for indexing
    x_filt=np.array((region[1]>=int(search_box['min']['x'])) & (region[1]<=int(search_box['max']['x'])),dtype=bool)
    y_filt=np.array((region[0]>=int(search_box['min']['y'])) & (region[0]<=int(search_box['max']['y'])),dtype=bool)
    region_filter=x_filt & y_filt
    #Build final type of pixel indices
    final_region=(region[0][region_filter],
                  region[1][region_filter])
    #Change color of pixels
    new_img=copy.deepcopy(img)
    new_img[final_region]=new_pixel
    
    # Manual 'Centroid' finding
    try:
        # if len(final_region[0])>200:
        # Identify mean index and mark with circle
        # Mean is trimmed on both tails
        x_ave=int(trim_mean(final_region[1],0.4))
        y_ave=int(trim_mean(final_region[0],0.4))
        center=(x_ave, y_ave)
        cv2.circle(new_img, center, 1, centroid_pixel, 6)
    except ValueError:
        center=(None,None)
        rospy.loginfo("CENTROID FINDING FAILED")
        pass

    #Calculate orientation (radians) of final_region
    reg=np.array(final_region,dtype=np.float64)
    reg=reg.T
    search_radius=50
    if show_pca:
        new_img, orientation, eigenvectors, eigenvalues, cntr=get_orientation(new_img,reg,center,search_radius=search_radius)
        #Draw axes from PCA
        pca_img1=copy.deepcopy(new_img)
        p1 = (cntr[0] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0])
        pca_img1=drawAxis(new_img, cntr, p1, (0, 255, 0), 10)
        final_img = drawAxis(pca_img1, cntr, p2, (255, 255, 0), 50)
        #Show red line where orientation is measured from
        final_img =cv2.line(final_img, cntr, (int(cntr[0]+25*sin(orientation)), int(cntr[1]+25*cos(orientation))), (0,0,255), 2, cv2.LINE_AA)
        #Show search PCA-inclusion radius
        final_img = cv2.circle(final_img, center, search_radius, (0,0,255), 1)
        return final_img, center, orientation
    else:
        orientation=None
        return new_img, center, orientation

def find_color(img,min_hsv,max_hsv):
    '''Identify pixels in image within a color range.'''
    #Convert img to HSV
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    region=np.where((img[:,:,0]>=int(min_hsv[0])) & (img[:,:,0]<=int(max_hsv[0])) & 
                    (img[:,:,1]>=int(min_hsv[1])) & (img[:,:,1]<=int(max_hsv[1])) & 
                    (img[:,:,2]>=int(min_hsv[2])) & (img[:,:,2]<=int(max_hsv[2])))
    
    return region

def add_box(img,x,y,width,height,color=[0,0,255]):
    '''Adds a box to an image'''
    img=cv2.rectangle(img,
                      (x-int(width/2),y-int(height/2)), #Start
                      (x+int(width/2),y+int(height/2)), #End
                      color, #Font Color (BGR)
                      2) #thickness
    return img

def add_quad(img,search_box,color=[0,0,255]):
    '''Adds a quadrilateral to an image'''
    p1=(int(search_box['min']['x']),int(search_box['max']['y']))
    p2=(int(search_box['min']['x']),int(search_box['min']['y']))
    p3=(int(search_box['max']['x']),int(search_box['min']['y']))
    p4=(int(search_box['max']['x']),int(search_box['max']['y']))
    img=cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
    img=cv2.line(img, p2, p3, color, 2, cv2.LINE_AA)
    img=cv2.line(img, p3, p4, color, 2, cv2.LINE_AA)
    img=cv2.line(img, p4, p1, color, 2, cv2.LINE_AA)
    return img

def detect_edges(img):
    '''Detects edges in image.'''
    edges = cv2.Canny(img,100,200)
    return edges

def depth(left_img,right_img):
    '''Calculate and view the disparity map from two images.'''
    stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=11) #16, 15
    disparity = stereo.compute(left_img,right_img)
    plt.imshow(disparity,'plasma')#'gray')
    plt.show()

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)
    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255
    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

def run_watershed(img,show_center=False):
    # Pre-processing.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,np.ones((3, 3), dtype=int))
    result = segment_on_dt(img, img_bin) 

    if show_center:
        region=np.where(result==255)
        #Identify center point
        x_idx,y_idx=int(trim_mean(region[1],0.4)),int(trim_mean(region[0],0.4))
        # Center coordinates
        center_coordinates = (x_idx,y_idx)
        # Radius of circle
        radius = 10
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        result = cv2.circle(result, center_coordinates, radius, color, thickness)
    return result

def add_blur(img):
    img = cv2.blur(img,(3,3))
    img=cv2.medianBlur(img,3)
    img= cv2.GaussianBlur(img,(5,5),0)
    img= cv2.bilateralFilter(img,9,5,5)
    return img

if __name__=='__main__':
    data=np.array([[1,1],[2,7],[3,3]], dtype=np.float64)
    print(data.T.shape)
    orientation=get_orientation(data)
    print(orientation)