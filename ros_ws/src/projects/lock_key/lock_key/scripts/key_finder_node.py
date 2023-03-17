#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS node for lock and key color segmentation and centroid finding.
Publishes lock and key centroids from color segmentation. Publishes
visualization image. Publishes key angle from PCA.
"""
import cv2
import image_utils
import numpy as np
import rospy
import image_geometry
from geometry_msgs.msg import PointStamped, QuaternionStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Float64
from cv_bridge import CvBridge, CvBridgeError
from math import pi
import tf
from scipy.stats import trim_mean

class FinderPub:
    '''Finds lock and key from RGBD data.'''
    def __init__(self):
        self.header=None
        self.key_search_box=None
        #TODO: Handle case when param is not yet available.
        self.hsv=rospy.get_param('hsv')
        self.key_min_hsv=self.hsv['key']['min']
        self.key_max_hsv=self.hsv['key']['max']
        self.key_center=None
        self.key_center_depth=None
        self.key_angle=None
        self.lock_search_box=None
        self.lock_min_hsv=self.hsv['lock']['min']
        self.lock_max_hsv=self.hsv['lock']['max']
        self.lock_center=None
        self.lock_center_depth=None
        #self.lock_yaw=None
        #Use image_geometry for camera calculations
        self.rgb_camera=image_geometry.PinholeCameraModel()
        self.rgb_camera_info_received=False
        self.rgb_image=None
        self.rgb_received=False
        self.depth_array=None
        self.depth_received=False
        #ROS Subscribers and Publishers
        self.bridge=CvBridge()
        self.rgb_sub=rospy.Subscriber('/camera/color/image_raw',
                                      Image,self.rgb_callback, queue_size=1)
        self.depth_sub=rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                        Image,self.depth_callback, queue_size=1)
        self.rgb_info=rospy.Subscriber('/camera/color/camera_info',
                                       CameraInfo,self.rgb_info_callback, queue_size=1)
        self.vis_pub = rospy.Publisher('finder_image', Image, queue_size=1)
        self.key_pub = rospy.Publisher('key_point', PointStamped, 
                                       queue_size=1)
        self.lock_pub = rospy.Publisher('lock_point', PointStamped, 
                                        queue_size=1)
        self.key_angle_pub = rospy.Publisher('key_angle', Float64, 
                                           queue_size=1)
        # self.lock_yaw_pub = rospy.Publisher('lock_yaw', Float64, 
        #                                     queue_size=1)
        rospy.loginfo('Running lock and key finder node.')

    def get_local_depth(self, center,px_offset=15,trim_portion=0.4):
        '''Returns trimmed mean of depth array within square area around center.'''
        #Identify local area (square around key center) to get depth
        ymin=center[1]-px_offset
        ymax=center[1]+px_offset
        xmin=center[0]-px_offset
        xmax=center[0]+px_offset
        depth_region=self.depth_array[ymin:ymax,
                                      xmin:xmax]
        # Mean is trimmed on both tails
        depth=trim_mean(depth_region,trim_portion,axis=None)
        return depth

    def calculate_positions(self,img):
        '''Calculates centers of lock and key.'''
        # Get lock and key centers (in RGB image) in terms of pixels
        # RGB and Depth are registered so pixel locations are the same
        vis_img=self.get_centers(img)
        #Lock
        try:
            #Get local depth    
            self.lock_center_depth=self.get_local_depth(self.lock_center,px_offset=15,trim_portion=0.4)
            # Get normalized XYZ displacement (in camera_color_optical_frame)
            lock_disp=self.rgb_camera.projectPixelTo3dRay(self.lock_center)
            #Get scale from depth
            lock_scale=self.lock_center_depth/lock_disp[2]/1000.0
            #Convert to meters
            lock_disp=lock_scale*np.array(lock_disp)
            # Publish center
            self.publish_lock_point(lock_disp)
        except TypeError:
            rospy.loginfo('Lock Not Found in Frame')
        #Key
        try:
            #Get local depth
            self.key_center_depth=self.get_local_depth(self.key_center,px_offset=10,trim_portion=0.4)
            # Get normalized XYZ displacement (in camera_color_optical_frame)
            key_disp=self.rgb_camera.projectPixelTo3dRay(self.key_center)
            # Get scale from depth
            key_scale=self.key_center_depth/key_disp[2]/1000.0
            #Convert to meters
            key_disp=key_scale*np.array(key_disp)
            # Publish center
            self.publish_key_point(key_disp)
        except TypeError:
            rospy.loginfo('Key Not Found in Frame')

        #Publish visualization image
        self.publish_vis(vis_img)

    def depth_callback(self,raw_depth_img):
        '''Converts depth image to np.array with actual depth.'''
        try:
            #Convert to cv2 format
            depth_img=self.bridge.imgmsg_to_cv2(raw_depth_img, 
                                                desired_encoding="passthrough")
            #Convert the depth image to a Numpy array (with actual depth values)
            self.depth_array = np.array(depth_img, dtype=np.float32)
            self.depth_received=True
        except CvBridgeError, e:
            print e

    def get_centers(self,img):
        '''Compute object centers and adds visuals to image.'''
        #Update img for key
        img,self.key_center,self.key_angle=image_utils.color_change(img,
                                                                 self.key_search_box,
                                                                 self.key_min_hsv,
                                                                 self.key_max_hsv,
                                                                 new_pixel=[0,255,255], #Yellow
                                                                 centroid_pixel=[0,0,255], #Red 
                                                                 show_pca=True)
        #Update img for lock (ignore orientation for now)
        img,self.lock_center,_=image_utils.color_change(img,
                                                        self.lock_search_box,
                                                        self.lock_min_hsv,
                                                        self.lock_max_hsv,
                                                        new_pixel=[0,255,0], #Green
                                                        centroid_pixel=[255,0,255],  #Pink
                                                        show_pca=False)
        
        #Add search box to img for key
        img=image_utils.add_quad(img,self.key_search_box,color=[0,0,255]) #Red
        #Add search box to img for lock
        img=image_utils.add_quad(img,self.lock_search_box,color=[255,0,255]) #Pink
        
        return img

    def publish_key_point(self,key_disp):
        '''Publish key position in terms of camera_color_optical_frame.'''
        #Build key msg
        key_msg=PointStamped()
        key_msg.header=self.header
        key_msg.point.x=key_disp[0]
        key_msg.point.y=key_disp[1]
        key_msg.point.z=key_disp[2]
        #Publish message
        self.key_pub.publish(key_msg)
        #Publish yaw
        self.key_angle_pub.publish(Float64(self.key_angle))

    def publish_lock_point(self,lock_disp):
        '''Publish lock position in terms of camera_color_optical_frame.'''
        #Build lock msg
        lock_msg=PointStamped()
        lock_msg.header=self.header
        lock_msg.point.x=lock_disp[0]
        lock_msg.point.y=lock_disp[1]
        lock_msg.point.z=lock_disp[2]
        #Publish message
        self.lock_pub.publish(lock_msg)
        #Publish yaw
        # self.lock_yaw_pub.publish(Float64(self.lock_yaw))

    def publish_vis(self,vis_img):
        '''Define and publish visualization image'''
        vis_img=self.bridge.cv2_to_imgmsg(vis_img,"bgr8")
        vis_img.header=self.header
        self.vis_pub.publish(vis_img)        

    def rgb_callback(self,raw_rgb_img):   
        '''Convert to cv2 format and store image.'''
        # try:
        self.header=raw_rgb_img.header
        self.rgb_img=self.bridge.imgmsg_to_cv2(raw_rgb_img,desired_encoding="bgr8")
        if self.rgb_img is not None:
            self.rgb_received=True

        if (self.rgb_received and self.depth_received and self.rgb_camera_info_received) and \
        (self.rgb_img is not None) and (self.depth_array is not None):
            self.calculate_positions(self.rgb_img)
        # except:
        #     rospy.loginfo('RGB Callback Exception')

    def rgb_info_callback(self,rgb_cam_info):   
        '''Store camera info. in Pinhole Model.'''
        try:
            if not self.rgb_camera_info_received:
                self.rgb_camera.fromCameraInfo(rgb_cam_info)
                self.rgb_camera_info_received=True
                rospy.loginfo('RGB Camera Info. Received')

            # Convert object bounds from meters (in /camera_color_optical_frame)
            # to pixels using Pinhole model
            
            # First Transform points from /camera_depth_optical_frame to /camera_color_optical_frame
            bounds=rospy.get_param('bounds')
            transformer=tf.TransformListener()
            transformer.waitForTransform(bounds['frame'], 
                                        "camera_color_optical_frame",
                                        rospy.Time(0), rospy.Duration(4.0))
            # This is for troubleshooting
            # bounds={'key':{'min':{'x':0.1,'y':-0.08,'z':0.4},
            #                'max':{'x':0.2,'y':0.0,'z':0.45}},
            #        'lock':{'min':{'x':-0.12,'y':-0.08,'z':0.4},
            #                'max':{'x':-0.05,'y':0.0,'z':0.45}},
            #        'frame':'/camera_depth_optical_frame'}
            key_min=PointStamped()
            key_max=PointStamped()
            lock_min=PointStamped()
            lock_max=PointStamped()
            #Define and add header (without timestamp)
            h=Header()
            h.frame_id=bounds['frame']
            key_min.header=h
            key_max.header=h
            lock_min.header=h
            lock_max.header=h
            #Define points from params
            key_min.point.x=bounds['key']['min']['x']
            key_min.point.y=bounds['key']['min']['y']
            key_min.point.z=bounds['key']['min']['z']
            key_max.point.x=bounds['key']['max']['x']
            key_max.point.y=bounds['key']['max']['y']
            key_max.point.z=bounds['key']['max']['z']
            lock_min.point.x=bounds['lock']['min']['x']
            lock_min.point.y=bounds['lock']['min']['y']
            lock_min.point.z=bounds['lock']['min']['z']
            lock_max.point.x=bounds['lock']['max']['x']
            lock_max.point.y=bounds['lock']['max']['y']
            lock_max.point.z=bounds['lock']['max']['z']
            #Calculate transformed points
            key_min_transformed=PointStamped()
            key_max_transformed=PointStamped()
            lock_min_transformed=PointStamped()
            lock_max_transformed=PointStamped()
            key_min_transformed=transformer.transformPoint('camera_color_optical_frame',key_min)
            key_max_transformed=transformer.transformPoint('camera_color_optical_frame',key_max)
            lock_min_transformed=transformer.transformPoint('camera_color_optical_frame',lock_min)
            lock_max_transformed=transformer.transformPoint('camera_color_optical_frame',lock_max)

            #Get points in pixel coordinates
            key_min_pix=self.rgb_camera.project3dToPixel((key_min_transformed.point.x,key_min_transformed.point.y,key_min_transformed.point.z))
            key_max_pix=self.rgb_camera.project3dToPixel((key_max_transformed.point.x,key_max_transformed.point.y,key_max_transformed.point.z))
            lock_min_pix=self.rgb_camera.project3dToPixel((lock_min_transformed.point.x,lock_min_transformed.point.y,lock_min_transformed.point.z))
            lock_max_pix=self.rgb_camera.project3dToPixel((lock_max_transformed.point.x,lock_max_transformed.point.y,lock_max_transformed.point.z))
            #Convert to list to make mutable
            key_min_pix=list(key_min_pix)
            key_max_pix=list(key_max_pix)
            lock_min_pix=list(lock_min_pix)
            lock_max_pix=list(lock_max_pix)

            #TODO: Find a better way to define/pad bounding box
            #Add padding to bounding boxes
            image_max_x=640
            image_max_y=480
            pad_key_x_max=55
            pad_key_x_min=0
            pad_key_y=0
            pad_lock_x=0
            pad_lock_y=0
            key_min_pix[0]=key_min_pix[0]-pad_key_x_min
            key_max_pix[0]=key_max_pix[0]+pad_key_x_max
            key_min_pix[1]=key_min_pix[1]-pad_key_y
            key_max_pix[1]=key_max_pix[1]+pad_key_y
            lock_min_pix[0]=lock_min_pix[0]-pad_lock_x
            lock_max_pix[0]=lock_max_pix[0]+pad_lock_x
            lock_min_pix[1]=lock_min_pix[1]-pad_lock_y
            lock_max_pix[1]=lock_max_pix[1]+pad_lock_y

            #Check if we exceed image bounds, then set to 0, 480, 640, etc.
            if key_min_pix[0]<0:
                key_min_pix[0]=0
            if key_max_pix[0]>image_max_x:
                key_max_pix[0]=image_max_x
            if key_min_pix[1]<0:
                key_min_pix[1]=0
            if key_max_pix[1]>image_max_y:
                key_max_pix[1]=image_max_y
            if lock_min_pix[0]<0:
                lock_min_pix[0]=0
            if lock_max_pix[0]>image_max_x:
                lock_max_pix[0]=image_max_x
            if lock_min_pix[1]<0:
                lock_min_pix[1]=0
            if lock_max_pix[1]>image_max_y:
                lock_max_pix[1]=image_max_y
            
            self.key_search_box={'min':{'x':int(key_min_pix[0]),'y':int(key_min_pix[1])},
                                'max':{'x':int(key_max_pix[0]),'y':int(key_max_pix[1])}}

            self.lock_search_box={'min':{'x':int(lock_min_pix[0]),'y':int(lock_min_pix[1])},
                                'max':{'x':int(lock_max_pix[0]),'y':int(lock_max_pix[1])}}
        except:
            rospy.loginfo('RGB Info Callback Exception')

def main():
    rospy.init_node('lock_key_finder_pub',anonymous=True)
    finder=FinderPub()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
