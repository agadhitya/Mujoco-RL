#!/usr/bin/env python
'''
Saves messages from color and depth image ROS topics to JPEGs.
Topics and bag are input by user at the bottom of this file.

Usage: python rosbag_utils.py
'''

import rosbag
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg,filename):
	'''Save color image to file.'''
	print "Received an image!"
	try:
		# Convert your ROS Image message to OpenCV2
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError, e:
		print(e)
	else:
		# Save your OpenCV2 image as a jpeg
		cv2.imwrite(path+filename+'.jpeg', cv2_img)

def depth_callback(msg,filename):
	'''Save depth image to file.'''
	print "Received depth!"
	try:
		NewImg = bridge.imgmsg_to_cv2(msg, "passthrough")
		depth_array = np.array(NewImg, dtype=np.float32)
		cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
		# Save your OpenCV2 image as a jpeg
		cv2.imwrite(path+filename+'.jpeg', depth_array*255)
	except CvBridgeError as e:
		print(e)

def get_first_msg(bag,topic):
	'''Returns type and contents of first message for given
	bag & topic.'''
	msgs=bag.read_messages(topics=topic)
	first_msg=msgs.next().message
	msg_type=type(first_msg)
	msg_shape=np.shape(np.array([first_msg]))
	#image_callback(first_msg)
	return msg_type, msg_shape, first_msg 

def get_all_msgs(bag,topic,image_type='color'):
	'''Saves all images to disk from a given bag & topic.'''
	msgs=bag.read_messages(topics=topic)

	if image_type=='color':
		callback=image_callback
		prefix='color'
	elif image_type=='depth':
		callback=depth_callback
		prefix='depth'

	for idx,msg in enumerate(msgs):
		msg=msgs.next().message
		filename=prefix+'_image_'+str(idx)
		callback(msg,filename)

if __name__ == '__main__':	
	#Define output path
	path="/mnt/hgfs/MER_Shared/rosbags/eye_in_hand/lock_and_key/"
	bag = rosbag.Bag("/mnt/hgfs/MER_Shared/rosbags/eye_in_hand/lock_and_key.bag")
	topics=['/camera/color/image_raw',
			'/camera/depth/image_rect_raw']

	get_all_msgs(bag,topics[0],'color')
	#get_all_msgs(bag,topics[1],'depth')