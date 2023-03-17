#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS Service Server for retrieving average of the past 30
lock and key positions. Publishes lock and key positions as topics.
Publishes key yaw as a topic.
"""
from __future__ import print_function

from lock_key_msgs.srv import GetLockKeyPoses, GetLockKeyPosesResponse
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header, Float64
from collections import deque
import rospy
import tf
from tf2_ros import TransformException
import tf_conversions

#Number of samples to use when average key/lock positions (and key angle)
nsamples=30
key_point={'x':deque(maxlen=nsamples),
		   'y':deque(maxlen=nsamples),
		   'z':deque(maxlen=nsamples)}
lock_point={'x':deque(maxlen=nsamples),
		    'y':deque(maxlen=nsamples),
		    'z':deque(maxlen=nsamples)}
key_angle=deque(maxlen=nsamples)
# lock_yaw=deque(maxlen=nsamples)

def key_callback(key_msg):
	'''Adds latest key_msg to FIFO queue.'''
	global key_point
	key_point['x'].append(key_msg.point.x)
	key_point['y'].append(key_msg.point.y)
	key_point['z'].append(key_msg.point.z)

def lock_callback(lock_msg):
	'''Adds latest lock_msg to FIFO queue.'''
	global lock_point
	lock_point['x'].append(lock_msg.point.x)
	lock_point['y'].append(lock_msg.point.y)
	lock_point['z'].append(lock_msg.point.z)

def key_angle_callback(key_angle_msg):
	'''Adds latest key_angle_msg to FIFO queue.'''
	global key_angle
	key_angle.append(key_angle_msg.data)

# def lock_yaw_callback(lock_yaw_msg):
#	'''Adds latest lock_yaw_msg to FIFO queue.''
# 	global lock_yaw
# 	lock_yaw.append(lock_yaw_msg.data)

def ave(data):
	'''Calculates average'''
	return sum(data)/len(data)

def get_poses(req):
	'''Returns poses of lock and key from vision system.'''
	global key_point, lock_point, key_angle
	pose_retrieved=False
	while not pose_retrieved:
		try:
			#Build PointStamped Messaged for lock and Key
			key_response=PointStamped()
			key_response.point.x=ave(key_point['x'])
			key_response.point.y=ave(key_point['y'])
			key_response.point.z=ave(key_point['z'])
			lock_response=PointStamped()
			lock_response.point.x=ave(lock_point['x'])
			lock_response.point.y=ave(lock_point['y'])
			lock_response.point.z=ave(lock_point['z'])
			h = Header()
			h.stamp = rospy.Time.now()
			h.frame_id='camera_color_optical_frame'
			key_response.header=h
			lock_response.header=h
			#Transform msgs to map frame
			transformer=tf.TransformListener()
			transformer.waitForTransform(h.frame_id, "map", rospy.Time(0), rospy.Duration(5.0))
			key_response=transformer.transformPoint('map',key_response)
			lock_response=transformer.transformPoint('map',lock_response)
			response=GetLockKeyPosesResponse()
			#Build key_pose msg
			response.key_pose.header=key_response.header
			response.key_pose.pose.position=key_response.point
			#TODO: Consider defining a heuristic to determine roll angle rather than
			#hardcoding. Current value assumes "tooth-side up"
			#Convert RPY to Quaternion
			key_quat=tf_conversions.transformations.quaternion_from_euler(-1.5708,0,ave(key_angle))
			response.key_pose.pose.orientation.x = key_quat[0]
			response.key_pose.pose.orientation.y = key_quat[1]
			response.key_pose.pose.orientation.z = key_quat[2]
			response.key_pose.pose.orientation.w = key_quat[3]

			response.lock_point=lock_response
			pose_retrieved=True

			print(response)
			print(key_angle)
		#If call fails due to "Transform lookup in past" error, then try again
		except TransformException:
			pass
	return response

def lock_key_pose_server():
	rospy.init_node('lock_and_key_poses_server')
	rospy.Subscriber("/lock_point", PointStamped, lock_callback)
	rospy.Subscriber("/key_point", PointStamped, key_callback)
	#TODO: Get lock_yaw (or maybe just use PCL plane detection and not do anything here)
	#rospy.Subscriber("/lock_yaw", Float64, lock_yaw_callback)
	rospy.Subscriber("/key_angle", Float64, key_angle_callback)
	s = rospy.Service('lock_and_key_poses', GetLockKeyPoses, get_poses)
	print("Ready to retrieve poses.")
	rospy.spin()

if __name__ == "__main__":
	lock_key_pose_server()