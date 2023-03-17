#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines high-level interface for using MoveIt Commander with the Panda arm.

https://frankaemika.github.io/docs/franka_ros.html?highlight=reflex
"""
import sys
import actionlib
from math import cos, pi, sin, sqrt
import moveit_commander
import numpy as np
import rospy
import tf
import tf_conversions
#Services
from lock_key.srv import getAveWrench, getWrench
from moveit_planner.srv import GetTF, GetPose
#Messages
from std_msgs.msg import Header
import franka_gripper.msg
import geometry_msgs.msg

class Commander:
	'''Commander interface for Panda.'''
	def __init__(self,group_name="panda_arm",vel_scale=0.1,gazebo=False):
		#Initialize commander
		moveit_commander.roscpp_initialize(sys.argv)
		self.robot = moveit_commander.RobotCommander()
		self.scene = moveit_commander.PlanningSceneInterface()
		self.add_table()
		self.gazebo=gazebo
		#Select control group
		if self.gazebo:
			self.group_name="arm"
		else:
			self.group_name = group_name
			from franka_msgs.srv import SetLoad, SetLoadRequest, SetEEFrame, SetEEFrameRequest
		self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
		#Set velocity scale
		self.set_vel_scale(vel_scale)
		#Define joint home positions (matches "start" pose)
		self.joint_home=[0.0,-0.785,0.0,-2.356, 0.0, 1.57, 0.784]
		#Wait for action server
		self.gripper_client = actionlib.SimpleActionClient('franka_gripper/grasp', 
											  	      	   franka_gripper.msg.GraspAction)
		rospy.loginfo('Waiting for actions, services, and TF listener')
		if not self.gazebo:
			self.gripper_client.wait_for_server()
			
		#Wait for services
		rospy.wait_for_service('get_transform')
		rospy.wait_for_service('get_pose')
		rospy.wait_for_service('getWrench')
		rospy.wait_for_service('getAveWrench')
		#Define service clients
		self.get_tf_client = rospy.ServiceProxy('get_transform', GetTF)
		self.get_pose_client = rospy.ServiceProxy('get_pose', GetPose)
		self.get_wrench_client = rospy.ServiceProxy('getWrench', getWrench)
		self.get_ave_wrench_client = rospy.ServiceProxy('getAveWrench', getAveWrench)
		self.transformer=tf.TransformListener()
		rospy.loginfo('Initialized Commander Interface and Action Clients')

		#Set EE Frame
		self.move_group.set_end_effector_link('panda_EE_control')
		rospy.loginfo('Set EE link for move_group')

	def __repr__(self):
		'''Defines string representation of Commander object.'''
		s=self.group_name + " Commander:\n" + str(self.robot.get_current_state())
		s+="\nEE:" + str(self.move_group.get_end_effector_link())
		return s

	def add_table(self):
		'''Adds table to planning scene.'''
		box_pose = geometry_msgs.msg.PoseStamped()
		box_pose.header.frame_id = "map"
		box_pose.pose.orientation.w = 1.0
		box_pose.pose.position.z = -0.05
		box_name = "table"
		self.scene.add_box(box_name, box_pose, size=(2.0, 2.0, 0.1))

	def found_hole(self,min_spiral_force,tx_limit,ty_limit):
		'''Returns bool stating whether hole was found.'''
		#Retrieve current FT reading
		ft=self.get_ft(remove_bias=True)
		#Return true if hole detected, else False
		return not ((ft['force']['z']>min_spiral_force) and
                	(sqrt(ft['torque']['x']**2)<tx_limit) and
                	(sqrt(ft['torque']['y']**2)<ty_limit))

	def get_ft(self,remove_bias=True):
		'''Reads force/torque value and optionally removes biases.'''
		#Get current sensor reading
		resp=self.get_wrench_client()
		#Convert to dict
		ft_raw={'force':{'x':resp.fx,'y':resp.fy,'z':resp.fz},
				'torque':{'x':resp.tx,'y':resp.ty,'z':resp.tz}}
		if remove_bias:
			#Subtract out bias from raw data
			ft_adjusted={'force':{'x':ft_raw['force']['x']-self.ft_bias['force']['x'],
								  'y':ft_raw['force']['y']-self.ft_bias['force']['y'],
								  'z':ft_raw['force']['z']-self.ft_bias['force']['z']},
						'torque':{'x':ft_raw['torque']['x']-self.ft_bias['torque']['x'],
								  'y':ft_raw['torque']['y']-self.ft_bias['torque']['y'],
								  'z':ft_raw['torque']['z']-self.ft_bias['torque']['z']}}
			return ft_adjusted
		else:
			return ft_raw

	def get_ft_bias(self):
		'''Retrieves force/torque offset biases.'''
		resp=self.get_ave_wrench_client()
		#Convert to dict
		self.ft_bias={'force':{'x':resp.fx,'y':resp.fy,'z':resp.fz},
					  'torque':{'x':resp.tx,'y':resp.ty,'z':resp.tz}}
		return self.ft_bias

	def get_ee_pose(self):
		'''Retrieves current EE pose in Map frame.'''
		if self.gazebo:
			return self.get_pose_client()
		else:
			ee_pose=geometry_msgs.msg.Pose()
			ee_pose = self.get_tf_client("map","panda_EE_control").pose
			return ee_pose

	def move_relative(self,xyz_disp=(0.0,0.0,0.01),hold_xyz=[None,None,None],
					  goal_orientation=None,goal_time=0.5):
		'''Moves end-effector by a relative displacement in panda_link8 frame.'''
		# Convert point to map frame
		point_goal=self.transform_ee_to_map(xyz_disp)
		# Build complete pose goal
		pose_goal=geometry_msgs.msg.Pose()
		pose_goal.position = point_goal.point
		#Use specified component of point to hold constant (in map frame)
		if hold_xyz[0] is not None:
			pose_goal.position.x=hold_xyz[0]
		if hold_xyz[1] is not None:
			pose_goal.position.y=hold_xyz[1]
		if hold_xyz[2] is not None:
			pose_goal.position.z=hold_xyz[2]
		#Use goal orientation (geometry_msgs/Quaternion) if given, otherwise use current orientation
		if goal_orientation is not None:
			pose_goal.orientation = goal_orientation
		else:
			# Get current orientation
			current_pose=self.get_ee_pose()
			pose_goal.orientation = current_pose.orientation
		# Move to desired pose
		self.move_to_pose(pose_goal,goal_time)

	def move_gripper(self, width, epsilon_inner=0.05, epsilon_outer=0.05, 
					 speed=0.1, force=40):
		'''Moves gripper to the specified width.'''
		#Build goal
		goal=franka_gripper.msg.GraspGoal()
		goal.width=width
		goal.epsilon.inner=epsilon_inner
		goal.epsilon.outer=epsilon_outer
		goal.speed=speed
		goal.force=force
		#Send
		rospy.loginfo('Sending gripper goal')
		self.gripper_client.send_goal(goal)
		rospy.loginfo('Waiting for gripper result')
		self.gripper_client.wait_for_result()

	def move_home(self, goal_time=0.8):
		'''Move to home position.'''
		self.move_to_joint_pos(self.joint_home,goal_time=0.5)

	def move_to_plane(self,step,axis,max_disp,max_force,max_iter=500,
					  orientation=None,step_goal_time=0.5,recalculate_bias=True,
					  hold_xyz=[False,False,False]):
		'''Incrementally move end-effector until it hits a plane.'''
		#Set tight goal tolerances
		self.set_joint_tol(0.01)
		ii=1
		disp=0
		pose_change_step=[0,0,0]
		if axis.lower()=='x':
			idx=0
		elif  axis.lower()=='y':
			idx=1
 		elif axis.lower()=='z':
			idx=2
		else:
			idx=2
			rospy.loginfo('Invalid axis selection, defaulting to z.')
		pose_change_step[idx]=step
		#Calculate force/torque bias if necessary
		if recalculate_bias:
			self.get_ft_bias()
		force=self.get_ft(remove_bias=True)['force'][axis]
		#Use components of original point for accuracy
		orig_point=self.get_tf_client("map","panda_EE_control").pose.position
		hold_xyz_pts=[None,None,None]
		if hold_xyz[0]:
			hold_xyz_pts[0]=orig_point.x
		if hold_xyz[1]:
			hold_xyz_pts[1]=orig_point.y
		if hold_xyz[2]:
			hold_xyz_pts[2]=orig_point.z
		#Move slightly until max. force or max. iterations are exceeded.
		while (force<=max_force) and (disp<=max_disp) and (ii<=max_iter):
			self.move_relative(pose_change_step,hold_xyz=hold_xyz_pts,
							   goal_orientation=orientation,goal_time=step_goal_time)
			force=self.get_ft(remove_bias=True)['force'][axis]
			disp+=step
			ii+=1
			rospy.sleep(0.05)
		#Return to default
		self.set_joint_tol()
		if ii<=max_iter:
			rospy.loginfo('Arrived at plane.')
		else:
			rospy.loginfo('Failed to reach plane.')

	def move_to_joint_pos(self,joint_goal,goal_time=0.5):
		'''Move to joint specificed joint positions joint_goal[0:6] in rads.'''
		#Set goal time for trajectory
		self.set_goal_time(goal_time)
		# Execute
		plan=self.move_group.go(joint_goal, wait=True)
		# Stop movement, clear targets
		self.stop_and_clear()

	def move_to_pose(self,pose_goal,goal_time=0.5):
		'''Move to pose in map frame.'''
		#Set goal time for trajectory
		self.set_goal_time(goal_time)
		#Plan to Pose goal
		self.move_group.set_pose_target(pose_goal) #default for 2nd arg is current end-effector link
		if self.move_group.plan().joint_trajectory.points:  # True if trajectory contains points
 			#Execute
			self.move_group.go(wait=True)
			# Stop movement, clear targets
			self.stop_and_clear()
		else:
			rospy.logerr("Trajectory is empty. Planning was unsuccessful.")

	def rotate_relative(self,axis,angle,goal_time=0.5):
		'''Rotates end-effector relative to its current pose.'''
		orig_pose=self.get_tf_client("map","panda_EE_control").pose
		#Define step goal
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position=orig_pose.position
		current_or=orig_pose.orientation
		euler = tf.transformations.euler_from_quaternion([current_or.x,
														  current_or.y,
														  current_or.z,
														  current_or.w])
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]
		#Update orientation
		if axis=='x':
			roll+=angle
		elif axis=='y':
			pitch+=angle
		elif axis=='z':
			yaw+=angle
		#Convert back to Quat
		pose_goal.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(roll, pitch, yaw))

		self.move_to_pose(pose_goal,goal_time=goal_time)

	def rotate_to_torque(self,step_angle,axis,max_angle,max_torque,
						 max_iter=500,step_goal_time=0.5,recalculate_bias=True):
		'''Incrementally rotate end-effector until torque theshold is exceeded.'''
		ii=1
		angle=0
		#Calculate force/torque bias if necessary
		if recalculate_bias:
			self.get_ft_bias()
		torque=self.get_ft(remove_bias=True)['torque'][axis]
		orig_pose=self.get_tf_client("map","panda_EE_control").pose
		#Define step goal
		pose_goal = geometry_msgs.msg.Pose()
		pose_goal.position=orig_pose.position
		current_or=orig_pose.orientation
		euler = tf.transformations.euler_from_quaternion([current_or.x,
														  current_or.y,
														  current_or.z,
														  current_or.w])
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]
		#Move slightly until max. torque or max. iterations are exceeded.
		while (torque<=max_torque) and (abs(angle)<=abs(max_angle)) and (ii<=max_iter):
			#Update for this step
			if axis=='x':
				roll+=step_angle
			elif axis=='y':
				pitch+=step_angle
			elif axis=='z':
				yaw+=step_angle
			#Convert back to Quat
			pose_goal.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(roll, pitch, yaw))

			self.move_to_pose(pose_goal,goal_time=step_goal_time)
			torque=self.get_ft(remove_bias=True)['torque'][axis]
			angle+=step_angle
			ii+=1

		if ii<=max_iter:
			rospy.loginfo('Rotated to max. torque or max. angle.')
		else:
			rospy.loginfo('Failed to reach max. torque or max. angle.')

	def set_goal_time(self,goal_time):
		'''Sets trajectory goal time.'''
		rospy.set_param('/position_joint_trajectory_controller/constraints/goal_time',goal_time)

	def set_joint_tol(self,tol=0.05):
		'''Set global parameters related to controller for joint control.'''
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint1/goal',tol)
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint2/goal',tol)
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint3/goal',tol)
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint4/goal',tol)
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint5/goal',tol)
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint6/goal',tol)
		rospy.set_param('/position_joint_trajectory_controller/constraints/panda_joint7/goal',tol)

	def set_vel_scale(self,vel_scale):
		'''Sets velocity scaling (0,1] for motion.'''
		self.move_group.set_max_velocity_scaling_factor(vel_scale)

	def spiral_gen(self,spiral_a,spiral_b,spiral_rot,max_iter,
				   x=0.0,y=0.0):
		'''Generator for spiral motion.'''
		ii=1.0
		while ii<=max_iter:
			phi = sqrt(ii/max_iter)*(spiral_rot*2.0*pi)
			spiral_rad=(spiral_a-spiral_b*phi)
			x+=spiral_rad*cos(phi)
			y+=spiral_rad*sin(phi)
			ii+=1.0
			yield x, y

	def spiral(self,spiral_a,spiral_b,min_spiral_force,tx_limit,ty_limit,
			   spiral_rot,spiral_fd,max_iter=500,step_goal_time=0.5):
		'''Performs spiral motion until hole is detected.'''
		#Use existing bias from initial plane detection
		ii=1
		#Set tight goal tolerances
		self.set_joint_tol(0.01)
		pose_goal = geometry_msgs.msg.Pose()
		orig_pose=self.get_tf_client("map","panda_EE_control").pose
		pose_goal=orig_pose
		#Generate xy points for spiral
		for x,y in self.spiral_gen(spiral_a,spiral_b,spiral_rot,max_iter,
				   				   x=0.0,y=0.0):
			#Check if hole was found
			at_hole=self.found_hole(min_spiral_force,tx_limit,ty_limit)
			if (not at_hole) and (ii<=max_iter):
				#Move to next point
				pose_goal.position=orig_pose.position
				pose_goal.position.x+=x
				pose_goal.position.y+=y
				self.move_to_pose(pose_goal,goal_time=step_goal_time)
				ii+=1
			else:
				break
			rospy.sleep(0.08)
		#Return to default
		self.set_joint_tol()
		if ii<=max_iter:
			rospy.loginfo('Found hole.')
		else:
			rospy.loginfo('Failed to find hole.')

	def stop_and_clear(self):
		'''Stops residual motions. Clears movement targets.'''
		self.move_group.stop()
		self.move_group.clear_pose_targets()

	def transform_ee_to_map(self,point):
		'''Transforms point (x,y,z) in panda_EE_control to map frame. Returns PointStamped.'''
		# Initialize objects
		h = Header()
		orig_point=geometry_msgs.msg.PointStamped()
		#Define original point as PointStamped
		h.frame_id='panda_EE_control'
		orig_point.header=h
		orig_point.point.x=point[0]
		orig_point.point.y=point[1]
		orig_point.point.z=point[2]
		#Wait for transform to become available
		self.transformer.waitForTransform("panda_EE_control", "map", rospy.Time(0), rospy.Duration(4.0))
		#Calculate transformed point
		return self.transformer.transformPoint('map',orig_point)
		
if __name__ == '__main__':
	rospy.init_node('commander_node')
	#Instantiate commander
	panda=Commander(group_name="panda_arm",gazebo=False) #Gazebo uses "arm". Physical uses "panda_arm"
	# panda.add_object()

	#Test __repr__ method
	#print panda

	#Test move_to_pose method
	pose_goal = geometry_msgs.msg.Pose()
	goal_time = 0.5
	quat=tf_conversions.transformations.quaternion_from_euler(-1.5708, #roll
															  0.0,  #pitch
															  0.0)     #yaw
	pose_goal.orientation.x = quat[0]
	pose_goal.orientation.y = quat[1]
	pose_goal.orientation.z = quat[2]
	pose_goal.orientation.w = quat[3]
	pose_goal.position.x = 0.3
	pose_goal.position.y = 0.0
	pose_goal.position.z = 0.48
	print 'GOAL:'
	print pose_goal
	panda.move_to_pose(pose_goal,goal_time)
	rospy.sleep(5)

	#Test move_gripper method
	# panda.move_gripper(0.02)

	#Test move_to_joint_pos method
	#panda.move_to_joint_pos(panda.joint_home,goal_time=0.8)

	#Test move_home method
	# panda.move_home(goal_time=0.8)

	#Test move_gripper method
	# panda.move_gripper(0.05)

	#Test get_ee_pose method
	print panda.get_ee_pose()

	#Test move_relative method
	# panda.move_relative(xyz_disp=(0.0,0.0,0.001),goal_time=0.5)
	# rospy.sleep(0.1)
	panda.move_home(goal_time=0.8)
	# print panda.get_ee_pose()
	# print panda.move_group.get_end_effector_link()

	# panda.move_to_pose(panda.get_ee_pose(),goal_time=0.8)

	# panda.move_to_plane(step=0.0005,axis='z',max_disp=0.2,max_force=3.5,max_iter=500,
			    		# orientation=None,step_goal_time=0.5,recalculate_bias=True,hold_xyz=[False,False,False])
	# panda.move_gripper(0.05)

	#Test rotate_to_torque method
	#1deg=0.0174533rad
	# panda.rotate_to_torque(step_angle=-0.0174533,axis='z',max_angle=-1.0472,max_torque=1.0,
			   			#    max_iter=500,step_goal_time=0.5,recalculate_bias=True)

	#Test spiral method
	# panda.spiral(spiral_a=0.0,spiral_b=0.000005,min_spiral_force=-2.25,tx_limit=1.5,ty_limit=1.5,
	# 	   		 spiral_rot=10.0,spiral_fd=4.0,max_iter=500,step_goal_time=0.5)
	# panda.move_home(0.8)
