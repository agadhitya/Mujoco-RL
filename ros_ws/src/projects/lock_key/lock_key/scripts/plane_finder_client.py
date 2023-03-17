#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS node for calling plane finder service. 
Publishes point clouds, lock orientation, and lock plane coefficients.
"""
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from lock_key_msgs.msg import PlaneCoefficients
from std_msgs.msg import Float64
import lock_key_msgs.srv
from math import asin, atan2

class CloudClient:
    '''Calls Find Planes Service.'''
    def __init__(self):
        self.pc2_sub=rospy.Subscriber('/camera/depth/color/points',
                                      PointCloud2,self.pc2_callback,
                                      queue_size=1)
        self.key_cloud_pub = rospy.Publisher('key_cloud', PointCloud2, 
                                             queue_size=1)
        self.lock_cloud_pub = rospy.Publisher('lock_cloud', PointCloud2, 
                                              queue_size=1)
        self.lock_cloud_plane_pub = rospy.Publisher('lock_plane_cloud', PointCloud2, 
                                                    queue_size=1)
        self.lock_coeffs_pub = rospy.Publisher('lock_coeffs', PlaneCoefficients, 
                                                queue_size=1)
        self.lock_pitch_pub = rospy.Publisher('lock_pitch', Float64, 
                                                queue_size=1)
        self.lock_yaw_pub = rospy.Publisher('lock_yaw', Float64, 
                                                queue_size=1)
        rospy.loginfo('Running plane finder client.')

    def call_find_planes(self,pc2_msg):
        rospy.loginfo('Waiting for FindPlanesServer.')
        rospy.wait_for_service('FindPlanesServer')
        rospy.loginfo('Found FindPlanesServer.')
        find_planes = rospy.ServiceProxy('FindPlanesServer', 
                                         lock_key_msgs.srv.FindPlanes)
        response = find_planes(pc2_msg)

        #Identify and print the min/maxes of XYZ
        key_xs=[]; key_ys=[]; key_zs=[]
        lock_xs=[]; lock_ys=[]; lock_zs=[]
        #Read in point cloud XYZ locations.
        #TODO: Do this faster. Vector operations or something
        for p in pc2.read_points(response.key_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            key_xs.append(p[0])
            key_ys.append(p[1])
            key_zs.append(p[2])
        for p in pc2.read_points(response.lock_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            lock_xs.append(p[0])
            lock_ys.append(p[1])
            lock_zs.append(p[2])

        #Identify min/max bounds of the key and lock point clouds so that we
        #know where to perform color segmentation.
        bounds={'key':{'min':{'x':min(key_xs),'y':min(key_ys),'z':min(key_zs)},
                       'max':{'x':max(key_xs),'y':max(key_ys),'z':max(key_zs)}},
                'lock':{'min':{'x':min(lock_xs),'y':min(lock_ys),'z':min(key_zs)},
                        'max':{'x':max(lock_xs),'y':max(lock_ys),'z':max(lock_zs)}},
                'frame':'/camera_depth_optical_frame'}
        #Set as parameter so that it is globally available
        rospy.set_param('bounds',bounds)
        
        #Publish point clouds
        try:
            self.key_cloud_pub.publish(response.key_cloud)
            self.lock_cloud_pub.publish(response.lock_cloud)
            self.lock_cloud_plane_pub.publish(response.lock_plane_cloud)
            plane_coeffs=self.convert_plane_coeffs(response.lock_plane_coeffs)
            #Calculate plane orientation from plane coeffs.
            self.lock_pitch_pub.publish(Float64(asin(-plane_coeffs.b)))
            self.lock_yaw_pub.publish(Float64(atan2(plane_coeffs.a,plane_coeffs.c)))

            self.lock_coeffs_pub.publish(plane_coeffs)
        except:
            print 'Error publishing key or lock point cloud'
        rospy.sleep(0.2)

    def convert_plane_coeffs(self, coeffs_msg):
        '''Converts plane coefficients to standard plane formula format.'''
        coeffs=PlaneCoefficients()
        coeffs.a=coeffs_msg.a/coeffs_msg.d
        coeffs.b=coeffs_msg.b/coeffs_msg.d
        coeffs.c=coeffs_msg.c/coeffs_msg.d
        coeffs.d=0.0
        return coeffs

    def pc2_callback(self,pc2_msg):
    	'''PointCloud2 Message Callback passthrough function.'''
        # try:
        rospy.loginfo('Received pointcloud2 message.')
        self.call_find_planes(pc2_msg)
        # except:
        #     print 'Error calling find planes'
        
def main():
    rospy.init_node('plane_finder_client',anonymous=True)
    cloud_client=CloudClient()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')

if __name__=='__main__':
    main()
