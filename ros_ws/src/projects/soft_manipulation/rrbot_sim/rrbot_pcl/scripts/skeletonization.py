#!/usr/bin/env python

from __future__ import print_function

from rrbot_pcl.srv import skelMsg, skelMsgResponse
from rrbot_pcl.srv import projectCloud

from skimage.morphology import skeletonize_3d
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from geometry_msgs.msg import Point
import rospy
import roslib; roslib.load_manifest('rrbot_pcl')

def getNeighbours(x, y, z, arr):
    nCount = -1                 # The following code includes the center point, which will add 1 always, so remove here
    for i in range(max(0, x - 1), min(x + 2, len(arr) - 1)):
        for j in range(max(0, y - 1), min(y + 2, len(arr[x]) - 1)):
            for k in range(max(0, z - 1), min(z + 2, len(arr[x,y]) - 1)):
                if arr[i, j, k]:
                    nCount += 1
    
    return nCount

# Simply calls above function with a tuple instead
def getTupleNeighbours(pt, arr):
    return getNeighbours(pt[0], pt[1], pt[2], arr)

# Returns the next adjacent neighbour, if prev is None, will return the first found, otherwise will return if !prev
def getNextNeighbour(arr, pt, prevPt=None):
    for i in range(pt[0] - 1, pt[0] + 2):
        for j in range(pt[1] - 1, pt[1]  + 2):
            for k in range(pt[2] - 1, pt[2] + 2):
                if i == pt[0] and j == pt[1] and k == pt[2]: # Skip central point
                    continue
                if arr[i, j, k] and (prevPt is None or (i, j, k) != prevPt):
                    return (i, j, k)

def getInitialPt(pts):
    for i in range(len(pts) - 1):
        for j in range(len(pts[i]) - 1):
            for k in range(len(pts[i, j]) - 1):
                if pts[i, j, k]:
                    return (i, j, k)

def getOrderedLine(pts):
    orderedPts = []
    # Get initial pt
    orderedPts.append(getInitialPt(pts))

    # Keep moving along line until we reach an endpoint
    prevPt = orderedPts[-1]
    while getTupleNeighbours(orderedPts[-1], pts) != 1:
        nextPt = getNextNeighbour(pts, orderedPts[-1], prevPt)
        prevPt = orderedPts[-1]
        orderedPts.append(nextPt)

    # Reverse order and begin moving ahead
    orderedPts.reverse()
    if len(orderedPts) == 1:
        print("First found point was an end point!")
        prevPt = getNextNeighbour(pts, orderedPts[-1], prevPt)
    else:
        prevPt = orderedPts[-2]
    while getTupleNeighbours(orderedPts[-1], pts) != 1:
        nextPt = getNextNeighbour(pts, orderedPts[-1], prevPt)
        prevPt = orderedPts[-1]
        orderedPts.append(nextPt)

    return orderedPts

# Services
projection = None

def handle_skel_msg(req):
    print("Received skeletonization request")
    
    global projection

    # Setting leafsize
    res = 0.001
    cloud = req.pc2
    # Calling projection service
    data = projection(cloud, res)

    print("Received projected cloud")

    # Converting projected cloud to np array
    x = np.array([])
    y = np.array([])
    z = np.array([])

    for i in range(2, len(data.x)):
        x = np.append(x, round(data.x[i], 5))
        y = np.append(y, round(data.y[i], 5))
        z = np.append(z, round(data.z[i], 5))
    xOffset = x.min()
    yOffset = y.min()
    zOffset = z.min()

    x -= xOffset
    x = x.round(5)/res
    y -= yOffset
    y = y.round(5)/res
    z -= zOffset
    z = z.round(5)/res

    skelArr = np.zeros((int(x.max()) + 2, int(y.max()) + 2, int(z.max()) + 2), dtype='bool')
    for i in range(0, len(x)):
        skelArr[int(round(x[i])), int(round(y[i])), int(round(z[i]))] = True

    #Skeletonize
    print("Skeletonizing")
    skeleton = (skeletonize_3d(skelArr))
    x,y,z = skelArr.nonzero()
    skx = np.array([])
    sky = np.array([])
    skz = np.array([])
    print("Done")
    for i in range(int(x.max()) + 1):
        for j in range(int(y.max()) + 1):
            for k in range(int(z.max()) + 1):
                if skeleton[i, j, k]:
                    skx = np.append(skx, i)
                    sky = np.append(sky, j)
                    skz = np.append(skz, k)

    skx = skx*res
    skx += xOffset
    sky = sky*res
    sky += yOffset
    skz = skz*res
    skz += zOffset

    print("Plotting skeleton")
    if req.view:
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        # ax.scatter(data.x, data.y, data.z, c='#FF0000', alpha=0.01)
        ax.scatter(skx, sky, skz, c='#00FF00', alpha=1.0)
        ax.set_title('skeleton')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.scatter(oX, oY, oZ, c=cp)
        plt.show()

    return skelMsgResponse()

def skeletonizer():
    global projection

    rospy.init_node('skeletonization_node')
    projection = rospy.ServiceProxy('project_cloud', projectCloud)
    skeletonizationServer = rospy.Service('skeletonization', skelMsg, handle_skel_msg)
    print("Skeletonization Service Ready")
    rospy.spin()

if __name__ == "__main__":
    skeletonizer()