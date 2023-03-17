#!/usr/bin/env python3

import numpy as np

minAngle = 20*(np.pi/180.0);

def sphericalToCartesian(pose,centre=[0,0,0]):
    res = [0,0,0]
    res[0] = centre[0]+pose[0]*np.sin(pose[2])*np.cos(pose[1])
    res[1] = centre[1]+pose[0]*np.sin(pose[2])*np.sin(pose[1])
    res[2] = centre[2]+pose[0]*np.cos(pose[2])
    return np.asarray(res)

def cartesianToSpherical(pose, centre=[0,0,0]):
  res = [0,0,0]
  res[0] = np.sqrt(pow(pose[0]-centre[0],2)+pow(pose[1]-centre[1],2)+pow(pose[2]-centre[2],2))
  res[1] = np.arctan2(pose[1]-centre[1],pose[0]-centre[0])
  res[2] = np.arctan2(np.sqrt(pow(pose[0]-centre[0],2)+pow(pose[1]-centre[1],2)),pose[2]-centre[2])
  return np.asarray(res)

def calcExplorationPoseB(startPose, dir):
  xyPlane = np.array([0,0,1]);

  stPoint = sphericalToCartesian(startPose);

  zAxis = stPoint; zAxis = zAxis/np.linalg.norm(zAxis);
  xAxis = np.cross(zAxis,xyPlane); xAxis /= np.linalg.norm(xAxis);
  yAxis = np.cross(zAxis,xAxis)

  switcher = {
        1: [ 1, 0],
        2: [ 1,-1],
        3: [ 0,-1],
        4: [-1,-1],
        5: [-1, 0],
        6: [-1, 1],
        7: [ 0, 1],
        8: [ 1, 1],
    }

  ratio = switcher.get(dir,[0,0])

  rotAxis = ratio[0]*xAxis + ratio[1]*yAxis; rotAxis /= np.linalg.norm(rotAxis)
  rotAxis.shape = (3,1)
  matA = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])
  matB = np.array([[0,-rotAxis[2],rotAxis[1]],
                   [rotAxis[2],0,-rotAxis[0]],
                   [-rotAxis[1],rotAxis[0],0]])
  matC = np.matmul(rotAxis,np.transpose(rotAxis))

  tempMat = np.cos(minAngle)*matA + np.sin(minAngle)*matB + (1-np.cos(minAngle))*matC
  endPoint = np.matmul(tempMat,stPoint)

  end = cartesianToSpherical(endPoint);

  # Polar angle 0 to 90 degree
  if(end[2] < 0):
    end[2] = -1*end[2]
    end[1] = end[1] + np.pi

  # Azhimuthal angle 0 to 360 degree
  end[1] = np.fmod(end[1],2*np.pi)
  if(end[1] < 0):
      end[1] += 2*np.pi

  return end.ravel()

def findDirection(start, end):
    diff = []
    for i in range(1,9):
        temp = np.asarray(calcExplorationPoseB(start,i))
        tempDiff = abs(end[0]-temp[0]) + \
                   min(abs(end[1]-temp[1]),abs(abs(end[1]-temp[1])-2*np.pi)) + \
                   min(abs(end[2]-temp[2]),abs(abs(end[2]-temp[2])-2*np.pi))
        diff.append(round(tempDiff,2))
    minimum = np.amin(diff)
    idx = np.argmin(diff)
    if(minimum != 0):
        return -1
    else:
        return idx+1
