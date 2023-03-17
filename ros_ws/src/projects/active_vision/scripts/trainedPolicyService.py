#!/usr/bin/env python3

import trainingPolicy as TP
import sys, csv, time, random, math, os, rospkg, rospy
import numpy as np
from os import path
from matplotlib import pyplot as plt
from active_vision.srv import trainedPolicySRV,trainedPolicySRVResponse
from Q_controller2 import *

model = None

def helpDisp(text):
    print(text)
    print('\n-----Policy Training Help-----\n')
    print('Arguments : [StateVec]')
    print('StateVec : StateVec CSV file name')
    print('\n-----End Help-----\n')
    sys.exit()

#Takes a state vector input and returns a predicted direction
def predictionServer(req):
    global model
    stateVec = np.asarray(req.stateVec.data)
    stateVec = stateVec.reshape((1,stateVec.shape[0]))
    predDir = model.predict(stateVec)
    rospy.loginfo("Service called. Direction -> "+str(predDir))
    return trainedPolicySRVResponse(direction=predDir)

if __name__ == "__main__":

    os.chdir(os.path.expanduser("~"))

    # Load parameters from yaml
    rospy.init_node('trainedPolicyServer')
    type = rospy.get_param("/active_vision/policyTester/policy")
    PCA_components = rospy.get_param("/active_vision/policyTester/PCAcomponents")

    # State vector location
    csvStVecDir = rospkg.RosPack().get_path('active_vision') + rospy.get_param("/active_vision/policyTester/csvStVecDir")
    csvStVec = rospy.get_param("/active_vision/policyTester/csvStVec")
    HAFstVecGridSize = rospy.get_param("/active_vision/policyTester/HAFstVecGridSize")

    # stateVec=List of all raw states
    # dirVec=List of the direction taken for each state
    # directions=N(1),NE(2),E(3),SE(4),S(5),SW(6),W(7),NW(8)
    stateVec, dirVec = TP.readHAFStVec(csvStVecDir+csvStVec, (pow(HAFstVecGridSize,2)*2)+2)
    dirVec = dirVec.ravel()

    if type == "PCA_LDA_LR":
        model = TP.PCA_LDA_LR_Model(PCA_components)
    elif type == "PCA_LDA":
        model = TP.PCA_LDA_Model(PCA_components)
    elif type == "PCA_LR":
        model = TP.PCA_LR_Model(PCA_components)
    elif type == "QLEARN":
        model = TP.Q_Model((pow(HAFstVecGridSize,2)*2)+2)
    elif type == "RANDOM":
        model = TP.Random_Model()
    elif type == "BRICK":
        model = TP.Brick_Model()

    model.train(stateVec, dirVec)

    s = rospy.Service('/active_vision/trained_policy', trainedPolicySRV, predictionServer)
    rospy.loginfo(type+" policy service ready.")
    rospy.spin()
