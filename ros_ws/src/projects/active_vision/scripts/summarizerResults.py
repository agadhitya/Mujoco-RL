#!/usr/bin/env python3

import sys, csv,os, copy, io
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from collections import OrderedDict
from summarizerDataCollected import readInput,fig2img,plots2jpg
from toolViewPointCalc import findDirection
from PIL import Image
import rospkg

baseDir = rospkg.RosPack().get_path('active_vision')
BFSDir = baseDir+"/misc/BFS_Results/"
BFSObjs = []
heuristicPolicies = ["Heuristic","3D_Heuristic","Random","Brick","BFS"]
maxSteps = 5

graphColors =  ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
graphMarkers = ["o", "v", "^", "<", ">", "*", "+", "s"]

def calcStepsDir(each):
    bin = [0]*maxSteps
    i = 14
    step = 1
    while i < len(each)-3 and step <= 5 and each[i+3] != '':
        st = [float(each[i]),float(each[i+1]),float(each[i+2])]
        end = [float(each[i+3]),float(each[i+4]),float(each[i+5])]
        dir = findDirection(st,end)
        if dir == -1:
            print ("Error")
        else:
            bin[step-1] = dir
        i += 3
        step += 1
    return bin

def updateSummary(policyWise, policyWise2, obj):
    if obj in BFSObjs:
        pass
    else:
        BFSObjs.append(obj)
        BFSData = readInput(BFSDir + "BFS_results.csv")
        for each in BFSData:
            obj2 = each[0]+"&"+each[1]+"&"+each[2]
            if(obj==obj2):
                keyPolicy = obj2+"*BFS"
                nSteps = int(each[13])
                temp = calcStepsDir(each)
                policyWise[keyPolicy][nSteps] += 1
                for step,dir in zip(list(range(len(temp))),temp):
                    if dir != 0:
                        policyWise2[keyPolicy][step][dir-1] += 1

'''Read file and summarize information about differnt experiments.
'''
def genSummary(path,fileNames):
    # Bins for storing the number of data points with x steps for successful grasp
    # 0,1,2,3...max,>max
    bins = [0]*(maxSteps+2)

    bin2 = []
    for i in range(maxSteps):
        bin2.append([0,0,0,0,0,0,0,0])

    # Setting up the dictionaries which will store the summaries
    policyWise = {}
    policyWise2 = {}
    stVecWise = {}
    for i in range(len(fileNames)):
        file = fileNames[i]
        data = readInput(path+file)
        policy = file.split(":")[0]
        stVec = file.split(":")[1]

        # Extracting the unique objects
        objects = []
        for each in data:
            obj = each[0]+"&"+each[1]+"&"+each[2]
            objects.append(obj)
        objects = list(dict.fromkeys(objects))

        # print(policy,stVec,objects)
        for obj in objects:
            keyPolicy = obj+"*"+policy
            keyStVec = obj+"*"+stVec
            if policy in heuristicPolicies:
                policyWise[keyPolicy] = np.array(bins)
                policyWise2[keyPolicy] = np.array(bin2)
            else:
                if (keyPolicy in policyWise) == False:
                    policyWise[keyPolicy] = {}
                    policyWise2[keyPolicy] = {}
                policyWise[keyPolicy][stVec] = np.array(bins)
                policyWise2[keyPolicy][stVec] = np.array(bin2)
                policyWise[obj+"*BFS"] = np.array(bins)
                policyWise2[obj+"*BFS"] = np.array(bin2)
                if (keyStVec in stVecWise) == False:
                    stVecWise[keyStVec] = {}
                stVecWise[keyStVec][policy] = np.array(bins)

    # Updating the generated summary
    for file in fileNames:
        data = readInput(path+file)
        policy = file.split(":")[0]
        stVec = file.split(":")[1]

        for each in data:
            obj = each[0]+"&"+each[1]+"&"+each[2]
            keyPolicy = obj+"*"+policy
            keyStVec = obj+"*"+stVec
            nSteps = min((len(each)-14)/3-1,maxSteps+1)
            temp = calcStepsDir(each)
            if policy in heuristicPolicies:
                policyWise[keyPolicy][nSteps] += 1
                for step,dir in zip(list(range(len(temp))),temp):
                    if dir != 0:
                        policyWise2[keyPolicy][step][dir-1] += 1
            else:
                updateSummary(policyWise, policyWise2, obj)
                policyWise[keyPolicy][stVec][nSteps] += 1
                stVecWise[keyStVec][policy][nSteps] += 1
                for step,dir in zip(list(range(len(temp))),temp):
                    if dir != 0:
                        policyWise2[keyPolicy][stVec][step][dir-1] += 1

    # for k, v in policyWise.items():
    #     print '%60s' % k, ' : ', v
    # print("-------")
    # for k, v in policyWise2.items():
    #     print k, ' : \n', v
    # print("-------")
    # for k, v in stVecWise.items():
    #     print '%60s' % k, ' : ', v
    # print("----------------")
    policyWise =  OrderedDict(sorted(policyWise.items()))
    policyWise2 =  OrderedDict(sorted(policyWise2.items()))
    stVecWise =  OrderedDict(sorted(stVecWise.items()))
    # print(policyWise)
    # print(policyWise2)
    # print(stVecWise)
    return policyWise,policyWise2,stVecWise

#Generate a bar graph of the summary
def graphSummary(path,policyWise,policyWise2,stVecWise):

    xAxis = [str(i) for i in range(maxSteps+1)]

    # Creating stVecWise plots i.e. policy comparisons
    plots = []
    for key, value in list(stVecWise.items()):
        keywords = key.split("*")
        fig, ax = plt.subplots(1,1); ax = np.ravel(ax)
        fig.set_size_inches(6, 4)
        fig.suptitle("Step number distribution - Comparison of various policies", fontsize='medium')
        idx = 0
        # print key, value
        for policy, summary in list(value.items()):
            temp = np.cumsum(summary)
            nData = temp[-1]
            avg = round(np.sum(np.array(summary)*np.array(list(range(len(summary)))))/float(nData),1)
            temp = temp / float(temp[-1])
            # print "\t", policy, "--->" ,temp
            ax[0].plot(xAxis,temp[0:maxSteps+1]*100,color=graphColors[idx], marker=graphMarkers[idx], label = policy + " (" + str(avg) + ")")
            idx = idx + 1

        # Adding the heuristic result if available
        for policy in heuristicPolicies:
            keyPolicy = keywords[0]+"*"+policy
            if keyPolicy in policyWise:
                temp = np.cumsum(policyWise[keyPolicy])
                nData = temp[-1]
                avg = round(np.sum(np.array(policyWise[keyPolicy])*np.array(list(range(len(policyWise[keyPolicy])))))/float(nData),1)
                temp = temp / float(temp[-1])
                # print "\t", policy, "--->" ,temp
                ax[0].plot(xAxis,temp[0:maxSteps+1]*100,color=graphColors[idx], marker=graphMarkers[idx], label = policy + " (" + str(avg) + ")")
                idx = idx + 1

        objDetails = keywords[0].split("&")
        titleStr = "{} : Roll - {} , Pitch - {} \n State Vec : {}".format(
                    objDetails[0],
                    str(int(round(np.degrees(float(objDetails[1])),0))),
                    str(int(round(np.degrees(float(objDetails[2])),0))),
                    keywords[1])
        ax[0].set_title(titleStr , fontsize='small')

        ax[0].grid(axis='y')
        ax[0].set_ylabel('Success percentage', fontsize='small')
        ax[0].set_ylim(0,110)

        ax[0].set_xlabel('Step number that leads to a successful grasp (Avg. steps in legend)', fontsize='small')

        ax[0].legend(fontsize='small')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plots.append(fig2img(fig))

    plots2jpg(plots,path+"policyComparison_",1)

    # Creating policyWise plots i.e. state vector comparisons
    plots = []
    for key, value in list(policyWise.items()):
        # Skipping the heuristics
        if isinstance(value,dict) == False:
            continue
        # Skipping if only one state vector is there
        if len(value) < 2:
            continue

        keywords = key.split("*")
        fig, ax = plt.subplots(1,1); ax = np.ravel(ax)
        fig.set_size_inches(6, 4)
        fig.suptitle("Step number distribution - Comparison of various state vectors", fontsize='medium')
        idx = 0

        for stVec, summary in list(value.items()):
            temp = np.cumsum(summary)
            nData = temp[-1]
            avg = round(np.sum(np.array(summary)*np.array(list(range(len(summary)))))/float(nData),1)
            temp = temp / float(temp[-1])
            # print "\t", policy, "--->" ,temp
            ax[0].plot(xAxis,temp[0:maxSteps+1]*100,color=graphColors[idx], marker=graphMarkers[idx], label = stVec + " (" + str(avg) + ")")
            idx = idx + 1

        objDetails = keywords[0].split("&")
        titleStr = "{} : Roll - {} , Pitch - {} \n Policy : {}".format(
                    objDetails[0],
                    str(int(round(np.degrees(float(objDetails[1])),0))),
                    str(int(round(np.degrees(float(objDetails[2])),0))),
                    keywords[1])
        ax[0].set_title(titleStr , fontsize='small')

        ax[0].grid(axis='y')
        ax[0].set_ylabel('Success percentage', fontsize='small')
        ax[0].set_ylim(0,110)

        ax[0].set_xlabel('Step number that leads to a successful grasp (Avg. steps in legend)', fontsize='small')

        ax[0].legend(fontsize='small')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plots.append(fig2img(fig))

    plots2jpg(plots,path+"stVecComparison_",1)

    # Heuristics only comparison
    plots = []
    unique = []
    for key, value in list(policyWise.items()):
        # Skipping non heuristics
        if isinstance(value,dict):
            continue

        keywords = key.split("*")
        if keywords[0] not in unique:
            unique.append(keywords[0])

            fig, ax = plt.subplots(1,1); ax = np.ravel(ax)
            fig.set_size_inches(6, 4)
            fig.suptitle("Step number distribution - Comparison of various heuristics", fontsize='medium')
            idx = 0

            # Adding the heuristic result if available
            for policy in heuristicPolicies:
                keyPolicy = keywords[0]+"*"+policy
                if keyPolicy in policyWise:
                    temp = np.cumsum(policyWise[keyPolicy])
                    nData = temp[-1]
                    avg = round(np.sum(np.array(policyWise[keyPolicy])*np.array(list(range(len(policyWise[keyPolicy])))))/float(nData),1)
                    temp = temp / float(temp[-1])
                    # print "\t", policy, "--->" ,temp
                    ax[0].plot(xAxis,temp[0:maxSteps+1]*100,color=graphColors[idx], marker=graphMarkers[idx], label = policy + " (" + str(avg) + ")")
                    idx = idx + 1

            objDetails = keywords[0].split("&")
            titleStr = "{} : Roll - {} , Pitch - {}".format(
                        objDetails[0],
                        str(int(round(np.degrees(float(objDetails[1])),0))),
                        str(int(round(np.degrees(float(objDetails[2])),0))))
            ax[0].set_title(titleStr , fontsize='small')

            ax[0].grid(axis='y')
            ax[0].set_ylabel('Success percentage', fontsize='small')
            ax[0].set_ylim(0,110)

            ax[0].set_xlabel('Step number that leads to a successful grasp (Avg. steps in legend)', fontsize='small')

            ax[0].legend(fontsize='small')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            plots.append(fig2img(fig))

    plots2jpg(plots,path+"heuristicComparison_",1)

    # Stepwise direction radar plots
    plots = []
    theta = list(np.linspace(0,np.pi/4*7,8))
    theta.append(0)
    for key, value in list(policyWise2.items()):
        keywords = key.split("*")
        objDetails = keywords[0].split("&")

        fig = plt.figure()
        fig.set_size_inches(5, 5)
        titleStr = "{} : Roll - {} , Pitch - {}".format(
                    objDetails[0],
                    str(int(round(np.degrees(float(objDetails[1])),0))),
                    str(int(round(np.degrees(float(objDetails[2])),0))))
        fig.suptitle("Test Data - % of time a direction was taken\n"+
                     titleStr, fontsize='medium')
        ax=plt.subplot(1, 1, 1, projection='polar')

        # Non-Heuristic
        if isinstance(value,dict):
            for stVec, summary in list(value.items()):
                ax.set_title("Policy : "+keywords[1]+", St Vec : "+stVec, fontsize='small')
                ax.set_theta_direction(-1)
                ax.set_theta_offset(np.pi/2)
                ax.set_thetagrids(np.degrees(theta), ["N","NE","E","SE","S","SW","W","NW","N"])
                for step,i in zip(summary,list(range(len(summary)))):
                    if i == 3:
                        break
                    if(np.sum(step) > 0):
                        sum = float(np.sum(step))
                        temp = list(np.array(step)/sum*100)
                        temp.append(temp[0])
                        ax.plot(theta,temp,color = graphColors[i],label="Step "+str(i+1)+" ("+str(int(sum))+")")
                        ax.fill(theta,temp, facecolor=graphColors[i], alpha=0.25)
                        ax.set_rmin(0); ax.set_rmax(100)
                fig.legend(loc="lower right")
                fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                plots.append(fig2img(fig))
                ax.cla()
        else:
            ax.set_title("Policy : "+keywords[1], fontsize='small')
            ax.set_theta_direction(-1)
            ax.set_theta_offset(np.pi/2)
            ax.set_thetagrids(np.degrees(theta), ["N","NE","E","SE","S","SW","W","NW","N"])
            for step,i in zip(value,list(range(len(value)))):
                if i == 3:
                    break
                if(np.sum(step) > 0):
                    sum = float(np.sum(step))
                    temp = list(np.array(step)/sum*100)
                    temp.append(temp[0])
                    ax.plot(theta,temp,color = graphColors[i],label="Step "+str(i+1)+" ("+str(int(sum))+")")
                    ax.fill(theta,temp, facecolor=graphColors[i], alpha=0.25)
                    ax.set_rmin(0); ax.set_rmax(100)
            fig.legend(loc="lower right")
            fig.tight_layout(rect=[0, 0.03, 1, 0.90])
            plots.append(fig2img(fig))
            ax.cla()

    plots2jpg(plots,path+'Radar_',1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(("Incorrect number of arguments ", len(sys.argv)))
        sys.exit()

    path = sys.argv[1]
    print(BFSDir)

    reqCSVs = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(":dataRec.csv"):
               reqCSVs.append(file)

    policyWise,policyWise2,stVecWise = genSummary(path,reqCSVs)
    graphSummary(path,policyWise,policyWise2,stVecWise)
