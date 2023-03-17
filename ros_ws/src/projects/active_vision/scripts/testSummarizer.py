#!/usr/bin/env python3

import sys, csv, os, copy
import numpy as np
import matplotlib.pyplot as plt
from summarizerDataCollected import readInput

'''Ugly code that calculates the first direction taken, assuming a direction has been taken- giving it a 0 length path will return -1.'''
def calcFirstDirection(each):
    directions = {'N':1,'NE':2,'E':3,'SE':4,'S':5,'SW':6,'W':7,'NW':8}
    iRoll = np.degrees(float(each[15]))
    iPitch = np.degrees(float(each[16]))
    fRoll = np.degrees(float(each[18]))
    fPitch = np.degrees(float(each[19]))
    dRoll = fRoll-iRoll
    dPitch = fPitch-iPitch
    epsilon = .1
    #Strictly E/W
    if abs(dPitch) < epsilon:
        if abs(dRoll) < epsilon:
            print("Error, unreachable statement")
            return -1
        elif dRoll < 0:
            return directions['E']
        elif dRoll > 0:
            return directions['W']
    #North
    elif dPitch < 0:
        if abs(dRoll) < epsilon:
            return directions['N']
        elif dRoll < 0:
            return directions['NE']
        elif dRoll > 0:
            return directions['NW']
    #South
    elif dPitch > 0:
        if abs(dRoll) < epsilon:
            return directions['S']
        elif dRoll < 0:
            return directions['SE']
        elif dRoll > 0:
            return directions['SW']
    else:
        print("Error, unreachable statement")
        return -1

'''Read file and summarize information about 1st directions chosen
summary = [[objName, objRoll, objPitch, [directionCounts]]]'''
def genSummary(fileName):
    bins = []
    directions = 9
    for bin in range(directions):
        bins.append(0)
    #Finding all unique configurations
    data = readInput(fileName)
    summary = []
    unique_list = []
    for each in data:
        param = each[0]+"-"+each[1]+"-"+each[2]
        if param not in unique_list:
            unique_list.append(param)
            summary.append([each[0],float(each[1]),float(each[2]),copy.deepcopy(bins)])
        index = unique_list.index(param)
        nSteps = (len(each)-14)/3-1
        if (nSteps <= 0):
            summary[index][3][0] += 1
        else:
            direction = calcFirstDirection(each)
            summary[index][3][direction] += 1
    return summary

def addArrow(chart, name, offX, offY, fraction):
    chart.annotate("", xy=(.5+offX, .5+offY), xytext=(0.5,0.5), arrowprops=dict(headlength=3, headwidth=6, width=0+(5*fraction)))
    ha = 'center'
    va = 'center'
    if(offX > 0):
        ha = 'left'
    elif(offX < 0):
        ha = 'right'
    if(offY > 0):
        va = 'bottom'
    elif(offY < 0):
        va = 'top'
    chart.annotate("{}-{}%".format(name, str(fraction*100)), xy=(.5+offX, .5+offY), horizontalalignment=ha, verticalalignment=va)

#Generate a bar graph of the summary
def graphSummary(summary):
    # Extracting different objects
    unique_list = []
    nUnique = []
    for each in summary:
        if each[0] not in unique_list:
            unique_list.append(each[0])
            nUnique.append(1)
        else:
            nUnique[unique_list.index(each[0])]+=1

    # Plotting each object in a different graph
    for obj,nPlots in zip(unique_list,nUnique):
        rows = min(3,nPlots); cols = int((nPlots-1)/3)+1
        fig, ax = plt.subplots(rows,cols); ax = np.ravel(ax)
        fig.set_size_inches(cols*4, 4*rows)
        fig.suptitle(obj+' - 1st step directions', fontsize='medium')
        idx = 0
        for each in summary:
            if each[0] == obj:
                totalPoints = 0
                for item in each[3]:
                    totalPoints+=item
                frac0 = each[3][0]/totalPoints
                centerScale = 0.02*(1.0+frac0)
                c = plt.Circle((.5,.5),centerScale,color=(0,0,0))
                ax[idx].add_artist(c)
                ax[idx].annotate("{}%".format(str(frac0*100)), (0.6,0.52))
                directions = {'N':1,'NE':2,'E':3,'SE':4,'S':5,'SW':6,'W':7,'NW':8}
                addArrow(ax[idx], 'N', 0, .4, each[3][1]/totalPoints)
                addArrow(ax[idx], 'NE', .22, .22, each[3][2]/totalPoints)
                addArrow(ax[idx], 'E', 0.4, 0, each[3][3]/totalPoints)
                addArrow(ax[idx], 'SE', .22, -.22, each[3][4]/totalPoints)
                addArrow(ax[idx], 'S', 0, -.4, each[3][5]/totalPoints)
                addArrow(ax[idx], 'SW', -.22, -.22, each[3][6]/totalPoints)
                addArrow(ax[idx], 'W', -.4, 0, each[3][7]/totalPoints)
                addArrow(ax[idx], 'NW', -.22, .22, each[3][8]/totalPoints)

                ax[idx].set_xticks([])
                ax[idx].set_yticks([])


        #fig.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'],loc='lower center',ncol=2,fontsize='x-small')
        # fig.legend()
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    s = genSummary('/home/diyogon/Documents/WPI/Berk_Lab/mer_lab/ros_ws/src/projects/active_vision/dataCollected/testData/HEURISTIC_Prism_truncated_stateVec_dataRec.csv')
    graphSummary(s)
    print(s)
