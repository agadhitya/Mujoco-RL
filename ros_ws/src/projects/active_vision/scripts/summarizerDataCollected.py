#!/usr/bin/env python3

import sys, csv,os, copy, io
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from PIL import Image

# Y-limit for histogram
minYaw = 0
maxYaw = 180
maxSteps = 5
nStepsSplit = 3

#Convert a Matplotlib figure to a PIL Image and return it
def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

#Convert the plots to a jpg format
def plots2jpg(plots,path,type):
    if len(plots) > 0:
        dims = plots[0].size
        nFigures = np.ceil(len(plots)/6.0)
        nDone = 0
        ctr = 1
        for figID in range(len(plots)):
            idx = figID % 6
            if idx == 0:
                remainingPlots = len(plots) - nDone
                if type == 1:
                    rows = min(2,remainingPlots)
                    cols = min(3,(remainingPlots+1)/2)
                else:
                    rows = min(3,remainingPlots)
                    cols = min(2,(remainingPlots+1)/3)
                newImg = Image.new('RGB', (cols*dims[0], rows*dims[1]))
            if type == 1:
                newImg.paste(plots[figID],((idx/2)%3*dims[0],idx%2*dims[1]))
            else:
                newImg.paste(plots[figID],((idx/3)%2*dims[0],idx%3*dims[1]))
            nDone += 1
            if idx == 5 or figID == len(plots) - 1:
                newPath = path+str(ctr)+".jpg"
                newImg.save(newPath)
                print("Graph saved to : "+newPath[newPath.rfind('/')+1:])
                ctr += 1

#Function to read the input data file
def readInput(fileName):
    data = []
    with open(fileName) as csv_file:
        # Reading the csv file
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            # Going through it line by line as storing it in a list
            data.append([i for i in line[:]])

    return np.asarray(data)

#Read file and plot camera angle versus initial direction chosen. TODO:
# Move to a separate file? Refactor for more general data analysis?
def PCASummary(fileName):
    data = readInput(fileName)
    plt.figure(figsize=(10,9))
    plt.xlabel('horizontal',fontsize=20)
    plt.ylabel('azimuthal',fontsize=20)
    plt.title("Prism_10x8x4_Pose_1.57x1.57 Direction analysis",fontsize=20)
    h = []
    a = []
    d = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    labels = ['N','NE','E','SE','S','SW','W','NW']
    for i in range(len(data)):
        each = data[i]
        if(each[1]==each[2]):
            nSteps = (len(each)-14)/3-1
            for step in range(nSteps):
                offset = step*3
                h.append(np.degrees(float(each[15+offset])))
                a.append(np.degrees(float(each[16+offset])))
                d.append(int(each[12])-1)
    h = np.array(h)
    a = np.array(a)
    d = np.array(d)
    for j in range(8):
        target_index = d == j
        plt.scatter(h[target_index], a[target_index], c=colors[j], label=labels[j])
    plt.legend()
    plt.show()

'''Read file and summarize information about each interval of camera data.
summary = [[objName, objRoll, objPitch,
            [YawData],
            [Min Steps for 0-360],
            [# yaw values with 0,1,... steps],
            [Direction chosen for each step]]]
YawData = [yawEndAngle, #ofDataPoints, minSteps, avgSteps, maxSteps, #stepsBelownStepsSplit, #stepsAbovenStepsSplit, [dataPoints]]'''
def genSummary(fileName,nBins):
    bins = []
    for bin in range(nBins):
        bins.append([(bin+1)*360/nBins,0,0,0,0,0,0,[]])

    # Min step for each yaw angle
    bin2 = [99]*360

    # Number of steps for a graph (calculated using bin2 values)
    bin3 = [0]*(maxSteps+2)

    bin4 = []
    for i in range(maxSteps):
        bin4.append([0,0,0,0,0,0,0,0])

    #Finding all unique configurations
    data = readInput(fileName)
    summary = []
    unique_list = []
    for each in data:
        param = each[0]+"-"+each[1]+"-"+each[2]
        if param not in unique_list:
            unique_list.append(param)
            summary.append([each[0],float(each[1]),float(each[2]),copy.deepcopy(bins),copy.deepcopy(bin2),copy.deepcopy(bin3),copy.deepcopy(bin4)])
        index = unique_list.index(param)
        nSteps = (len(each)-14)/3-1
        direction = int(each[12])
        whichStepData = int(each[10])
        summary[index][6][whichStepData-1][direction-1] += 1

        yaw = int(round(np.degrees(float(each[3]))))
        binIndex = int(yaw/(360/nBins))

        summary[index][4][yaw] = min(nSteps,summary[index][4][yaw])

        summary[index][3][binIndex][1] += 1
        summary[index][3][binIndex][3] += nSteps  # Will be dividived my nData in the end
        if summary[index][3][binIndex][1] == 1:
            summary[index][3][binIndex][2] = nSteps
            summary[index][3][binIndex][4] = nSteps
        else:
            summary[index][3][binIndex][2] = min(nSteps,summary[index][3][binIndex][2])
            summary[index][3][binIndex][4] = max(nSteps,summary[index][3][binIndex][4])
        if nSteps <= nStepsSplit:
            summary[index][3][binIndex][5] += 1
        else:
            summary[index][3][binIndex][6] += 1
        summary[index][3][binIndex][7].append(nSteps)

    #Calculating the average
    for each in summary:
        for eachYaw in each[3]:
            if eachYaw[1] != 0:
                eachYaw[3] = round(eachYaw[3]/float(eachYaw[1]),2)
                eachYaw[5] = round(eachYaw[5]/float(eachYaw[1]),2)*100
                eachYaw[6] = round(eachYaw[6]/float(eachYaw[1]),2)*100
        # Updating bin3
        for nSteps in each[4]:
            if nSteps != 99:
                nSteps = min(maxSteps+1,nSteps)
                each[5][nSteps] += 1

        # print each[0] , "->" , each[1] , "->" , each[2] , "->" , each[5]

    return summary

#Output neat table of summary.
def printSummary(summary):
    t = PrettyTable(['Bins(Yaw)', '#Data' , 'Min Steps', 'Avg Steps', 'Max Steps', '% <= '+ str(nStepsSplit) + ' steps', '% > '+ str(nStepsSplit)  + ' steps'])
    for each in summary:
        t.clear_rows()
        t.title = each[0] + " - Roll : " + str(int(round(np.degrees(each[1]),0))) + " , Pitch : " + str(int(round(np.degrees(each[2]),0)))
        for eachYaw in each[3]:
            if eachYaw[0] >= minYaw and eachYaw[0] <= maxYaw:
                if eachYaw[1] > 0:
                    t.add_row([eachYaw[0] ,eachYaw[1], eachYaw[2], eachYaw[3], eachYaw[4] , eachYaw[5] , eachYaw[6]])
                else:
                    t.add_row([eachYaw[0] , " ", " " , " " , " " , " ", " "])
        print(t)

#Generate a bar graph of the summary
def graphSummary(path,summary):

    # Plotting the different object configurations - box plot
    plots = []
    for each in summary:
        fig, ax = plt.subplots(1,1); ax = np.ravel(ax)
        fig.set_size_inches(7, 3)
        fig.suptitle(each[0]+' - Box plot of number of steps for a successful grasp', fontsize='medium')

        degree = []; number = []
        minGrasp = []; meanGrasp = []; maxGrasp = []
        split1 = []; split2 = []
        bxPlt = [];
        titleStr = "Config : Roll - {} , Pitch - {}".format(str(int(round(np.degrees(each[1]),0))), str(int(round(np.degrees(each[2]),0))))
        for eachYaw in each[3]:
            if eachYaw[0] >= minYaw and eachYaw[0] <= maxYaw:
                degree.append(int(eachYaw[0]))
                number.append(eachYaw[1])
                minGrasp.append(float(eachYaw[2]))
                meanGrasp.append(float(eachYaw[3]))
                maxGrasp.append(float(eachYaw[4]))
                split1.append(float(eachYaw[5]))
                split2.append(float(eachYaw[6]))
                bxPlt.append(eachYaw[7])

        ax[0].set_title(titleStr, fontsize='small')

        bp = ax[0].boxplot(bxPlt,positions=np.arange(len(bxPlt)),meanline=True,showmeans=True,patch_artist=True)
        for median in bp['medians']:
            median.set(color ='red',linewidth = 2)
        for mean in bp['means']:
            mean.set(color ='blue',linewidth = 1)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')

        ax[0].grid(axis='y')
        ax[0].set_yticks(np.arange(8))
        ax[0].set_ylabel('# Steps to grasp', fontsize='x-small')
        ax[0].yaxis.set_tick_params(labelsize=7)
        ax[0].tick_params(axis='y', labelcolor="k")
        ax[0].set_ylim(bottom=0)
        ax[0].set_ylim(top=8)

        ax[0].set_xlabel('Yaw bins\n(# Datapoints)', fontsize='x-small')
        ax[0].xaxis.set_tick_params(labelsize=7)
        ax[0].set_xticklabels(["<"+str(s)+"\n("+str(n)+")" for s,n in zip(degree,number)])

        fig.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'],loc='lower center',ncol=2,fontsize='x-small')
        # fig.legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plots.append(fig2img(fig))

    plots2jpg(plots,path[:path.rfind('/')+1] + each[0] + '_BP_',2)


    # Plotting the different object configurations - Num steps to grasp plot
    xAxis = [str(i) for i in range(maxSteps+1)]
    plots = []
    for each in summary:
        fig, ax = plt.subplots(1,1); ax = np.ravel(ax)
        fig.set_size_inches(6, 4)

        temp = np.cumsum(each[5])
        nData = temp[-1]
        avg = round(np.sum(np.array(each[5])*np.array(list(range(len(each[5])))))/float(nData),1)
        temp = temp / float(temp[-1])
        ax[0].plot(xAxis,temp[0:maxSteps+1]*100,color='b', marker="o")

        titleStr = "{} : Roll - {} , Pitch - {}".format(
                    each[0],
                    str(int(round(np.degrees(float(each[1])),0))),
                    str(int(round(np.degrees(float(each[2])),0))))
        ax[0].set_title(titleStr , fontsize='small')

        ax[0].grid(axis='y')
        ax[0].set_ylabel('Success percentage', fontsize='small')

        ax[0].set_xlabel('Step number that leads to a successful grasp', fontsize='small')

        fig.suptitle("Step number distribution - Training Data : Avg Steps "+str(avg), fontsize='medium')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plots.append(fig2img(fig))

    plots2jpg(plots,path[:path.rfind('/')+1] + each[0] + '_nSteps_',1)

    # Plotting the direction charts
    plots = []
    graphColors =  ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    theta = list(np.linspace(0,np.pi/4*7,8))
    theta.append(0)
    for each in summary:
        fig = plt.figure()
        fig.set_size_inches(5, 5)
        titleStr = "{} : Roll - {} , Pitch - {}".format(
                    each[0],
                    str(int(round(np.degrees(float(each[1])),0))),
                    str(int(round(np.degrees(float(each[2])),0))))
        fig.suptitle("Training Data\n"+
                     "% of time a direction was taken\n"+
                     titleStr, fontsize='medium')
        ax=plt.subplot(1, 1, 1, projection='polar')

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2)
        ax.set_thetagrids(np.degrees(theta), ["N","NE","E","SE","S","SW","W","NW","N"])
        for step,i in zip(each[6],list(range(len(each[6])))):
            if(np.sum(step) > 0):
                sum = float(np.sum(step))
                temp = list(np.array(step)/sum*100)
                temp.append(temp[0])
                ax.plot(theta,temp,color = graphColors[i],label="Step "+str(i+1)+" ("+str(int(sum))+")")
                ax.fill(theta,temp, facecolor=graphColors[i], alpha=0.25)
                ax.set_rmin(0)

        fig.legend(loc="lower right")
        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        plots.append(fig2img(fig))

    plots2jpg(plots,path[:path.rfind('/')+1] + each[0] + '_Radar_',1)

#Save the summary for future use.
def saveToCSV(path,summary):
    newPath = path[:-4] + "_summary.csv"
    with open(newPath, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for each in summary:
            for eachYaw in each[3]:
                if eachYaw[0] >= minYaw and eachYaw[0] <= maxYaw:
                    csvwriter.writerow([each[0], each[1], each[2], eachYaw[0] ,eachYaw[1], eachYaw[2], eachYaw[3], eachYaw[4], eachYaw[5], eachYaw[6]])

    print("Data Written to : "+newPath)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect number of arguments")
        sys.exit()

    path = sys.argv[1]
    mode = sys.argv[2]
    if mode!='GRAPH_SAVE' and mode!='PRINT' and mode!='CSV' :
        mode = 'GRAPH_SAVE'

    nBins = 360/10

    summary = genSummary(path,nBins)
    if(mode == "PRINT"):
        printSummary(summary)
    elif(mode == "GRAPH_SAVE"):
        graphSummary(path,summary)
    elif(mode == "CSV"):
        saveToCSV(path,summary)
    elif(mode == "PCA"):
        PCASummary(path)
