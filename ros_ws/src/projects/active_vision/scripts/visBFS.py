#!/usr/bin/env python3

import sys, csv
import numpy as np
from graphviz import Digraph
import rospy

def helpDisp(text):
    print(text)
    print('\n-----BFS Visualization Help-----\n')
    print('Arguments : [CSV filename]')
    print('CSV filename : CSV file name (Data.csv)')
    print('\n-----End Help-----\n')
    sys.exit()

#Function to read the input data file
def readInput(fileName):
    header = []
    data = []
    lineNo = 1;
    with open(fileName) as csv_file:
        # Reading the csv file
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            if lineNo == 1:
                header.append([i for i in line[:]])
            else:
                data.append([i for i in line[:]])
            lineNo = lineNo+1;

	return np.asarray(header),np.asarray(data)

def genGraph(fileName):
    header,data = readInput(fileName)
    tree = Digraph()
    tree.graph_attr['layout'] = "dot"
    tree.graph_attr['ranksep'] = "3.0"
    tree.graph_attr['nodesep']= "0.1"
    tree.node_attr['shape'] = "box"
    tree.node_attr['style'] = "striped"

    ratio = round(float(data[0][3])/float(data[0][4]),2)
    text = header[0][0] + "\n" + \
           "Pose Code : " + header[0][1] + "\n" + \
           "Yaw : " + header[0][2] + "\n" + \
           "Search Depth : " + fileName[-7] + "\n" + \
           data[0][3] + "/" + data[0][4] + " = " + str(ratio);
    tree.node(data[0][1],label=text,color="gray")
    tree.node("A"+data[0][1],label=text,color="gray")
    for x in data:
        if x[0] != "-1":
            ratio = round(float(x[3])/float(x[4]),2)
            text = x[6] + "\n" + \
                   x[3] + "/" + x[4] + " = " + str(ratio);
            if x[2] == "1":
                tree.node(x[1],label=text,color="lawngreen")
            else:
                tree.node(x[1],label=text,color="gray")
            tree.edge(x[0],x[1])

            if x[0] == "0":
                tree.node("A"+x[1],label=text,color="gray")
                tree.edge("A"+x[0],"A"+x[1])

    tree.render(fileName[:-4])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        helpDisp("ERROR : Incorrent number of arguments")
    else:
        dir = rospy.get_param("/active_vision/data_dir")
        file = sys.argv[2]

    genGraph(dir+file)
