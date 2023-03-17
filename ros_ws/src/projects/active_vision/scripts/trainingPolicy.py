#!/usr/bin/env python3

import sys, csv, time, random, math
import numpy as np
from os import path
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump
from Q_controller2 import *
import pickle

def helpDisp(text):
    print(text)
    print('\n-----Policy Training Help-----\n')
    print('Arguments : [CSV filename]')
    print('CSV filename : CSV file name (Data.csv)')
    print('\n-----End Help-----\n')
    sys.exit()

'''Base class to handle the entire input->preprocessing->prediction
pipeline. Must be trained with calculateFunction before using.
Subclasses are for different specific preprocessing schemes.'''
class methodPipeline(object):
    def __init__(self):
        self.ready = False
        self.scaler = StandardScaler()

    # Normalizing the data using StandardScaler. Not designed for external use.
    def normalize(self, X):
        return self.scaler.transform(X)

    # Train the model
    def train(self, X, y):
        return self.ready

    # Predict with the trained model
    def predict(self, X):
        return -1

    def saveModel(self, name):
        dump(self, open(name+'_pipeline.pkl', 'wb'))

    def loadModel(self, name):
        self = pickle.load(open(name+'_pipeline.pkl', 'rb'))
        self.ready = True
        return self

# PCA -> LDA -> LR Model
class PCA_LDA_LR_Model(methodPipeline):
    def __init__(self, n_components):
        methodPipeline.__init__(self)
        self.n_components = n_components

    def train(self, X, y):
        self.scaler.fit(X)

        self.pca = PCA(self.n_components)
        X_PCA = self.pca.fit_transform(self.normalize(X))

        self.lda = LDA()
        X_LDA = self.lda.fit_transform(X_PCA, y)

        self.lr = LogisticRegression(solver='liblinear', multi_class='auto')
        self.lr.fit(X_LDA, y)

        self.ready = True
        return methodPipeline.train(self, X, y)

    def predict(self, X):
        if(not self.ready):
            return int(methodPipeline.applyFunction(self, X))
        else:
            return int(self.lr.predict(self.lda.transform(self.pca.transform(self.normalize(X)))))

class Q_Model(methodPipeline):
    def __init__(self, num_inputs):
        methodPipeline.__init__(self)
        self.agent = Agent(alpha=0.005, gamma=0.99, n_actions=8, epsilon=0.01, batch_size=64, input_dims=num_inputs, epsilon_decount=0.996, epsilon_min=0.01, mem_size=1000000, fname='4layersall2_')

    def train(self, X, y):
        self.agent.load_model(None)
        self.ready = True
        return self.ready

    def predict(self, X):
        if(not self.ready):
            return int(methodPipeline.applyFunction(self, X))
        else:
            return int(self.agent.choose_action(X)+1)

# PCA -> LDA Model
class PCA_LDA_Model(methodPipeline):
    def __init__(self, n_components):
        methodPipeline.__init__(self)
        self.n_components = n_components

    def train(self, X, y):
        self.scaler.fit(X)

        self.pca = PCA(self.n_components)
        X_PCA = self.pca.fit_transform(self.normalize(X))

        self.lda = LDA()
        self.lda.fit(X_PCA, y)

        self.ready = True
        return methodPipeline.train(self, X, y)

    def predict(self, X):
        if(not self.ready):
            return int(methodPipeline.applyFunction(self, X))
        else:
            return int(self.lda.predict(self.pca.transform(self.normalize(X))))

# PCA -> LR Model
class PCA_LR_Model(methodPipeline):
    def __init__(self, n_components):
        methodPipeline.__init__(self)
        self.n_components = n_components

    def train(self, X, y):
        self.scaler.fit(X)

        self.pca = PCA(self.n_components)
        X_PCA = self.pca.fit_transform(self.normalize(X))

        self.lr = LogisticRegression(solver='liblinear', multi_class='auto')
        self.lr.fit(X_PCA, y)

        self.ready = True
        return methodPipeline.train(self, X, y)

    def predict(self, X):
        if(not self.ready):
            return int(methodPipeline.applyFunction(self, X))
        else:
            return int(self.lr.predict(self.pca.transform(self.normalize(X))))

# Model that randomly returns a direction [1-8].
class Random_Model(methodPipeline):
    def train(self, X, y):
        self.ready = True
        return methodPipeline.train(self, X, y)

    def predict(self, X):
        return int(np.random.choice(range(1,9), X.shape[0]))

# Model that always returns 2.
class Brick_Model(methodPipeline):
    def train(self, X, y):
        self.ready = True
        return methodPipeline.train(self, X, y)

    def predict(self, X):
        return int(np.random.choice(range(2,3), X.shape[0]))

# class BrickPipeline(methodPipeline):
#     def __init__(self):
#         super(methodPipeline, self).__init__()
#
# #Dummy model that always predicts the same direction.
# class BrickModel():
#     def fit(self, X, y):
#         return
#
#     def predict(self, X):
#         return np.ones(X.shape[0])

#Function to read the HAF state vector csv file
def readHAFStVec(fileName,dim):
    states = []
    dir = []
    with open(fileName) as csv_file:
        # Reading the csv file
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            # Going through it line by line as storing it in a list
            if(len(line) != dim+4): helpDisp("ERROR : Data set inconsistent")
            currentDirection = float(line[dim+3])
            #Remove final images
            if(-1 != currentDirection):
                states.append([float(i) for i in line[2:dim+2]])
                dir.append([currentDirection])

    return np.asarray(states),np.asarray(dir)

#K-fold validation of the given pipeline with the given model.
# def kFold(filename, pipeline, model, name='na', dim=52, k=5):
#     stateVec, dirVec = readInput(filename, dim)
#     dirVec = dirVec.ravel()
#     kFold = KFold(n_splits=k, random_state=None)
#     accuracies = []
#     #Split the data, then train and test on the split.
#     for train_index, test_index in kFold.split(stateVec):
#         X_train, X_test = stateVec[train_index, :], stateVec[test_index, :]
#         y_train, y_test = dirVec[train_index], dirVec[test_index]
#         pipeline.calculateFunction(X_train, y_train)
#         X_train_vec = pipeline.applyFunction(X_train)
#         X_test_vec = pipeline.applyFunction(X_test)
#         model.fit(X_train_vec, y_train)
#         pred_values = model.predict(X_test_vec)
#         acc = accuracy_score(pred_values, y_test)
#         accuracies.append(acc)
#     #Display results
#     avg_accuracy = sum(accuracies)/k
#     for acc in accuracies:
#         print("%1.2f" % acc)
#     print("Overall average: %1.2f" % (avg_accuracy))
#     # Storing the trained models
#     if name != 'na':
#         pipeline.calculateFunction(stateVec, dirVec)
#         pipeline.saveModel(name)
#         model.fit(pipeline.applyFunction(stateVec), dirVec)
#         dump(model, open(name+'_lr.pkl', 'wb'))

# #Convenience function to run several tests at once on the same data.
# def standardRun(file):
#     model = LogisticRegression(solver='liblinear', multi_class='multinomial')
#     PCA_components = rospy.get_param("/active_vision/PCA_component_number")
#     PCALDA_components = rospy.get_param("/active_vision/PCALDA_component_number")
#     # model = KNeighborsClassifier(n_neighbors=7)
#     print("*****%s*****" % file)
#     print("*****PCA-LDA*****")
#     kFold(file, PCALDAPipeline(PCALDA_components), model, file[0:-12]+'PL')
#     print("*****PCA*****")
#     kFold(file, PCAPipeline(PCA_components), model, file[0:-12]+'P')
#     print("*****Brick*****")
#     kFold(file, PCAPipeline(PCA_components), BrickModel(), file[0:-12]+'B')

# if __name__ == "__main__":
    # dir = rospy.get_param("/active_vision/data_dir")
    # #Galen mode for convenience- feel free to edit for your local files
    # if len(sys.argv) == 1:
    #     files = ["./DataRecAV/pose0Data1_stateVec.csv", "./GitData1/Step1_NormalVP/dataNormalVPStep1_stateVec.csv", "./GitData1/Step2_NormalVP/dataNormalVPStep2_stateVec.csv"]
    #     for file in files:
    #         standardRun(file)
    #
    # elif len(sys.argv) != 2: helpDisp("ERROR : Incorrent number of arguments")
    # else:
    #     file = sys.argv[1]
    #     standardRun(dir+file)

    # if len(sys.argv) != 2: helpDisp("ERROR : Incorrent number of arguments")
    # else:
    #     stateVec, dirVec = readHAFStVec(sys.argv[1], 52)
    #     dirVec = dirVec.ravel()
    #     model = PCA_LDA_LR_Model(0.95)
    #     model.train(stateVec, dirVec)
