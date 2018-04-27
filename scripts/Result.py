#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017

@author: Baris Bozkurt
"""
import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

class Result(object):
    """Results for a single classification test

    lossTrain,accTrain,lossValid,accValid are lists that contain results at
    each learning epoch

    For definitions, refer to:
        https://en.wikipedia.org/wiki/Precision_and_recall
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """
    def __init__(self, saveFolder='', modelName='',featureName='', featureDims=[]):
        '''Constructor

        Args:
            saveFolder (str): Path to where results will be saved
            modelName (str): Name of the machine learning model ('uocSeq0', 'uocSeq1', etc.)
            featureName (str): Name of the feature ('mfcc', 'subenv', etc.)
            featureDims (list): Dimensions of the feature [timeDimension,frequencyDimension]
        '''
        self.saveFolder=saveFolder
        #if folder not exists, create it
        self._createFolder()

        self.modelName=modelName
        self.featureName=featureName
        self.featureDims=featureDims

        self.accuracy=-1
        self.sensitivity=-1
        self.specificity=-1
        self.F1=-1
        self.matthewsCorCoeff=-1
        self.ROC=[]
        self.confMat=None
        #results gathered during training at each epoch
        self.lossTrain=[]
        self.accTrain=[]
        self.lossValid=[]
        self.accValid=[]
        self.tp=0#true positive
        self.fp=0#false positive
        self.tn=0#true negative
        self.fn=0#false negative
        #
        self.num_classes=0#number of labels/categories in the data
        self.test_y_probsFrame=[]#estimated probabilities for the test set on the frame level
        self.test_y_predsFrame=[]#estimated labels for the test set on the frame level
        self.test_y_trueFrame=[]#true labels(ground truth) for test frames
        #
        #dictionary of file names versus true category and predicted category
        self.test_fileDecisions={}#example '0001.wav' -> [4 5] (file's true category was 4, predicted as 5)
        #dictionary of file names versus predicted frame probabilities
        self.test_fileProbs={}
        #dictionary of misclassified files versus their frame probabilities
        self.misLabeledFiles={}


    def _computeMeasures(self):
        ''''Main function for computing rates and the confusion matrix'''
        self.confMat=np.zeros((self.num_classes,self.num_classes))
        for (fileName,labels) in self.test_fileDecisions.items():
            true_class=int(labels[0])
            pred_class=int(labels[1])
            self.confMat[true_class,pred_class]+=1
            if true_class==pred_class:#correct classification
                if true_class==1:
                    self.tp+=1
                else:
                    self.tn+=1
            else:#false classification
                self.misLabeledFiles[fileName]=self.test_fileProbs[fileName]
                if pred_class==1:
                    self.fp+=1
                else:
                    self.fn+=1
        #tp,tn,fp,fn counted, compute measures from these counts
        self._computeSensitivity()
        self._computeSpecificity()
        self._computeAccuracy()
        self._computeF1()
        self._computeMCC()

    def _computeSensitivity(self):
        if (self.tp+self.fn)>0:#avoid zero-division error
            self.sensitivity=(self.tp)/(self.tp+self.fn)
        else:
            self.sensitivity=-1

    def _computeSpecificity(self):
        if (self.tn+self.fp)>0:#avoid zero-division error
            self.specificity=(self.tn)/(self.tn+self.fp)
        else:
            self.specificity=-1

    def _computePrecision(self):
        if (self.tp+self.fp)>0:#avoid zero-division error
            self.precision=self.tp/(self.tp+self.fp)
        else:
            self.precision=-1

    def _computeAccuracy(self):
        if (self.tp+self.tn+self.fp+self.fn)>0:#avoid zero-division error
            self.accuracy=(self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        else:
            self.accuracy=-1

    def _computeF1(self):
        if (2*self.tp+self.fp+self.fn)>0:#avoid zero-division error
            self.F1=2*self.tp/(2*self.tp+self.fp+self.fn)
        else:
            self.F1=-1

    def _computeMCC(self):
        tp=self.tp
        tn=self.tn
        fp=self.fp
        fn=self.fn
        if ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))>0:#avoid zero-division error
            self.matthewsCorCoeff=(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        else:
            self.matthewsCorCoeff=-1


    '''Incremental updates on results'''
    def appendEpoch(self,scoreTrain,scoreValid):
        '''Appends new epoch results to existing list of result values

        Args:
            scoreTrain [list]: contains [lossTrain,accTrain]
            scoreValid [list]: contains [lossValid,accValid]

        '''
        self.lossTrain.append(scoreTrain[0])
        self.lossValid.append(scoreValid[0])
        self.accTrain.append(scoreTrain[1])
        self.accValid.append(scoreValid[1])

    '''Plotting functions'''
    def plotLossVsEpochs(self,outFileName=str()):
        '''Plotting loss versus epoch number for train and validation

        Args:
            outFileName (str): [optional] name of the file where plot will be saved
        '''
        plt.plot(np.array(range(1,len(self.lossTrain)+1)),np.array(self.lossTrain), color='black', label='Train')
        plt.plot(np.array(range(1,len(self.lossTrain)+1)),np.array(self.lossValid), color='red', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        if not outFileName==str():#save to file if outFileName specified
            plt.savefig(outFileName,dpi=300)
            plt.clf()

    def plotAccVsEpochs(self,outFileName=str()):
        '''Plotting accuracy versus epoch number for train and validation

        Args:
            outFileName (str): [optional] name of the file where plot will be saved
        '''
        plt.plot(np.array(range(1,len(self.accTrain)+1)),np.array(self.accTrain), color='black', label='Train')
        plt.plot(np.array(range(1,len(self.accTrain)+1)),np.array(self.accValid), color='red', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        if not outFileName==str():#save to file if outFileName specified
            plt.savefig(outFileName,dpi=300)
            plt.clf()

    def report(self,filename):
        '''Writing measures and misclassified files to report/results file'''
        self._computeMeasures()
        with open(filename, "w") as text_file:
            text_file.write('File-level measures\n')
            for rowInd in range(self.num_classes):
                for colInd in range(self.num_classes):
                    text_file.write(str(self.confMat[rowInd,colInd])+'\t|')
                text_file.write('\n')
            text_file.write('\nSensitivity =\t'+str(self.sensitivity))
            text_file.write('\nSpecificity =\t'+str(self.specificity))
            text_file.write('\nAccuracy    =\t'+str(self.accuracy))
            text_file.write('\nF1          =\t'+str(self.F1))
            text_file.write('\nMatthews CC =\t'+str(self.matthewsCorCoeff))
            text_file.write('\n------------------------------------------\n')
            text_file.write('Frame-level measures\n')
            (frmConfMat,frmSens,frmSpec)=self._computeFrameMeasures()
            for rowInd in range(self.num_classes):
                for colInd in range(self.num_classes):
                    text_file.write(str(frmConfMat[rowInd,colInd])+'\t|')
                text_file.write('\n')
            text_file.write('\nSensitivity =\t'+str(frmSens))
            text_file.write('\nSpecificity =\t'+str(frmSpec)+'\n')
            self.frameConfMat=frmConfMat
            self.frameSensitivity=frmSens
            self.frameSpecificity=frmSpec
            text_file.write('\n------------------------------------------\n')

        ROCfile=filename.replace('.txt','_ROC.png')
        self._plotROC(ROCfile)

    def _computeFrameMeasures(self):
        ''''Computation of measures on frame level'''
        confMat=np.zeros((self.num_classes,self.num_classes))
        (tp,tn,fp,fn)=(0,0,0,0)
        for ind in range(len(self.test_y_trueFrame)):
            true_class=self.test_y_trueFrame[ind]
            pred_class=self.test_y_predsFrame[ind]
            confMat[true_class,pred_class]+=1
            if true_class==pred_class:#correct classification
                if true_class==1:
                    tp+=1
                else:
                    tn+=1
            else:#false classification
                if pred_class==1:
                    fp+=1
                else:
                    fn+=1
        if (tp+fn)>0:#avoid zero-division error
            sensitivity=(tp)/(tp+fn)
        else:
            sensitivity=-1
        
        if (tn+fp)>0:#avoid zero-division error
            specificity=(tn)/(tn+fp)
        else:
            specificity=-1
            
        return (confMat,sensitivity,specificity)
        
    def _plotROC(self,ROCfile,title=''):
        '''Re-performing file level decision from frame level probabilities
        Implemented for binary-classification tasks. For more than 2 labels, 
        consider implementing a new version
        
        For computing mean frame probabilities, the probabilities are sorted
        and the values at two ends are dropped out first. The number of frames 
        to drop is controlled by the portionLeaveOutFrms variable which is 
        specified in percentage. Ex: if portionLeaveOutFrms=40, that means
        20% lowest values and 20% highest will be left out and the mean will 
        be computed afterwards 

        '''
        portionLeaveOutFrms=30#portion of frames to be left out as extreme values of probability to be able to compute a realiable mean-prob.
        #creating a dictionary: file -> average frame-level pathology probability
        meanPathologyProbs={}
        for curFile in self.test_fileProbs:
            frameProbs=self.test_fileProbs[curFile]
            #sumPathProb=0#sum of probability of pathology
            allPathProb=[]#we will collect all frame pathology probabilities in a list
            for frmProb in frameProbs:#each frame's probability vector
                #sumPathProb+=frmProb[1]
                allPathProb.append(frmProb[1])
            #for leaving out extreme prob. values, values will be sorted and the mid-values will be used
            sortedProb=np.sort(allPathProb)
            numFrms2leaveOnEnds=int(np.round(len(sortedProb)*(portionLeaveOutFrms/2)/100))
            if len(sortedProb[numFrms2leaveOnEnds:-numFrms2leaveOnEnds])>3:#if at least 3 frames left after removing extremes
                meanVal=np.mean(sortedProb[numFrms2leaveOnEnds:-numFrms2leaveOnEnds])
            else:
                meanVal=np.mean(sortedProb)
            #meanPathologyProbs[curFile]=(sumPathProb/len(frameProbs))
            meanPathologyProbs[curFile]=meanVal

        #trying different threshold values to compute a ROC curve
        allTpr=[];allFpr=[];#ROC curve y and x points, Tpr: true positive rate, Fpr: false positive rate
        for threshold in np.linspace(0.0, 1.0, num=100):#assiging new threshold values in range 0-1
            fileDecDict={}
            for curFile in self.test_fileProbs:
                (true_class,pred_class)=self.test_fileDecisions[curFile]
                prob=meanPathologyProbs[curFile]
                #Making a new decision via thresholding
                if prob>=threshold:
                    pred_class=1
                else:
                    pred_class=0
                fileDecDict[curFile]=[true_class,pred_class]
            (tpr,fpr)=self._computeROCpoint(fileDecDict)
            #add point to ROC curve
            allTpr.append(tpr)
            allFpr.append(fpr)
        #plotting the ROC curve
        plt.clf()
        plt.plot(np.array(allFpr),np.array(allTpr), color='black')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(title, loc='left')
        plt.savefig(ROCfile,dpi=300)
        plt.clf()

        #storing ROC curve data to be able to plot various ROC curves in one figure
        self.ROC=(np.array(allTpr),np.array(allFpr))

    def _computeROCpoint(self,fileDecDict):
        '''Given a file-decision dictionary, computes
        True positive rate(tpr) and
        False positive rate(fpr)

        It is assumed that the dictionary has the following structure:
            file -> [true_class,predicted_class]

        (This function is used for computing ROC curves)
        '''
        tp=0; fp=0; tn=0; fn=0
        for (fileName,labels) in fileDecDict.items():
            true_class=int(labels[0])
            pred_class=int(labels[1])
            if true_class==pred_class:#correct classification
                if true_class==1:
                    tp+=1
                else:
                    tn+=1
            else:#false classification
                if pred_class==1:
                    fp+=1
                else:
                    fn+=1
        tpr=tp/(tp+fn)
        fpr=fp/(fp+tn)
        return (tpr,fpr)

    def _createFolder(self):
        '''Checks existence of the folder and creates if it does not exist'''
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
