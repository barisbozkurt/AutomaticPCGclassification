#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test definition embodying all components of a learning experiment:

    Data (files as well as segmentation and features specifications),
    Machine learning model
    Test specifications (batch and eopch sizes, etc.)

Created on May 15th 2017

@author: Baris Bozkurt
"""
import pickle
import time
import numpy as np
from keras.models import load_model
from Result import Result
from models import loadModel

class Test(object):
    '''Design of a complete learning test

    '''
    def __init__(self,modelName,data,resultsFolder,batch_size=128,num_epochs=100,mergeType='majorityVoting'):
        '''Constructor

        Args:
            modelName (str): Name of the NN model to be applied on data.
                Model loaded using the loadModel() function defined in models.py
            data (Data): Data object (embodies data specifications as well
                features, segmentation, etc)
            resultsFolder (str): Path where results will be saved
            batch_size (int): Batch size (default: 128)
            num_epochs (int): Number of epochs for the learning tests (default: 100)
            mergeType (str): If more than a single feature is used, the final
                outputs obtained for each feature can be merged. mergeType
                defines the strategy for merging this information.
                Ex:"majorityVoting", etc. If not specified, default
                method ('majorityVoting') will be applied.

        '''
        self.modelName=modelName
        self.data=data
        self.mergeType=mergeType
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.resultsFolder=resultsFolder
        #opening a report file for printing logs
        localtime = time.asctime( time.localtime(time.time()))
        self.logFile=open(resultsFolder+'TestRep_'+localtime.replace(':','').replace(' ','_')[4:-7]+'.txt','w')
        self.logFile.write(self.data.info+'\n')
        self.logFile.write('Model name: '+modelName+', dba: '+data.name+'\n')
        self.logFile.write('Test started at: '+localtime+'\n')


    def run(self):
        '''Runs the test for each feature in the feature list

        Each test on a separate feature is performed in isolation due to memory
        concerns. Outputs are written to files

        '''
        #Load and perform test for each feature
        #For each feature a Keras model needs to be separately loaded since the input size
        #is defined by the feature data size
        for feature in self.data.features:
            #new file and variables for outputs of this test for the specific feature
            dimsStr=str(feature.dimensions[0])+'by'+str(feature.dimensions[1])
            if feature.involveDelta:
                deltaStr='del'
            else:
                deltaStr=''
            self.modelFile=self.resultsFolder+'M'+'_'+self.modelName+feature.name+dimsStr+deltaStr+'_'+feature.segmentation.abbr+feature.window4frame[0]+'.h5'
            self.logFile.write('-------------------------------------------------\n')
            self.logFile.write('New test initiated: '+self.modelFile.replace('.h5','')+'\n')
            self.logFile.write('Feature: '+feature.name+dimsStr+' using segmentation: '+feature.segmentation.sourceType+', size: '+feature.segmentation.sizeType+'\n')
            self.logFile.write('\tperiodSync:'+str(feature.segmentation.periodSync)+', with window: '+feature.window4frame+'\n')
            print('-------------------------------------------------')
            print('New test initiated: '+self.modelFile.replace('.h5',''))
            #initiating a new Result object for computing and writing results
            self.result=Result(self.resultsFolder,self.modelName,feature.name,feature.dimensions)

            #Loading train, validation, test samples
            # data.train_x, .train_y, .valid_x, .valid_y, .test_x, .test_y
            dataIsLoaded=self.data.loadSplittedData(feature)
            #If data could be loaded, performing the test
            if dataIsLoaded:
                self.result.num_classes=self.data.num_classes
                #Load NN model
                self.model=loadModel(self.modelName,self.data.shape,self.data.num_classes)
                #perform training
                self._train()
                #perform testing
                self._test()
                #printing to log
                self.logFile.write('Results saved for '+self.testName+'\n')
                self.logFile.write(time.asctime( time.localtime(time.time()))+'\n')
                print('Results saved for '+self.testName)
            #printing to log
            self.logFile.write('-------------------------------------------------\n')
            print('-------------------------------------------------')
        #printing to log
        localtime = time.asctime( time.localtime(time.time()))
        self.logFile.write('Test finished at: '+localtime+'\n')
        self.logFile.close()

    def _train(self):
        '''Training step

        Saves model that gives highest accuracy on validation(on frame level)
        in an h5 file. Plots change of loss and accuracy versus epochs

        '''
        #Creating a unique name for the test
        self.testName=self.modelFile.replace('.h5','')+'_'+self.mergeType[0:3]
        #printing to log
        self.logFile.write('Training: '+self.testName+'\n')
        self.logFile.write('Number of frames/features: (train,validation,test): '+str([len(self.data.train_x),len(self.data.valid_x),len(self.data.test_x)])+'\n')
        #count data in category for
        uniqueLabels, counts = np.unique(np.argmax(self.data.train_y,axis=1), return_counts=True)
        self.logFile.write('Train categories and number of samples: '+str(uniqueLabels)+', '+str(counts)+'\n')
        print('Number of frames/features: (train,validation,test): ',len(self.data.train_x),len(self.data.valid_x),len(self.data.test_x))
        print('Train categories and number of samples: '+str(uniqueLabels)+', '+str(counts))

        highestTestScore=0
        #-----TRAINING-----
        #Creating a flag to track if there is any learning or not. It is possible in some cases that accuracy values 
        #   do not change at all for all the learning process. Then, this 
        validScoreNeverIncreased=True
        for epoch in range(self.num_epochs):
            if epoch%20==0:#print position each 20 epochs
                print('Epoch:',epoch)
            #One epoch of learning
            self.model.fit(self.data.train_x, self.data.train_y,batch_size=self.batch_size,epochs=1,verbose=0,
                  validation_data=(self.data.valid_x, self.data.valid_y))
            scoreValid = self.model.evaluate(self.data.valid_x, self.data.valid_y, verbose=0)
            scoreTrain = self.model.evaluate(self.data.train_x, self.data.train_y, verbose=0)
            #Storing the model in a h5 file if accuracy on validation is improved
            if scoreValid[1]>highestTestScore:
                validScoreNeverIncreased=False
                highestTestScore=scoreValid[1]
                #saving model with highest score
                self.model.save(self.modelFile)
                self.logFile.write('Model saved: epoch %d, accuracyOnValidation: %f\n' % (epoch,highestTestScore))
                print('Model saved: epoch %d, accuracyOnValidation: %f' % (epoch,highestTestScore))
            if scoreTrain[1]>0.99999 and False:#comment part after 'and' if you like to stop training when an accuracy of 1 is achieved for train
                break
            self.result.appendEpoch(scoreTrain,scoreValid)
        #Report training results
        localtime = time.asctime( time.localtime(time.time()))
        self.result.plotLossVsEpochs(self.testName+localtime.replace(':','').replace(' ','_')[4:-7]+'_loss.png')
        self.result.plotAccVsEpochs(self.testName+localtime.replace(':','').replace(' ','_')[4:-7]+'_acc.png')
        
        if validScoreNeverIncreased:
            self.result.learningFailed=True
        else:#this flag will be checked while computing average of success measures, if no learning happened it should not be used in average comp.
            self.result.learningFailed=False

    def _test(self):
        '''Testing the learned model(using train and validation sets)
            on the test data set

        Assigns:
            self.result.test_fileDecisions (dict {str->[#,#]}): File level decisions
                {file-name, [true_class,predicted_class]}
            self.result.test_fileProbs (dict {str->[#,#]}): Frame classification probabilities for a file
                {file-name, [probabilities_of_frames]}
        '''
        #Load best model saved in modelFile for this test
        model = load_model(self.modelFile)#load_model is imported from Keras
        #Perform estimation on the test data set
        y_probs=model.predict(self.data.test_x, batch_size=self.data.test_x.shape[0], verbose=0)
        #Ground truth is in test_y in a vector form, convert to 1 dimensional label and store as test_y_trueFrame
        self.result.test_y_trueFrame=np.argmax(self.data.test_y,axis=1)
        self.result.test_y_probsFrame=y_probs
        self.result.test_patSegMaps=self.data.test_patSegMaps
        #converting from activation output value to class-index by picking the largest value
        self.result.test_y_predsFrame=np.argmax(y_probs,axis=1)

        for curFile in self.data.test_patSegMaps:#for each file in the test set
            segInds=self.data.test_patSegMaps[curFile]#get indexes of frames for the file
            #Get true class for the file from frame labels(all should be the same: checked within the function)
            true_class=self._patientTrueClass(self.result.test_y_trueFrame[segInds])
            #Get estimated probabilities for frames of a file
            filesFrameProbs=list(y_probs[segInds])
            self.result.test_fileProbs[curFile]=filesFrameProbs
            #Using the 'mergeType', make a file level prediction from frame level probabilities
            pred_class=self._patientPredClass(filesFrameProbs)
            self.result.test_fileDecisions[curFile]=[true_class,pred_class]#store true and predicted classes

        localtime = time.asctime( time.localtime(time.time()))
        self.result.report(self.testName+localtime.replace(':','').replace(' ','_')[4:-7]+'_res.txt')

        #saving results
        pickleProtocol=1#choosen for backward compatibility
        with open(self.modelFile+localtime.replace(':','').replace(' ','_')[4:-7]+self.mergeType[0:3]+'.pkl' , 'wb') as f:
            pickle.dump(self.result, f, pickleProtocol)

    def _patientPredClass(self,pat_y_probs):
        '''File level estimation from file's estimated frame probabilities

        To be implemented: mergeType='meanProbs': computing mean probability of
            frames after removing outliers and taking decision from mean-probability
        '''
        if self.mergeType=="majorityVoting":
            pat_y_preds=np.argmax(pat_y_probs,axis=1)
            pat_y_preds=list(pat_y_preds)
            return max(pat_y_preds,key=pat_y_preds.count)
        else:
            print('Unknown merge type from frame level to file level')
            return -1

    def _patientTrueClass(self,trueFrameClasses):
        '''Given all frame level decisions for a specific file/patient,
        checks if they all are the same (they should be!)
        and returns that common value as the true class else -1'''

        trueClass=trueFrameClasses[0]
        for cat in trueFrameClasses:
            if trueClass!=cat:
                self.logFile.write('Error: frame level true classes do not match for a patient\n')
                print('Error: frame level true classes do not match for a patient')
                print(trueFrameClasses)
                return -1
        return trueClass
