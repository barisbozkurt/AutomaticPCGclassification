#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running repeated experiments for the system with settings:
    Sub-band envelopes computed from async. frames extracted with 
    length of 2 seconds and hop size of 1 second. 

The data balancing operation involves creation of new samples. This will add
    new files in your database folder. Files named *aug_ are created that way. 
    You can see a complete list of files created by checking the train.txt and 
    validation.txt file-list files. New files' names are placed at the end of 
    each list. cleanFilesOfPrevSess function can be used to delete these files
    at the end of the tests
    
This script has been tested on Physionet 2016 data accessible from:
https://www.physionet.org/physiobank/database/challenge/2016/training.zip 
Example resulting files are available in the 'exampleResults' folder

The script first creates a 'data' folder, downloads Physionet data and unzips 
into that folder. The original validation package of Physionet is used as
test data and the train package is split into train and validation.

To replicate the tests, the user simply needs to run this script. The test 
results are written to data/results

Created on May 15th 2017

@author: Baris Bozkurt
"""
import os
import urllib.request
import zipfile

#os.chdir(os.path.dirname(__file__))
localRepoDir=os.getcwd()

#%%Database download
dataFolder=localRepoDir+'/data/'
if not os.path.exists(dataFolder):
    os.mkdir(dataFolder);
dbaName='Physionet'
urls={}
urls['train']='https://www.physionet.org/physiobank/database/challenge/2016/training.zip'
urls['test']='https://www.physionet.org/physiobank/database/challenge/2016/validation.zip'

#Dowloading Physionet train data
for dataCategory in urls.keys():
    targetDir=dataFolder+'/'+dataCategory
    if not os.path.exists(targetDir):
        os.mkdir(targetDir);
        url=urls[dataCategory]
        filename=url.split('/')[-1]
        #Downloading the zip file from the url
        urllib.request.urlretrieve(url,filename)
        #Unzipping to a specific folder
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(targetDir)
        zip_ref.close()
        os.remove(filename)#Removing the zip file
        print('Data downloaded and unzipped to: ',targetDir)
    else:
        print('Folder ',targetDir,' already exists, delete it if you want to re-download data')

#The results will be put in 'results' folder in the database folder
resultsFolder=dataFolder+"/results/"
if not os.path.exists(resultsFolder):#create folder if not exists
    os.mkdir(resultsFolder)
#The features will be saved in 'features' folder in the database folder
featureFolder=dataFolder+"/features/"
if not os.path.exists(featureFolder):#create folder if not exists
    os.mkdir(featureFolder)

#%%TEST specification
from Segmentation import Segmentation
from Feature import Feature
from Data import Data
from Test import Test
from miscFuncs import cleanFilesOfPrevSess


info="Single-test repeated experiment"#any info about the test
#Setting train-validation-test split ratios, 
# if 'test' exists as a static folder, the first two values are used 
# and rest of the data is split taking into account their ratio
splitRatios=[0.65,0.15,0.20]

#This flag defines if the same number of samples will be used for each class in training
#   If True, data augmentation(via up-sampling some existing files) will be carried
#   to have balanced set for training and validation. Not applicable to test files
useBalancedData=True

#Define segmentation strategy
async2secSegments=Segmentation("None",periodSync=False,sizeType="fixed",frameSizeMs=2000.0,hopSizeMs=1000.0)

#Define features to be used
features=[]#use of multiple features is possible, that's why we use a list here
timeDim=64
freqDim=16
#This implementation uses only the sub-band envelopes feature
# other features can be appended here
features.append(Feature('SubEnv',[timeDim,freqDim],"frame",async2secSegments))

#Define data specifications for this database
data=Data(dbaName,dataFolder,featureFolder,features,useBalancedData,splitRatios,info)
#Defining NN model with a name. 
#   Implementation is in models.py. Feel free to add your own models and 
#   test by just changing the name here
modelName='uocSeq2'

#Running random split and testing several times (1/testSetPercentage)
# ex: if test set is 20%, tests will be repeated 5 times
numExperiments=int(1/splitRatios[-1])
for i in range(numExperiments):
    #Define test specifications and run
    singleTest=Test(modelName,data,resultsFolder,batch_size=128,num_epochs=50)
    #Run the tests: outputs will be put in the results folder
    singleTest.run()
    #Cleaning this test sessions' intermediate files
    cleanFilesOfPrevSess([dataFolder])
