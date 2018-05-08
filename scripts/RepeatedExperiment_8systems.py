#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running repeated experiments for the system with settings:
    Sub-band envelopes computed from async. frames extracted with 
    length of 2 seconds or 3 seconds, with a hop size of 1 second.
    Feature dimensions:
        Time: 32 or 64
        Frequency: 16
    Models:
        'uocSeq1' or 'uocSeq2' (see models.py)

    This script tests a total of 8 systems settings, 
    repeats the tests 5 times and writes results in the data/results folder

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
dataFolder=localRepoDir+'/PhysionetData/'
if not os.path.exists(dataFolder):
    os.mkdir(dataFolder);
dbaName='Physionet'
urls={}
urls['train']='https://www.physionet.org/physiobank/database/challenge/2016/training.zip'
urls['test']='https://www.physionet.org/physiobank/database/challenge/2016/validation.zip'

#Dowloading Physionet train data
print('Downloading Physionet data (200Mb) as zip files ... ')
print('Validation dataset will be used as the Test set and train dataset will be splitted to obtain the Train and Validation sets')
print('After the download, possible duplicates(in test and train) are checked and removed')
for dataCategory in urls.keys():
    targetDir=dataFolder+'/'+dataCategory
    if not os.path.exists(targetDir):
        os.mkdir(targetDir);
        url=urls[dataCategory]
        filename=url.split('/')[-1]
        #Downloading the zip file from the url
        print('Downloading ',filename)
        urllib.request.urlretrieve(url,filename)
        #Unzipping to a specific folder
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(targetDir)
        zip_ref.close()
        os.remove(filename)#Removing the zip file
        print('Data downloaded and unzipped to: ',targetDir)
    else:
        print('Folder ',targetDir,' already exists, delete it if you want to re-download data')

#I have observed later that validation data of Physionet is in fact a subset of the training set
# so, here is the code to remove duplicates 
# !! we will assume files having the same name and size are identical

#Collect test file list together with size information
testDataFolder=dataFolder+'/test/'
testFileSizes={}
for root, dirs, files in os.walk(testDataFolder):
    for file in files:
        if file.endswith('.wav') or file.endswith('.WAV'):
            testFileSizes[file]=os.stat(os.path.join(root, file)).st_size
            
#Remove duplicates in the train folder checking file name and size
trainDataFolder=dataFolder+'/train/'
for root, dirs, files in os.walk(trainDataFolder):
    for file in files:
        if file in testFileSizes:
            fileSize=os.stat(os.path.join(root, file)).st_size
            if(fileSize == testFileSizes[file]):
                #print('File to be deleted: ', file, ' size: ',fileSize)
                os.remove(os.path.join(root, file))
                #If exists, remove also the label file
                labelFile=os.path.join(root, file.replace('.wav','.hea'))
                if os.path.exists(labelFile):
                    os.remove(labelFile)

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
async3secSegments=Segmentation("None",periodSync=False,sizeType="fixed",frameSizeMs=3000.0,hopSizeMs=1000.0)
segStrategies=[async2secSegments,async3secSegments]

#Define features to be used
features=[]
for featName in ['SubEnv']:#other options: 'MFCC','MelSpec'
    for segType in segStrategies:
        for timeDim in [32,64]:
            for freqDim in [16]:
                features.append(Feature(featName,[timeDim,freqDim],"frame",segType,involveDelta=False))


#Define data specifications for this database
data=Data(dbaName,dataFolder,featureFolder,features,useBalancedData,splitRatios,info)
#Defining NN model with a name. 
#   Implementation is in models.py. Feel free to add your own models and 
#   test by just changing the name here
modelNames=['uocSeq1','uocSeq2']

#Running random split and testing several times (1/testSetPercentage)
# ex: if test set is 20%, tests will be repeated 5 times
numExperiments=int(1/splitRatios[-1])
for i in range(numExperiments):
    for modelName in modelNames:
        #Define test specifications and run
        singleTest=Test(modelName,data,resultsFolder,batch_size=128,num_epochs=50)
        #Run the tests: outputs will be put in the results folder
        singleTest.run()
        #Cleaning this test sessions' intermediate files
        cleanFilesOfPrevSess([dataFolder])
