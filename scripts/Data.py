#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017

@author: Baris Bozkurt
"""
import os
import pickle
import random
from random import randint
import numpy as np
from keras.utils import to_categorical
import soundfile as sf
import resampy

class Data(object):
    '''Data specification, loading and processing functions
    '''
    def __init__(self,name,wavFolder,featureFolder,features,useBalanced=True,splitRatios=[0.65,0.15,0.2],info="",dataAugRatio=1):
        '''Constructor

        Args:
            name (str): Name of the database. This will define how the
                data will be read. Ex: "Uoc","Physionet","UrbanSound", etc

            wavFolder (str): Path to folder containing wave files

            featureFolder (str): Path to folder containing feature files.
                If the feature files exist, they will be directly read. If they
                are missing, they will be computed and saved in subdirectories
                in this folder

            feature (list of Feature objects): Different features to be computed
                and used for this data(base).

            useBalanced (bool): Flag for using balanced data. If set, balancing
                will be performed by augmenting data with up-sampling

            splitRatios (list of floats): ratios of train, validation and test
                to applied as split ratios to the whole dba. ex: [0.65,0.15,0.20]
                If only two splits are given, it is assumed to refer to
                train-validation split and test data is separately available

            info (str): Additional info to be saved in report files

            dataAugRatio (float): Data augmentation ratio. If dataAugRatio>1,
                then this will indicate an augmentation request for the train
                data set. Augmentation will be carried via down-sampling of
                existing samples

        '''
        self.name=name
        self.wavFolder=wavFolder
        self._initiateLists()

        self.featureFolder=featureFolder
        self.features=features
        self.balancingPerformed=False
        self.splitRatios=splitRatios
        self.info=info           
        self.dataAugRatio=dataAugRatio#disabled for this implementation
        self.useBalanced=useBalanced
        self.trainSetAugmented=False

        self.numTrainSamp=0#number of train samples
        self.numValidSamp=0#number of valid samples
        self.numTestSamp=0#number of test samples
        self.numTestSampPerCat=[]#number of test samples for each category

        self.preDefFileLabelMap={}#pre-defined file-label map (used in Physionet2016 dba)


    def _initiateLists(self):
        '''Initializing lists in this object '''
        self.trainFiles=[]#list of wave files for the train set
        self.validFiles=[]#list of wave files for the validation set
        self.testFiles=[]#list of wave files for the test set

        self.trainFileLabels=[]#labels for wave files for the train set
        self.validFileLabels=[]#labels for wave files for the validation set
        self.testFileLabels=[]#labels for wave files for the test set

        #x is used for inputs/feature-samples and y for outputs/labels
        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []
        self.test_x = []
        self.test_y = []

    def loadSplittedData(self,feature):
        '''Load feature data

        First checks if the pickle files are available in the featureFolder for the feature
        (in a subdirectory with name of the feauture). If the files are available, they are loaded and the data
        is returned. If the files are not available, feature estimation is performed, saved to file and reloaded.

        Args:
            featureName (str): name of the feature for which data will be loaded

        Returns:
            dataRead (bool): flag to indicate success of feature reading

        Assigns:
            self.train_x (numpy array): Input data(feature) for train
            self.train_y (numpy array): Outputs(labels) for train
            self.valid_x (numpy array): Input data(feature) for validation
            self.valid_y (numpy array): Outputs(labels) for validation
            self.test_x (numpy array) : Input data(feature) for test
            self.test_y (numpy array) : Outputs(labels) for test
            self.num_classes (int): number of classes
        '''
        #Start by cleaning the lists, if not called, new data will be appended on previous
        self._initiateLists()

        #Checking if feature files exist for train and test.
        #   If validation does not exist, it will be obtained by applying random split of the train data

        #Creating path for feature files
        dimsStr=str(feature.dimensions[0])+'by'+str(feature.dimensions[1])
        if feature.involveDelta:
            deltaStr='del'
        else:
            deltaStr=''
        subFolder=feature.name+dimsStr+deltaStr+'_'+feature.segmentation.abbr+feature.window4frame[0]+'/'
        self.featureSubFolder=subFolder
        dirName=self.featureFolder+subFolder
        feature.checkFeatureFiles(dirName)#creates file names and checks their availability

        if (not feature.trainFeatFilesExist):
            #Feature files missing, running feature extraction

            #If feature files folder does not exist, create it
            if not os.path.exists(dirName):
                os.mkdir(dirName)

            print('Computing feature: '+subFolder)
            #Get list of files in the wavFolder subDirectories: train, validation and test
            #   Data augmentation, balancing and splitting performed in this step
            self.getWaveFilesAndLabels()

            #Running feature extraction for each file in a given file list
            #   Feature computation produces 'input' data and labels as 'true outputs'
            #   All are saved in pickle files. To be able to compute the features,
            #   segmentation is needed. Hence, segmentation files are read. If they are not
            #   available, segmentation is performed and saved into files
            #   Inexistence of a segmentation file indicates either some missing
            #   source(for example ecg-channel may be missing and ecg-based segmentation may have
            #   been targeted). This will result in skipping that file

            feature.compute('train',self.trainFiles,self.trainFileLabels)
            if len(feature.segmentation.startsMs)==0:#if source files for segmentation could not be found, feature will not be computed
                return False
            feature.compute('valid',self.validFiles,self.validFileLabels)
            feature.compute('test',self.testFiles,self.testFileLabels)
            feature.checkFeatureFiles(dirName)

        #Reading train-feature/label files
        with open(feature.trainFeatureFile, 'rb') as f:
            trainExamples=pickle.load(f)
        with open(feature.trainLabelFile, 'rb') as f:
            trainLabels=pickle.load(f)
        with open(feature.trainMapFile, 'rb') as f:
            train_patSegMaps=pickle.load(f)
        #feature dimensions
        dims=trainExamples[0].shape[0:2]
        if feature.involveDelta:
            featDims=(feature.dimensions[0],feature.dimensions[1]*2)
        else:
            featDims=(feature.dimensions[0],feature.dimensions[1])
        #Check match with feature dimension
        if featDims!=dims:
            print('Error: data dimensions ',dims,', does not match feature dimensions ', featDims)
        self.shape = (dims[0], dims[1], 1)

        #Number of distinct classes in data
        self.num_classes = len(np.unique(trainLabels))

        #if validation feature,label files exist, read them
        if(feature.validFeatFilesExist):
            with open(feature.validFeatureFile, 'rb') as f:
                self.valid_x=pickle.load(f)
            with open(feature.validLabelFile, 'rb') as f:
                self.valid_y=pickle.load(f)
            with open(feature.validMapFile, 'rb') as f:
                self.valid_patSegMaps=pickle.load(f)

        #if test feature,label files exist, read them
        if(feature.testFeatFilesExist):
            with open(feature.testFeatureFile, 'rb') as f:
                self.test_x=pickle.load(f)
            with open(feature.testLabelFile, 'rb') as f:
                self.test_y=pickle.load(f)
            with open(feature.testMapFile, 'rb') as f:
                self.test_patSegMaps=pickle.load(f)

        #if valid and test files do not exist, perform splitting of the trainSamples into three
        if ( not feature.testFeatFilesExist) and (not feature.validFeatFilesExist):
            #to be implemented if a database contains just the train features (no validation or test)
            pass

        #if only train and test files exist, split train in two to get validation
        if (feature.testFeatFilesExist) and (not feature.validFeatFilesExist):
            print("Train data split into train - validation")
            #splitting data
            indexes=list(range(trainExamples.shape[0]))#phyton 3.5: range creates an object, in python 2.7 a list
            random.shuffle(indexes)
            splitRatio=self.splitRatios[0]# ratio of the train samples within the whole
            self.numTrainSamp=int(trainExamples.shape[0]*splitRatio)
            self.numValidSamp=trainExamples.shape[0]-self.numTrainSamp
            trainIndexes=indexes[:self.numTrainSamp]
            validIndexes=indexes[self.numTrainSamp:]

            #type conversions applied to fit to Keras input specifications
            self.train_x=trainExamples[trainIndexes];self.train_x = self.train_x.astype('float32')
            self.valid_x=trainExamples[validIndexes];self.valid_x = self.valid_x.astype('float32')

            self.train_y=trainLabels[trainIndexes];self.train_y=self.train_y.astype('uint8')
            self.valid_y=trainLabels[validIndexes];self.valid_y=self.valid_y.astype('uint8')

            # convert class vectors to binary class matrices
            self.train_y = to_categorical(self.train_y, self.num_classes)
            self.valid_y = to_categorical(self.valid_y, self.num_classes)

        #if all exist, just copy trainSamples to train data without split
        if (feature.testFeatFilesExist) and (feature.validFeatFilesExist):
            self.train_x=trainExamples
            self.train_y=trainLabels
            self.train_patSegMaps=train_patSegMaps

        return True#features computed, files created, features loaded
        #   return value is False if segmentation was not available and feature
        #   extraction could not be performed

    def getWaveFilesAndLabels(self):
        '''Gathers file lists for train, valid and test

        This function assumes the following presentation for the database:
            File lists for wave files are presented in "train.txt","validation.txt", "test.txt"
            Each list file contains a row which is composed of the filename and the label separated by a tab
            If the files do not exist, they will be created

        Assigns:
            self.trainFiles (list): List of wave files for the train set
            self.validFiles (list): List of wave files for the validation set
            self.testFiles (list): List of wave files for the test set

            self.trainFileLabels (list):labels for wave files for the train set
            self.validFileLabels (list):labels for wave files for the validation set
            self.testFileLabels (list):labels for wave files for the test set

        '''
        trainListFile=self.wavFolder+"train.txt"
        validListFile=self.wavFolder+"validation.txt"
        testListFile=self.wavFolder+"test.txt"
        #if list-files do not exist, attempt to create
        if ((not os.path.exists(trainListFile)) and (not os.path.exists(validListFile)) and (not os.path.exists(testListFile)) ):
            self._createFileLists(self.wavFolder)

        #If balanced data will be used, perform data augmentation using upsampling
        if self.useBalanced and (not self.balancingPerformed):
            self._augmentForBalancing(trainListFile)#augmentation of train for making it balanced
            self._augmentForBalancing(validListFile)#augmentation of validation for making it balanced

        #Reading file-lists from files
        if os.path.exists(trainListFile):
            fid_listFile=open(trainListFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                self.trainFiles.append(os.path.join(self.wavFolder, name))
                self.trainFileLabels.append(label)
            fid_listFile.close()

        if os.path.exists(validListFile):
            fid_listFile=open(validListFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                self.validFiles.append(os.path.join(self.wavFolder, name))
                self.validFileLabels.append(label)
            fid_listFile.close()

        if os.path.exists(testListFile):
            fid_listFile=open(testListFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                self.testFiles.append(os.path.join(self.wavFolder, name))
                self.testFileLabels.append(label)
            fid_listFile.close()

        if(len(self.trainFiles)==0):
            print("Error: train files could not be found in expected format. The database should either file list files or stored in specific subfolders: train, validation and test")

        print('Number of files for train(augmented), validation(augmented) and test: ',[len(self.trainFiles),len(self.validFiles),len(self.testFiles)])

    def _createFileLists(self,dirName):
        '''File list creation

        1)If the directory(dirName) includes 'train', 'validation','test' directories
        file-lists will be created for files in these folders
        
        1)If the directory(dirName) includes 'test' and not 'validation' directories
        train will be splitted into train and validation using their relative sizes defined in split sizes

        3)If those subdirectories do not exist, a single voulume of data without splits will be assumed
        and random splitting will be performed to gather file-lists

        For each of the cases, file lists (together with class information) is
        written to train.txt, validation.txt and test.txt
        It would be a good practice to check their content to make sure splitting operation is correctly handled

        '''
        if self.name=='Physionet':
            '''For Physionet-2016 data, labels are stored in csv files
            First these label files are read and this information is further used in _getLabel
            '''
            for root, dirs, files in os.walk(dirName):
                for file in files:
                    if file.lower().endswith('.csv'):
                        fileFullName=os.path.join(root, file)
                        with open(fileFullName) as f:
                            lines = f.readlines()
                        for line in lines:
                            tokens=line.strip().split(',')
                            val=int(tokens[1])
                            val=int((val+1)/2)#conversion from [-1,1](normal,abnormal) to [0,1]
                            self.preDefFileLabelMap[tokens[0]+'.wav']=val

        #Checking options
        splitNames=['train','validation','test']
        folderExists={}
        for subFolder in splitNames:
            if os.path.exists(dirName+subFolder):
                folderExists[subFolder]=True
            else:
                folderExists[subFolder]=False
        
        #Testing option 1
        if(folderExists['train'] and folderExists['validation'] and folderExists['test']):
            for subFolder in splitNames:
                if os.path.exists(dirName+subFolder):
                    listFile=open(dirName+subFolder+'.txt','w')
                    for root, dirs, files in os.walk(dirName+subFolder):
                        for file in files:
                            if file.lower().endswith('.wav') and (not 'ecgChn' in file):
                                fileFullName=os.path.join(root, file)
                                label=self._getLabel(root,file)
                                listFile.write('%s\t%s\n' % (fileFullName,label))
                    listFile.close()

        #Testing option 2
        if(folderExists['test']):
            #Creating test set list file
            subFolder='test'
            if os.path.exists(dirName+subFolder):
                listFile=open(dirName+subFolder+'.txt','w')
                for root, dirs, files in os.walk(dirName+subFolder):
                    for file in files:
                        if file.lower().endswith('.wav') and (not 'ecgChn' in file):
                            fileFullName=os.path.join(root, file)
                            label=self._getLabel(root,file)
                            listFile.write('%s\t%s\n' % (fileFullName,label))
                listFile.close()
            #Split ratio modified not to include a split for test when 'test' folder exists
            self.splitRatios[0]=self.splitRatios[0]/(self.splitRatios[0]+self.splitRatios[1])
            self.splitRatios[1]=self.splitRatios[1]/(self.splitRatios[0]+self.splitRatios[1])
            self.splitRatios[2]=0

        #Testing option 3:
        if (not folderExists['validation']):
            #Gathering file-label list and applying random split such that 
            #   each set has similar distribution for each category
            #   ex: if split ratio for train is 0.7, 70 percent random selection
            #   will be performed from each label(normal, pathological)
            fileLabelDict={}
            labels=[]
            if(folderExists['train']):#if train sub-folder exists use that, if not use the whole directory
                subDir='train'
            else:
                subDir=''
            for root, dirs, files in os.walk(dirName+subDir):
                for file in files:
                    if file.lower().endswith('.wav') and (not 'ecgChn' in file):
                        fileFullName=os.path.join(root, file)
                        label=self._getLabel(root,file)
                        fileLabelDict[fileFullName]=label
                        labels.append(label)
            #set targeted number of samples per label for train, validation and test sets
            uniqueLabels, counts = np.unique(labels, return_counts=True)
            uniqueLabels=uniqueLabels.tolist()
            if len(self.splitRatios)>2:
                numTrain=[int(x*self.splitRatios[0]) for x in counts]
                numValid=[int(x*self.splitRatios[1]) for x in counts]
                numTest=[]
                for ind in range(len(numTrain)):
                    numTest.append(counts[ind]-(numTrain[ind]+numValid[ind]))
            else:
                print('!!! Random split for ratios with sizes other than 3 is not implemented yet')
            
            #APPLY SPLITS ON FILE LEVEL
            trainFilesDict={}
            while max(numTrain)>0:#add files until num2add becomes a zero-list
                #pick a random file, check if a copy is needed for that category
                randFile=random.sample(list(fileLabelDict),1)[0]
                #check if sample needed for that category
                label=fileLabelDict[randFile]
                if numTrain[uniqueLabels.index(label)]>0:#if a new file is needed for that category
                    trainFilesDict[randFile]=label#add to train set
                    del fileLabelDict[randFile]#remove that file from all-files dictionary
                    numTrain[uniqueLabels.index(label)]-=1#reduce number of new files needed for that category
            
            validFilesDict={}
            while max(numValid)>0:#add files until num2add becomes a zero-list
                #pick a random file, check if a copy is needed for that category
                randFile=random.sample(list(fileLabelDict),1)[0]
                #check if sample needed for that category
                label=fileLabelDict[randFile]
                if numValid[uniqueLabels.index(label)]>0:#if a new file is needed for that category
                    validFilesDict[randFile]=label#add to validation set
                    del fileLabelDict[randFile]#remove that file from all-files dictionary
                    numValid[uniqueLabels.index(label)]-=1#reduce number of new files needed for that category
            

            #creating train.txt file
            listFile=open(dirName+'train.txt','w')
            for (fileName,label) in trainFilesDict.items():
                listFile.write('%s\t%s\n' % (fileName,label))
            listFile.close()

            #creating validation.txt file
            listFile=open(dirName+'validation.txt','w')
            for (fileName,label) in validFilesDict.items():
                listFile.write('%s\t%s\n' % (fileName,label))
            listFile.close()

            #creating test.txt file
            if(self.splitRatios[2]>0):
                testFilesDict=fileLabelDict#rest is left to test set
                listFile=open(dirName+'test.txt','w')
                for (fileName,label) in testFilesDict.items():
                    listFile.write('%s\t%s\n' % (fileName,label))
                listFile.close()


    def _getLabel(self,root,file):
        '''Finding label/class for a specific wave file
        Database specific:
            Uoc: the label/class information is available in the sub-folder name
                normal: 0, pathological: 1
                nomur:0, murmus:1, murpat:2
            Physionet: labeles are stored in csv files
            UrbanSound: the label/class information is available in the filename
        '''
        label=''
        #get label (dba-specific)
        if self.name=='UrbanSound':
            #In UrbanSound dba, the label is coded in the file-name
            label = file.split('/')[-1].split('-')[1]
        elif self.name=='Physionet':
            return self.preDefFileLabelMap[file]
        else:
            print('Error: unknown dba-name:', self.name)
        return label

    def _augmentForBalancing(self,listFile):
        '''Augmenting the set defined by the listFile by upsampling wave files (%10-%x percent)

        Modifies the file list and creates new wav files with names *aug_*

        '''
        #reading the file names and labels
        allFiles=[]
        allFileLabels=[]
        if os.path.exists(listFile):
            fid_listFile=open(listFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                allFiles.append(os.path.join(self.wavFolder, name))
                allFileLabels.append(label)
            fid_listFile.close()

        #deduce number of files to be added to each category
        uniqueLabels, counts = np.unique(allFileLabels, return_counts=True)
        uniqueLabels=uniqueLabels.tolist()
        num2add=[max(counts)-x for x in counts]
        maxAugRatio=max(num2add)/min(counts)
        print('Data augmentation for balancing via upsampling: ',listFile.split('/')[-1])

        #creating list of new files and their labels
        files2add=[]
        files2addLabel=[]
        print('Number of files to add for each category:',num2add)
        while max(num2add)>0:#add files until num2add becomes a zero-list
            #pick a random file, check if a copy is needed for that category
            randInd=randint(0,len(allFiles)-1)
            #check if duplicate needed for that category
            label=allFileLabels[randInd]
            if num2add[uniqueLabels.index(label)]>0:#if a new file is needed for that category
                curFile=allFiles[randInd]
                #Decide random-modification percentage [should be two digits!]
                #   change amount will be coded in filename as #aug
                #   if the set will be augmented by 3, random values will be taken in range 10-16
                modifPerc=randint(10,10+int(maxAugRatio*2))
                curFile=curFile.replace('.wav','_'+str(modifPerc)+'aug_.wav')
                #if this new file has not been added yet and it is not a file already
                #   created via resampling in a previous data augmentation[see _augmentTrain()], add
                if (not curFile in files2add) and ('resampAdd' not in curFile):
                    files2add.append(curFile)
                    files2addLabel.append(label)
                    num2add[uniqueLabels.index(label)]-=1#reduce number of new files needed for that category

        print(len(files2add),' files selected for up-sampling')

        #creation of new files and adding them in the file lists
        for ind in range(len(files2add)):
            newPcgFile=files2add[ind]
            label=files2addLabel[ind]
            modifPerc=int(newPcgFile.split('aug_.wav')[0][-2:])#change amount coded in filename as #aug
            orgPcgFile=newPcgFile.replace('_'+str(modifPerc)+'aug_.wav','.wav')
            #reading, resampling and writing to new files
            data, samplerate = sf.read(orgPcgFile)
            data=resampy.resample(data, samplerate, int(samplerate*(1+modifPerc/100)), axis=-1)
            sf.write(newPcgFile, data, samplerate)#write as if sample rate has not been modified

            #add files to the list
            allFiles.append(newPcgFile)
            allFileLabels.append(label)

        #over-write final list
        listFileFID=open(listFile,'w')
        for ind in range(len(allFiles)):
            fileFullName=allFiles[ind]
            label=allFileLabels[ind]
            listFileFID.write('%s\t%s\n' % (fileFullName,label))
        listFileFID.close()

        #re-check and report number of file for each category
        #reading the file names and labels
        allFiles=[]
        allFileLabels=[]
        if os.path.exists(listFile):
            fid_listFile=open(listFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                allFiles.append(os.path.join(self.wavFolder, name))
                allFileLabels.append(label)
            fid_listFile.close()
        uniqueLabels, counts = np.unique(allFileLabels, return_counts=True)
        print('Number of files for each category in ',listFile.split('/')[-1],counts, ' after augmentation to balance')

