#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017
@author: Baris Bozkurt
"""
import os
import pickle
import soundfile as sf
import numpy as np
from scipy.signal import hilbert
from spectrum import create_window
from gammatone.filters import make_erb_filters,erb_filterbank,centre_freqs
from keras.utils import to_categorical

class Feature(object):
    ''' '''
    def __init__(self,name,dimensions,signalLevel,segmentation,involveDelta=False,window4frame='tukey'):
        '''Constructor

        Args:
            name (str): Name of the feature, ex: "mfcc", "pncc",...
            dimensions (list): Time and frequency dimensions of the feature,
                ex: [48 16] refers to 48 time-segments and 16 freq.-bands
                    [48] refers to a single dimensional array
            signalLevel (str): level for which feature is computed,
                should be "file" or "frame"
            segmentation (Segmentation object): type of segmentation applied on wave
                files. Ex: Segmentation("ecg",periodSynch=True,sizeType="1period")
            involveDelta (bool): involving delta features or not
            window4frame (str): Window function to be applied to each frame before feature extraction

        '''
        self.name=name
        self.signalLevel=signalLevel
        self.segmentation=segmentation
        self.dimensions=dimensions
        if not(name=='MFCC' or name=='MelSpec') and involveDelta:
            involveDelta=False
            print('Only MFFC and MelSpec can include delta coeffs... involveDelta disabled for '+name)
        self.involveDelta=involveDelta
        #flags for availability of feature files
        self.trainFeatFilesExist=False
        self.testFeatFilesExist=False
        self.validFeatFilesExist=False
        self.window4frame=window4frame#window function that will be applied to the frame before feature extraction

    def checkFeatureFiles(self,folderName):
        '''Checking availability of feature files
        If the files are available, the features will not be re-computed

        Args:
            folderName (str): Folder name where feature files will be searched
        Assigns:
            self.trainFeatFilesExist, self.validFeatFilesExist, self.testFeatFilesExist
        '''
        self.trainFeatureFile=folderName+'trainFeatures_'+self.name+'.pkl'
        self.trainLabelFile=folderName+'trainLabels_'+self.name+'.pkl'
        self.trainMapFile=folderName+'trainFrmFileMaps_'+self.name+'.pkl'
        if os.path.exists(self.trainFeatureFile) and os.path.exists(self.trainLabelFile) and os.path.exists(self.trainMapFile):
            self.trainFeatFilesExist=True
        else:
            self.trainFeatFilesExist=False

        self.validFeatureFile=folderName+'validFeatures_'+self.name+'.pkl'
        self.validLabelFile=folderName+'validLabels_'+self.name+'.pkl'
        self.validMapFile=folderName+'validFrmFileMaps_'+self.name+'.pkl'
        if os.path.exists(self.validFeatureFile) and os.path.exists(self.validLabelFile) and os.path.exists(self.validMapFile):
            self.validFeatFilesExist=True
        else:
            self.validFeatFilesExist=False

        self.testFeatureFile=folderName+'testFeatures_'+self.name+'.pkl'
        self.testLabelFile=folderName+'testLabels_'+self.name+'.pkl'
        self.testMapFile=folderName+'testFrmFileMaps_'+self.name+'.pkl'
        if os.path.exists(self.testFeatureFile) and os.path.exists(self.testLabelFile) and os.path.exists(self.testMapFile):
            self.testFeatFilesExist=True
        else:
            self.testFeatFilesExist=False

    def compute(self,splitName,filelist,fileLabels):
        '''Feature computation for files in the filelist

        Args:
            splitName (str): 'train', 'valid' or 'test'
            filelist (list): List containing file names(fullpath)
            fileLabels (list): List of labels(of categories) for each file

        Writes the outputs to files:
            self.trainFeatureFile, self.trainLabelFile, self.trainMapFile, etc.
        '''

        timeDim=self.dimensions[0]
        freqDim=self.dimensions[1]
        if self.involveDelta:
            freqDim=freqDim*2

        #initializing feature vectors
        allFeatures=np.zeros((1,timeDim,freqDim))
        allLabels=[]
        fileSegmentMap={}#map containing filename versus indexes of segments/features within all samples in this set
        globalInd = 0
        for ind in range(len(filelist)):#for each file
            if(ind!=0 and ind%50==0):
                print('Number of files processed: ',ind)
            file=filelist[ind]
            label=fileLabels[ind]

            #segmentation: if file does not exist, create it by running segmentation
            segFile=file.replace('.wav','_'+self.segmentation.abbr+'_seg.pkl')
            if not os.path.exists(segFile):
                self.segmentation.extract(file)
                if len(self.segmentation.startsMs)==0:#if segmentation could not be performed skip the process
                    continue
                #writing segmentation result as a Segmentation object
                pickleProtocol=1#choosen for backward compatibility
                with open(segFile, 'wb') as f:
                    pickle.dump(self.segmentation, f, pickleProtocol)

            #loading segmentation info from file
            with open(segFile, 'rb') as f:
                self.segmentation=pickle.load(f)

            #Frame-level feature extraction
            # Extract signal segments and apply feature extraction on each
            # Gather the result in allFeatures
            if (not self.segmentation.sizeType=='wholefile'):
                sig, samplerate = sf.read(file)#read wave signal
                self.samplerate=samplerate

                #windowing using segmentation info and performing feature extraction
                starts=[int(round(x*samplerate/1000)) for x in self.segmentation.startsMs]
                stops=[int(round(x*samplerate/1000)) for x in self.segmentation.stopsMs]
                for ind in range(len(starts)):
                    segment=sig[starts[ind]:stops[ind]]
                    #applying windowing function to the segment
                    if self.window4frame=='tukey':#windowing with Tukey window, r=0.08
                        segment=segment*create_window(stops[ind]-starts[ind],'tukey',r=0.08)
                    #windowing with Hanning window: to suppress S1 in 1-period frame length cases
                    elif self.window4frame=='hanning':
                        segment=segment*create_window(stops[ind]-starts[ind],'hanning')
                    if(np.max(segment)>0):#normalization
                        segment=segment/np.max(segment)
                    feature=self._computeSingleFrameFeature(segment)
                    #adding computed feature
                    if globalInd==0:#if this is the first feature assign it directly
                        allFeatures[0]=feature
                    else:#add one more element in the feature vector and then assign
                        allFeatures=np.vstack([allFeatures,np.zeros((1,timeDim,freqDim))])
                        allFeatures[globalInd]=feature
                    #adding segment to file-segment map
                    if file in fileSegmentMap:#if file already exists, append segment
                        val=fileSegmentMap[file]
                        val.append(globalInd)
                        fileSegmentMap[file]=val
                    else:#file does not exist in map, add the first file-segment map
                        fileSegmentMap[file]=[globalInd]
                    allLabels.append(label)
                    globalInd+=1
            else:#File-level feature extraction
                pass

        #If no data is read/computed at this point skip the rest [this happens when ecg signal is not available and segmentation on ecg was targeted for a test]
        if len(allLabels)==0:
            return

        #re-formatting feature vectors
        allFeatures=allFeatures.reshape(allFeatures.shape[0],timeDim,freqDim,1)
        allLabels=np.array(allLabels,dtype = np.int)
        allLabels = to_categorical(allLabels)

        #Writing to files
        if splitName=='train':
            featureFile=self.trainFeatureFile
            labelFile=self.trainLabelFile
            mapFile=self.trainMapFile
        elif splitName=='valid':
            featureFile=self.validFeatureFile
            labelFile=self.validLabelFile
            mapFile=self.validMapFile
        elif splitName=='test':
            featureFile=self.testFeatureFile
            labelFile=self.testLabelFile
            mapFile=self.testMapFile
        else:
            print('Error: split-name should be train, valid or test')
        #saving features, labels and maps to pickle files
        pickleProtocol=1#choosen for backward compatibility
        with open(featureFile, 'wb') as f:
            pickle.dump(allFeatures, f, pickleProtocol)
        with open(labelFile , 'wb') as f:
            pickle.dump(allLabels, f, pickleProtocol)
        with open(mapFile, 'wb') as f:
            pickle.dump(fileSegmentMap, f, pickleProtocol)
        print('--- ',splitName,' features computed')

    def _computeSingleFrameFeature(self,sig):
        '''Feature computation for a single time-series frame/segment

        Args:
            sig (numpy array): The signal segment for which feature will be computed
        Returns:
            feature (numpy array): Computed feature vector
        '''


        if self.name=='SubEnv':
            '''Sub-band envelopes feature computation'''
            #Computing sub-band signals
            timeRes=self.dimensions[0]
            numBands=self.dimensions[1]
            low_cut_off=2#lower cut off frequency = 2Hz
            centre_freqVals = centre_freqs(self.samplerate,numBands,low_cut_off)
            fcoefs = make_erb_filters(self.samplerate, centre_freqVals, width=1.0)
            y = erb_filterbank(sig, fcoefs)

            subenv = np.array([]).reshape(timeRes,0)
            for i in range(numBands):
                subBandSig=y[i,:]
                analytic_signal = hilbert(subBandSig)
                amp_env = np.abs(analytic_signal)
                np.nan_to_num(amp_env)
                #amp_env=resampy.resample(amp_env, len(amp_env), timeRes, axis=-1)#resampy library used resampling
                #resampling may lead to unexpected computation errors, 
                #I prefered average amplitudes for short-time windows
                downSampEnv=np.zeros((timeRes,1))
                winSize=int(len(amp_env)/timeRes)
                for ind in range(timeRes):
                    downSampEnv[ind]=np.log2(np.mean(amp_env[ind*winSize:(ind+1)*winSize]))
                subenv=np.hstack([subenv,downSampEnv])
            #removing mean and normalizing
            subenv=subenv-np.mean(subenv)
            subenv=subenv/(np.max(np.abs(subenv)))
            feature=subenv
        else:
            print('Error: feature '+self.name+' is not recognized')
            feature=[]

        return feature
