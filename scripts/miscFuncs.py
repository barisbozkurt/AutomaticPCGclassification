#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper function collection

Created on May 15th 2017

@author: Baris Bozkurt
"""
import os
import numpy as np

import soundfile as sf
from spectrum import create_window

#%%

def cleanFilesOfPrevSess(dbaFolders):
    '''CLEANING intermediate files from previous test session

    !!!Use with precaution, you may end up deleting important files un-expectedly
    To avoid unexpected cases, only files with specific extensions are deleted and the
    shortest dba-path accepted is 20 characters (so that by error it does not get
    a root path and delete)

    Deletes all files with extension: h5, pkl, png and those
    end with results.txt

    Args:
        wavFolders (list of paths): Folders containing databases
    '''
    count=0
    if type(dbaFolders) is list:
        for dbaFolder in dbaFolders:#for each database folder
            if os.path.exists(dbaFolder) and len(dbaFolder)>25:
                #walking through subfolders of the folder
                for root, dirs, files in os.walk(dbaFolder):
                    for file in files:
                        fileName=os.path.join(root, file)#creating filename with path
                        if (file.endswith('_seg.pkl') or
                            file=='test.txt' or
                            file=='train.txt' or
                            file=='validation.txt' or
                            (file.endswith('.wav') and ('aug_' in file)) or
                            (file.endswith('.wav') and ('resampAdd_' in file)) or
                            (file.endswith('.pkl') and ('train' in file) and ('features' in root)) or
                            (file.endswith('.pkl') and ('valid' in file) and ('features' in root)) or
                            (file.endswith('.pkl') and ('test' in file) and ('features' in root))):
                            #print('Deleting: ',fileName)
                            os.remove(fileName)
                            count+=1
            else:
                print('Path does not exist or too short(min expected len: 25 chars): ',dbaFolder)
                print('!!! Check out !!! this process is destructive')
        print('Number of files deleted by cleanFilesOfPrevSess: ',count)
    else:
        print('Input should be a list of paths!...')

def findAllSubStrIndexes(mainStr,subStr):
    """Finding all occurences of a substring in a string(mainStr)
    """
    indexes=[];
    index=mainStr.find(subStr);
    if index != -1:
        indexes.append(index)
    else:
        return []
    while index < len(mainStr):
        index = mainStr.find(subStr, index+1)
        if index == -1:
            break
        indexes.append(index)
    return indexes

def stringNotContainsListElems(mainStr,excludeList):
    """Given a string and a list 
    checks if none of the list elements are included in the string"""
    for elem in excludeList:
        if elem in mainStr:
            return False
    return True

def findIndOfMatchElemInStr(strList,mainStr):
    """
    Find the index of the element in the string-list that has a match in the 
    mainStr
    """
    for ind in range(len(strList)):
        if strList[ind] in mainStr:
            return ind
    return -1#refers to element not found


def getTimeSegsFromWavFile(fileName,winSizeSample,hopSizeSample,FS):
    '''Windowing function to gather time segments from a wavefile
    in a numpy array.
    winSizeSample: window size in number of samples
    hopSizeSample: hop size in number of samples
    FS: expected sampling frequency for the wave files
    
    Reason for using sample sizes and a fixed sampling rate:
        It would be preferable to use seconds for sizes and set window sizes
        using the sampling frequency of each file. However, all time frames 
        from all wave files in the database is stacked in a single array which
        necessitates the same size for all time segments. Hence, number of
        samples is prefered and sampling frequency of each file is checked for
        making sure all files have the same sampling frequency
        
    Returns a numpy array containing all time segments
    '''
    timeSegArr=np.array([])
    tukeyWinR=0.2#a value in range [0,1] that specifies fade in-out region portion of the window
    winFunc=create_window(winSizeSample,'tukey',r=tukeyWinR)
    
    data, samplerate = sf.read(fileName)
    if(samplerate!=FS):
        print('Error: Sampling frequency mismatch:'+fileName)
        return timeSegArr
    
    for ind in range(0,len(data),hopSizeSample):
        if(ind+winSizeSample>len(data)):
            break
        segSig=data[ind:ind+winSizeSample]
        segSig=segSig*winFunc
        #adding the segment to the arrays to be returned
        if ind==0:
            timeSegArr=np.hstack((timeSegArr, segSig))
        else:
            timeSegArr=np.vstack((timeSegArr, segSig))
    return timeSegArr
   


