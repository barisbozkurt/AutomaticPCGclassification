#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017

@author: Baris Bozkurt
"""
import soundfile as sf
import numpy as np

class Segmentation(object):
    def __init__(self,sourceType="ecg",periodSync=True,sizeType="1period",frameSizeMs=1000.0,hopSizeMs=500.0):
        '''Constructor

        Args:
            source (str): source on which segmentation is computed from,
                ex: "ecg", "pcg"
            periodSync (bool): specifies if windows are period sync
            sizeType (str): "1period","2periods", "3periods", "fixed"
            frameSizeMs (float): size of frame/window in milliseconds [not needed for "1period","2periods", "3periods"]
            hopSizeMs (float): hop-size/window-shift in milliseconds [not needed for "1period","2periods", "3periods"]

        An abbreviation for segmentation is created and used in file names to be
            able to store results of different tests in the same folder

        '''
        self.sourceType=sourceType.lower()
        self.periodSync=periodSync
        self.frameSizeMs=frameSizeMs
        self.frameSizeSamp=-1#not assigned
        self.hopSizeMs=hopSizeMs
        self.sizeType=sizeType.lower()
        self.wavfile=""
        self._createAbbreviation()#abbreviation for the segmentation will be used in file/folder-naming
        self.sourceFilesExists=True#initiate with assumption that source for segmentation(i.e Ecg channel) exists, modified if not found

    def _createAbbreviation(self):
        '''Creating the abbreviation string for the specific segmentation
        which will be used in file naming
        '''
        self.abbr=""
        self.abbr+=self.sourceType[0]
        if self.periodSync:
            self.abbr+='Syn'
        else:
            self.abbr+='ASyn'
        if self.sizeType=="fixed":
            self.abbr+=str(int(self.frameSizeMs))+'len'
            if not self.periodSync:
                self.abbr+=('_'+str(int(self.hopSizeMs)))+'hop'
        else:
            self.abbr+=self.sizeType[0:4]


    def extract(self,wavfile):
        '''Segmentation extraction

        Args:
            wavfile (str): source signal from which segmentation will be performed

        Assigns:
            self.startsMs, self.stopsMs, self.startsSamp, self.stopsSamp

        Note:
            Saving of segmentation is performed in Feature.compute. The segmentation
            object is directly saved in a pickle file
        '''
        self.startsMs=[]
        self.stopsMs=[]
        self.periodMarksSec=[]

        self.wavfile=wavfile
        signal, samplerate = sf.read(wavfile)
        lenSigSamp=len(signal)
        lenSigMs=1000*lenSigSamp/samplerate
        self.lenSigMs=lenSigMs
        self.frameSizeSamp=int(samplerate*self.frameSizeMs/1000)

        if self.periodSync:#period-sync segmentation--------------------
            print('Period synchronous segmentation disabled for this implementation')
            return
        else:#period-async segmentation--------------------
            if self.sizeType=='fixed':
                '''Constant frame size windowing'''
                self.startsMs=list(np.arange(0,lenSigMs-self.frameSizeMs,self.hopSizeMs))
                self.stopsMs=[x+self.frameSizeMs for x in self.startsMs]


