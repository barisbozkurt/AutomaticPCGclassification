#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wave processing function  definitions

Created on May 15th 2017

@author: Baris Bozkurt
"""
import soundfile as sf#using pysound package for sound IO, doc: http://pysoundfile.readthedocs.io/
import numpy as np
from spectrum import create_window#http://www.thomas-cokelaer.info/software/spectrum/html/contents.html
import matplotlib.pyplot as plt

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
   
def extractPCGSegments(pcgFileName,segBoundsStart,segBoundsEnd,NFFT,maxSegLenMs=800,PlotOn=False):
    """Extraction of PCG signal segments
    
    Takes in a Pcg wavefile name and segmentation boundaries in 
    'segBoundsStart' an 'segBoundsEnd' , ith segment boundaries are:
        segBoundsStart[i] and segBoundsEnd[i]
    
    maxSegLen defines the length of time-domain signal that will be returned 
    as the output. If specified by the client, it will be used. If not 
    specified(i.e. has the value -1) segment size will be set to sampling freq.
    which will correspond to 1 second of segment. If segment is longer than 
    target, it will be trimmed, else zero-padded
    
    NFFT: number of FFT points. Amplitude spec. of each segment is also returned 
    
    Segments(as time-domain and spectral-domain data) stored in numpy arrays and returned
    
    maxSegLenMs: maximum segment length in miliseconds
    
    If PlotOn flag is set: For facilitating data investigation, a plot that contains time-domain and
    frequency domain sub-plots is created and saved in the same folder
    """
    tukeyWinR=0.2#a value in range [0,1] that specifies fade in-out region portion of the window
    data, samplerate = sf.read(pcgFileName)

    #convert miliseconds to number of samples
    maxSegLen=int(maxSegLenMs*samplerate/1000.0)
    
    #creating empty array to stack segments
    timeSegArr=np.array([])
    specSegArr=np.array([])
    
    if PlotOn:
        # Subplots sharing both x/y axes
        f, axarr = plt.subplots(len(segBoundsStart),2,figsize=(5, 10))
        spectrumCutFreq=200#defines range in spectrum plot
        spectrumCutInd=int(spectrumCutFreq*NFFT/samplerate)
        axarr[0,0].set_title('Time-dom. segments')
        axarr[0,1].set_title('Normalised amp. spec.')
        #creating x-axis series for plotting spectrum
        freqAx=np.linspace(0,samplerate/2,NFFT/2)
        freqAx=freqAx[0:spectrumCutInd]   
    
    for ind in range(len(segBoundsStart)):
        start=segBoundsStart[ind]
        stop=segBoundsEnd[ind]
        segSig=data[start:stop]
        if len(segSig)>maxSegLen:#trim data longer than targeted len
            segSig=segSig[0:maxSegLen]
        winFunc=create_window(len(segSig),'tukey',r=tukeyWinR)
        segSig=segSig*winFunc
        segSig=np.concatenate([segSig, np.array([0]*(maxSegLen-len(segSig)))],axis=0)
        
        #computing amplitude spectrum for the positive frequencies
        fft_segSig=np.fft.fft(segSig,NFFT)
        fft_segSig=np.log10(abs(fft_segSig[range(int(NFFT/2))])+np.finfo(float).eps)
        #adding the segment to the arrays to be returned
        if ind==0:
            timeSegArr=np.hstack((timeSegArr, segSig))
            specSegArr=np.hstack((specSegArr, fft_segSig))
        else:
            timeSegArr=np.vstack((timeSegArr, segSig))
            specSegArr=np.vstack((specSegArr, fft_segSig))
            
        if PlotOn:
            axarr[ind,0].plot(segSig/max(segSig))
            axarr[ind,1].plot(freqAx,fft_segSig[0:spectrumCutInd]/max(fft_segSig))
            
    if PlotOn:
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-2]], visible=False)
        axarr[ind,0].set_xlabel('Samples [n], Fs='+str(samplerate))
        axarr[ind,1].set_xlabel('Frequency(Hz)')
        #plt.savefig(fileName.replace('.wav','.jpg'),dpi=300)#use this for creating higher resolution images
        plt.show()
        f.savefig(pcgFileName.replace('.wav','seg.png'))
    
    return (timeSegArr,specSegArr)
        
        







