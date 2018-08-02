# Automatic heart sound classification for pathology detection

This repository contains python implementation of one of the systems described in:

> Bozkurt, B., Germanakis, I., & Stylianou, Y. (2018). A study of time-frequency features for CNN-based automatic heart sound classification for pathology detection. Computers in Biology and Medicine, 100(July 2017), 132–143. http://doi.org/10.1016/j.compbiomed.2018.06.026.

This is part of a larger code collection used for running all experiments(90 distinct settings) defined in our paper: system with settings: CNN classifier using feature of Sub-band envelopes computed from asyncronous frames extracted with length of 2 seconds and hop size of 1 second.  

Run the code at your own risk and please cite the publication if you use this code in your own work. This repo is not aimed for distributing a well-designed code package but more for quickly sharing part of our code for readers of our manuscript who would like to learn implementation details. 

Dependencies (python packages used): numpy, spectrum, soundfile, keras, tensorflow, matplotlib, scipy, resampy, gammatone (https://github.com/detly/gammatone). (list of packages in the environment used for testing is available in environmentPackages.txt) 

It is suggested that you create a virtual environment, install dependencies. Following that step, to run the experiments, activate your environment, cd to the 'scripts' folder and run in terminal:
$python SingleRepeatedExperiment.py

The script downloads data(~200Mb) from Physionet site (https://www.physionet.org/physiobank/database/challenge/2016/) and runs the experiment. Physionet data involves train and validation sets. Physionet-validation set is used as the test set. Physionet-train set is splitted to form train and validation sets. Data augmentation is applied to train and validation sets.

The script performs 5 random experiments with random splits of train-validation sets (test set is fixed for all experiments). The results are written in data/results folder.

If you would like to simply check results produced for one sample run (5 learning experiments) on the shared system, example resulting files are available in the 'exampleResults4Physionet2016' folder including:
- png files: plots of learning curves (acc vs epoch#, loss vs epoch#) and ROC curves
- _res.txt files: contains measures for the system including confusion matrix, sensitivity, specificity, F-measure, etc. 
- other .txt files: log of experiments

Our manuscript discusses tests carried on 90 different settings (various segmentation strategies, feature types, dimensions, CNN models) on our proprietary PCG database. The ROCs for each system setting is shared in folder "results4allSystems_UocDba" which also includes sorting with respect to area under the ROC. All ROCs are computed via averaging results of 5 random experiments (since %20 of data is reserved as test data in each experiment). Scripts folder includes a sample script for running tests for a number of different systems settings: RepeatedExperiment_8systems.py
