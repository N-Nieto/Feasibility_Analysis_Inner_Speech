#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 02:09:39 2021

@author: Nieto Nicolás - nnieto@sinc.unl.edu.ar
"""
import warnings
import timeit
import numpy as np
import mne

from mne.decoding import CSP

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
  
from Relevance_Based_Pruning.Python_Implementation.RBP_Utilitys import RBP

from Inner_Speech_Dataset.Python_Processing.Utilitys import Ensure_dir
from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator 

from Clasification_Utilitys import randomize_trials, ELM_labels
mne.set_log_level('CRITICAL')

warnings.filterwarnings(action= "ignore", category = DeprecationWarning ) 
# In[]
# # o ----------------- o ----------------- o ----------------- o ----------------- o ----------------- o
# Root where the data are stored
root_dir = '../'
save_dir = "../Results/"

# Subjets
N_S_list = [1,2,3,4,5,6,7,8,9,10]

# Data Parameters
datatype = "EEG"
# Compared conditions
Conditions = [["Pron"],["Inner"]]
# Use all classes regardless of the class label
Classes = [["All"],["All"]]

# Signal processing-----------------------------------------------------------
# Time cut (seg)
fs = 254
t_start = 1.5
t_end = 3.5
# Features Generation ---------------------------------------------------------
channels_list = ["all"]

# Filter signal band = [low_band , high_band, "band_name"]
# =============================================================================
bands = [
          (0.5, 4, 'Delta (0.5-4 Hz)')
         ,(4, 8, 'Theta (4-8 Hz)') 
         ,(8, 12, 'Alpha (8-12 Hz)')
         ,(12, 20, 'Low Beta (12-20 Hz)')
         ,(20, 30, 'High Beta (20-30 Hz)')
         ,(30, 45, 'Low Gamma (30-45Hz)')
          ]

# Spatial Filter --------------------------------------------------------------
# CSP parameters
n_components = 6
rank = None
reg = 'empirical'
log = True
norm_trace = False
cov_est = 'concat'
transform_into = 'average_power'

# Clasification ---------------------------------------------------------------
# Extreme Learning Machines
# Nodes to search to find the optimal
M_search = [ 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
# Regularization parameter
Reg = 1
# Number of initializations in the validations process to find the optimal number of hidden nodes.
W_iter = 10
# Number of cross validations in the loop of tuning the number of nodes
val_fold = 3
# Validations size
val_size= 0.2

# Cross Validations Parameters ------------------------------------------------
k_folds = 20
test_size = 0.2

# Randomization of labels -----------------------------------------------------
Randomize_labels = [False,True]

# Plot ------------------------------------------------------------------------
results_save = True

# In[] Fixed Variables and initializations
# Random states
random_state = 23

# Fix random seed
np.random.seed(random_state)

# 20 cross validations between train and test
str_cv = StratifiedShuffleSplit(k_folds,test_size=test_size,random_state=random_state)

# 3 cros validations for hyperparameter search
val_cv = StratifiedShuffleSplit(val_fold,test_size=val_size,random_state=random_state)

scaler = MinMaxScaler()

rbp = RBP()

csp = CSP(n_components=n_components,rank=rank, reg=reg, log=log, norm_trace=norm_trace, cov_est=cov_est,transform_into=transform_into)
      

# In[] = Load Database
for N_S in N_S_list :
    print("Loading Data from Subject "+ str(N_S))
    start = timeit.default_timer()

    # Load data
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time 
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)
    
    # Select Clases
    X , Y =  Transform_for_classificator(X, Y, Classes, Conditions)
    
    stop = timeit.default_timer()
    tim = stop - start
    print("Load Time = "+ str(tim))


    # In[] Randomization of labels
    for random_label in Randomize_labels : 
        
        # Random permutation
        X_t , Y_t = randomize_trials(Y, X, random_label)
        
        # Set labels as -1 and +1
        Y_t = ELM_labels(Y_t)
        
        # In[] Cros validation
    
        cv = 0
        Data_train_full = []
        Data_test_full = []    
        
        for train_index, test_index in str_cv.split(X = X_t, y = Y_t):
            print("Cross Validation: " +str(cv+1) +"/"+str(k_folds))
            Y_train =  Y_t[train_index]
            Y_test = Y_t[test_index]
            X_train = X_t[train_index]
            
            # In[]  Validation loop
            for train_index_val, val_index in val_cv.split(X = X_train, y = Y_train):
                
                Y_trn = Y_train[train_index_val]
                Y_val = Y_train[val_index]
                CV_VAL = 0

                # In[] Feature Generation
                # Keep selected channes
                start = timeit.default_timer()
                # Bandpass filter
                for n_band in range(len(bands)):
                    
                    f_low = bands[n_band][0]
                    f_high =bands[n_band][1]
                    
                    # Applied filter to train and test
                    X_t_filtered = mne.filter.filter_data(X_train, fs,f_low,f_high)
                    
                    # split train and val
                    Data_train = X_t_filtered[train_index_val]
                    Data_val = X_t_filtered[val_index]
                        
                    # Spatial Filter with CPS
                    Data_train = csp.fit_transform(Data_train, Y_trn)
                    Data_val = csp.transform(Data_val)
                                       
                    # Stack features
                    if n_band == 0:
                        Data_train_full = Data_train
                        Data_val_full = Data_val
                    else: 
                        Data_train_full = np.hstack([Data_train_full, Data_train])
                        Data_val_full = np.hstack([Data_val_full, Data_val])
                
                stop = timeit.default_timer()

                time = stop - start
                print("Filter Bank + CSP Time = "+ str(time))
           
                Data_train_full = scaler.fit_transform(Data_train_full)
                
                Data_val_full = scaler.transform(Data_val_full)
                    
                # In[] Finding the best number of hidden nodes M
               
            
                # W and b initializations
                for w in range(W_iter):
                    
                    # Initialization
                    acc_val_full = 0
                    
                    W , b = rbp.generate_rand_network(Data_train_full, np.max(M_search))    
                    
                    H_full = rbp.generate_H(Data_train_full, W, b)
                    
                    H_val_full = rbp.generate_H(Data_val_full, W, b)
                    
                    # Fitting with pinv
                    B_full = rbp.fit(Y_trn, H = H_full, Reg=Reg)
                    
                    # Relevance Based Pruning
                    for nodes in M_search: 
                        
                        B_prun, H_prun, H_val  = rbp.Relevance_based_pruning(B_full, prn_perc = nodes, H = H_full, H_test = H_val_full)
                        
                        # Make a prediction with the pruned netwokr                       
                        y_pred = rbp.predict(B_prun, H = H_val)
                        
                        acc_val = accuracy_score(Y_val, y_pred)
                        
                        acc_val_full = np.append(acc_val_full, acc_val)
                        
                    # Delet initialization
                    acc_val_full = np.delete(acc_val_full,0)
                    # find max validation accuracy
                    acc_max = np.max(acc_val_full)
                    # get the number of nodes where the acc was max
                    acc_max_pos = np.where(acc_val_full==acc_max)
                    # get the min position
                    pos = np.min(acc_max_pos)
                    
                    # Acumulate the results for differents initializations of W
                    if w == 0:
                        acc_max_w = acc_max
                        pos_w = pos
                    else:
                        acc_max_w = np.vstack([acc_max_w, acc_max])
                        pos_w = np.vstack([pos_w, pos])
        
                # Acumulate results for differents Cross validations
                if CV_VAL == 0:
                    ACC_VAL_CV = acc_max_w
                    POS_MAX = pos_w
                else: 
                    ACC_VAL_CV = np.vstack([ACC_VAL_CV, acc_max_w])
                    POS_MAX = np.vstack([POS_MAX, pos_w])
                
                CV_VAL = CV_VAL +1
                        
            # End of validation Loop. 
            # Get the best number of nodes as the mean of the different pos
            Best_M = M_search[int(round(np.mean(POS_MAX)))]
            
            # In[] Final clasification
            
            start = timeit.default_timer()
            
            # Bandpass filter
            for n_band in range(len(bands)):
                
                f_low = bands[n_band][0]
                f_high = bands[n_band][1]
                
                X_t_filtered = mne.filter.filter_data(X_t, fs, f_low, f_high)
                
                Data_train = X_t_filtered[train_index]
                
                Data_test = X_t_filtered[test_index]
                    
                # Spatial Filter with CPS
                Data_train = csp.fit_transform(Data_train, Y_train)
                
                Data_test = csp.transform(Data_test)
                                   
                # Stack features
                if n_band == 0:
                    Data_train_full = Data_train
                    Data_test_full = Data_test
                else: 
                    Data_train_full = np.hstack([Data_train_full, Data_train])
                    Data_test_full = np.hstack([Data_test_full, Data_test])
            
            stop = timeit.default_timer()

            time = stop - start
            
            print("Filter Bank + CSP time = "+ str(time))
          
            # Scaler
            Data_train_full = scaler.fit_transform(Data_train_full)
            Data_test_full= scaler.transform(Data_test_full)
               
            # Generate the new network
            W , b = rbp.generate_rand_network(Data_train_full, Best_M)
            
            H_full = rbp.generate_H(Data_train_full, W, b)
            
            H_test = rbp.generate_H(Data_test_full, W, b)
            
            # Fitting with all the available training data
            B = rbp.fit(Y_train, H = H_full, Reg=Reg)
            
            # get test prediction
            y_pred = rbp.predict(B , H = H_test)
            
            # Compute test accuracy
            acc_test = accuracy_score(Y_test,y_pred)
      
            # Acumulate results for cross validation
            if cv == 0:
                ACC_TST_cv = acc_test
            else: 
                ACC_TST_cv = np.append(ACC_TST_cv, acc_test)
        
            cv = cv + 1
        
        # In[]: Save results for each participant
        
        if results_save:
            
            Cond_0 = Conditions[0][0] 
            
            Cond_1 = Conditions[1][0] 
            
            rand = str(random_label)
            
            Ensure_dir(save_dir)
            
            file_name = save_dir + "ACC_TST_" + Cond_0 + "_vs_" + Cond_1 + "_Random_label_" + rand + "_Subject_" + str(N_S)+ ".npy"
    
            # Save the test accuracy
            np.save(file_name, ACC_TST_cv)
