#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 19:05:27 2021

@author: nnieto

Classification Utilitys
"""

def randomize_trials(Y,X,randomize):
    import numpy as np
    
    if randomize:
        idx = np.random.permutation(len(Y))
        Y_t =  Y
        X_t = X[idx]
    else:
        idx = np.random.permutation(len(Y))
        X_t, Y_t = X[idx], Y[idx]
            
    return X_t, Y_t


def ELM_labels(Y):
    Y =  Y*2-1
    return Y 

def Filter_bank_CSP(X_t,bands,fs,train_index,test_index,csp,Y_train):
    import mne
    import numpy as np
    # Bandpass filtar
    for n_band in range(len(bands)):
        
        f_low = bands[n_band][0]
        f_high =bands[n_band][1]
        X_t_filtered = mne.filter.filter_data(X_t, fs,f_low,f_high)
        
        Data_train = X_t_filtered[train_index]
        Data_test = X_t_filtered[test_index]
            
        # Spatial Filter with CPS
        Data_train = csp.fit_transform(Data_train, Y_train)
        Data_test = csp.transform(Data_test)
                           
        # Stack features
        if n_band==0:
            Data_train_full = Data_train
            Data_test_full = Data_test
        else: 
            Data_train_full = np.hstack([Data_train_full, Data_train])
            Data_test_full = np.hstack([Data_test_full, Data_test])
                            
    return Data_train_full , Data_test_full

def train_ELM (W_iter,M_search,Reg,Data_train_full,Y_train,Data_test_full,Y_test):
    from RBP_Utilitys import RBP
    import numpy as np
    from sklearn.metrics import accuracy_score

    RBP=RBP()
    
    for w in range (W_iter):
        acc_trn_full=0
        acc_tst_full=0
        W_full , b_full = RBP.generate_rand_network(Data_train_full, np.max(M_search))

        for nodes in M_search: 
            W =  W_full[:,0:nodes+1] 
            b = b_full[0:nodes+1]
            # Fitting with pinv2
            B = RBP.fit(x=Data_train_full,W=W,b=b,y=Y_train,Reg=Reg)
                
            # Make a prediction with the pruned netwokr
            y_pred_train = RBP.predict(Data_train_full, W, b, B)
            
            acc_train= accuracy_score(Y_train,y_pred_train)
            
            y_pred = RBP.predict(Data_test_full, W, b, B)
            acc_test = accuracy_score(Y_test,y_pred)
            
            acc_trn_full= np.append(acc_trn_full,acc_train)
            acc_tst_full= np.append(acc_tst_full,acc_test)
            
        acc_trn_full = np.delete(acc_trn_full,0)
        acc_tst_full = np.delete(acc_tst_full,0)
        
        if w==0:
            ACC_TRN = acc_trn_full
            ACC_TST = acc_tst_full
        else: 
            ACC_TRN = np.vstack([ACC_TRN, acc_trn_full])
            ACC_TST = np.vstack([ACC_TST, acc_tst_full])
            
    return ACC_TRN , ACC_TST

def train_2_layer_ELM (W_iter,M_search,Reg,Data_train_full,Y_train,Data_test_full,Y_test):
    from RBP_Utilitys import RBP
    import numpy as np
    from sklearn.metrics import accuracy_score

    RBP=RBP()
    
    for w in range (W_iter):
        acc_trn_full=0
        acc_tst_full=0
        W_full , b_full = RBP.generate_rand_network(Data_train_full, np.max(M_search))

        for nodes in M_search: 
            W =  W_full[:,0:nodes+1] 
            b = b_full[0:nodes+1]
            # Fitting with pinv2
            B = RBP.fit(x=Data_train_full,W=W,b=b,y=Y_train,Reg=Reg)
                
            # Make a prediction with the pruned netwokr
            y_pred_train = RBP.predict(Data_train_full, W, b, B)
            
            acc_train= accuracy_score(Y_train,y_pred_train)
            
            y_pred = RBP.predict(Data_test_full, W, b, B)
            acc_test = accuracy_score(Y_test,y_pred)
            
            acc_trn_full= np.append(acc_trn_full,acc_train)
            acc_tst_full= np.append(acc_tst_full,acc_test)
            
        acc_trn_full = np.delete(acc_trn_full,0)
        acc_tst_full = np.delete(acc_tst_full,0)
        
        if w==0:
            ACC_TRN = acc_trn_full
            ACC_TST = acc_tst_full
        else: 
            ACC_TRN = np.vstack([ACC_TRN, acc_trn_full])
            ACC_TST = np.vstack([ACC_TST, acc_tst_full])
            
    return ACC_TRN , ACC_TST


def train_RBP(W_iter,M_search,Reg,Data_train_full,Y_train,Data_test_full,Y_test):
    from RBP_Utilitys import RBP
    import numpy as np
    from sklearn.metrics import accuracy_score
    

    RBP=RBP()
    
    for w in range (W_iter):
        acc_trn_full=0
        acc_tst_full=0
        W_full , b_full = RBP.generate_rand_network(Data_train_full, np.max(M_search))            
        # Fitting with pinv2
        B_full = RBP.fit(x=Data_train_full,W=W_full,b=b_full,y=Y_train,Reg=Reg)
        
        for nodes in M_search: 
            W, b, B  = RBP.fix_prunning(W_full, b_full, B_full, prn_perc=nodes, mode="keep")
            # Make a prediction with the pruned netwokr
            y_pred_train = RBP.predict(Data_train_full, W, b, B)
            
            acc_train= accuracy_score(Y_train,y_pred_train)
            
            y_pred = RBP.predict(Data_test_full, W, b, B)
            acc_test = accuracy_score(Y_test,y_pred)
            
            acc_trn_full= np.append(acc_trn_full,acc_train)
            acc_tst_full= np.append(acc_tst_full,acc_test)
            
        acc_trn_full = np.delete(acc_trn_full,0)
        acc_tst_full = np.delete(acc_tst_full,0)
        
        if w==0:
            ACC_TRN = acc_trn_full
            ACC_TST = acc_tst_full
        else: 
            ACC_TRN = np.vstack([ACC_TRN, acc_trn_full])
            ACC_TST = np.vstack([ACC_TST, acc_tst_full])
            
    return ACC_TRN , ACC_TST



class Filter_bank_CSP_ELM():
    
    import mne
    import numpy as np

    from sklearn.model_selection import StratifiedShuffleSplit    
    from sklearn.metrics import accuracy_score
  
    from mne.decoding import CSP
    from sklearn.preprocessing import MinMaxScaler
    from Relevance_Based_Pruning.Python_Implementation.RBP_Utilitys import RBP

    
    def __init__(self, bands, fs=254, k_folds=5, test_size=0.2, random_state=23, M_max=500, Reg=0, CSP_COMP=6):
        
        self.fs = fs
        self.bands = bands
        self.k_folds = k_folds
        self.test_size = test_size
        self.random_state = random_state
        self.M_max = M_max
        self.Reg = Reg
        self.CSP_COMP = CSP_COMP
        self.str_cv = self.StratifiedShuffleSplit(self.k_folds,test_size=self.test_size,random_state=self.random_state)
        n_components = self.CSP_COMP
        rank = None
        reg='empirical'
        log=True
        norm_trace=False
        cov_est='concat'
        transform_into = 'average_power'
        self.csp = self.CSP(n_components=n_components,rank=rank, reg=reg, log=log, norm_trace=norm_trace, cov_est=cov_est,transform_into=transform_into)
        self.RBP = self.RBP()
        self.scaler= self.MinMaxScaler()
        return

    
    def fit(self, x, y):
        from sklearn.metrics import accuracy_score
        CV = 0  
        # internal Cross validation
        for train_index, val_index in self.str_cv.split(X=x,y=y):
            
            Y_train =  y[train_index]
            Y_val= y[val_index]
  
            f_low = self.bands[0]
            f_high = self.bands[1]
            X_t_filtered = self.mne.filter.filter_data(x, self.fs, f_low, f_high)
            
            Data_train = X_t_filtered[train_index]
            Data_val = X_t_filtered[val_index]
                
            # Spatial Filter with CPS
            Data_train = self.csp.fit_transform(Data_train, Y_train)
            Data_val = self.csp.transform(Data_val)
                                   
            Data_train = self.scaler.fit_transform(Data_train)
            Data_val= self.scaler.transform(Data_val)
            

            acc_val = 0
            M_search = range(self.M_max)
            W_full , b_full = self.RBP.generate_rand_network(Data_train, self.M_max)
            
            for nodes in M_search: 
               W =  W_full[:,0:nodes+1] 
               b = b_full[0:nodes+1]
               # Fitting with pinv2
               B = self.RBP.fit(Data_train,W,b,Y_train,self.Reg)
                                          
               y_pred = self.RBP.predict(Data_val, W, b, B)
               
               acc_val = self.np.append(acc_val,accuracy_score(Y_val,y_pred))


            acc_val = self.np.delete(acc_val,0)
            
            acc_max = self.np.max(acc_val)
            
            acc_max_pos = self.np.where(acc_val==acc_max)
            
            pos = self.np.min(acc_max_pos)

            if CV==0:
                ACC_VAL_CV = acc_max
                POS_MAX = pos
            else: 
                ACC_VAL_CV = self.np.vstack([ACC_VAL_CV, acc_max])
                POS_MAX = self.np.vstack([POS_MAX, pos])
            
            CV = CV +1
            
     
        Best_M = int(round(self.np.mean(POS_MAX)))
        
      
        self.W , self.b = self.RBP.generate_rand_network(Data_train, Best_M)
        # Fitting with pinv2
        self.B = self.RBP.fit(x=Data_train, W=self.W, b=self.b, y=Y_train, Reg=self.Reg)
        
        self.ACC_TST_CV = ACC_VAL_CV
        self.POS_MAX = POS_MAX
        
        return 
    
    
    def predict(self, x):
        
        # Filter
        f_low = self.bands[0]
        f_high = self.bands[1]
        x = self.mne.filter.filter_data(x, self.fs, f_low, f_high)

        # Spatial Filter with CPS
        Data = self.csp.transform(x)
                           
        Data = self.scaler.transform(Data)
        
        y = self.RBP.predict(Data, self.W, self.b, self.B)

        return y
        
    def predict_proba(self, x):
        

        f_low = self.bands[0]
        f_high = self.bands[1]
        x = self.mne.filter.filter_data(x, self.fs, f_low, f_high)

        # Spatial Filter with CPS
        Data = self.csp.transform(x)
                          
                
        Data= self.scaler.transform(Data)
        
        y = self.RBP.predict_proba(Data, self.W, self.b, self.B)

        return y
    
# =============================================================================
# class EU():
#     from pyriemann.estimation import Covariances
#     from pyriemann.utils.base import invsqrtm
#     import numpy as np   
#     def __init__():
#         return
#     
#     def fit (self,Xtr):
#     
#         self.cov_tr = self.Covariances().transform(Xtr)
#         self.Ctr = self.cov_tr.mean(0)
# 
#         return
# 
#     def fit_transform(self,X_trn):
#         
#         self.cov_tr = self.Covariances().transform(X_trn)
#         self.Ctr = self.cov_tr.mean(0)
#         Xtr_eu = self.np.asarray([self.np.dot(self.invsqrtm(self.Ctr), epoch) for epoch in X_trn])
#         
#         return Xtr_eu
#     
#     def transform(self,X_tst):
#         
#         X_tst = self.np.asarray([self.np.dot(self.invsqrtm(self.Ctr), epoch) for epoch in X_tst])
#     
#         return X_tst
#     
# =============================================================================

