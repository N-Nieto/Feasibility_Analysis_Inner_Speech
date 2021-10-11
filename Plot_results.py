#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:32:03 2021

@author: Nieto Nicolás - nnieto@sinc.unl.edu.ar
"""
# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statannot.statannot import add_stat_annotation

# Setings
# Root where the data are stored
root_dir="../Results/"

# Subjets
N_S_list=[1,2,3,4,5,6,7,8,9,10]

# Compared Conditons
Conditions=[["Pron"],["Inner"]]

# Prefix figure name
fig_name="Fig2"

# Save generated plot on the root folder
save_plot = False
# In[]: Load stored data and plot
N_aux=0

Cond_0 = Conditions[0][0] 

Cond_1 = Conditions[1][0] 

for N_S in  N_S_list:
    
    # Random label results
    file_name = root_dir + "ACC_TST_" + Cond_0 + "_vs_" + Cond_1 + "_Random_label_True_Subject_" + str(N_S)+ ".npy"
 
    results_rand = np.load(file_name)
    
    results_s_rand= np.vstack([100*results_rand, int(N_S)*np.ones(results_rand.shape[0],dtype=int), True*np.ones(results_rand.shape[0])])
    print("RANDOM Subject: "+str(N_S)+" ACC: " + str(np.mean(results_s_rand[0,:])) + " STD: "+ str(np.std(results_s_rand[0,:])))

    # Real labels results
    file_name = root_dir + "ACC_TST_" + Cond_0 + "_vs_" + Cond_1 + "_Random_label_False_Subject_" + str(N_S)+ ".npy"
 
    results = np.load(file_name) 
    
    results_s= np.vstack([100*results,int(N_S)*np.ones(results.shape[0],dtype=int), False*np.ones(results.shape[0])])
    
    # Avoiding 100 in all the CV to get a standar deviation 
    if sum(results_s[0,:]) == 100 * results_s.shape[1]:
        results_s[0,0] = 99.9

    
    print("REAL Subject: "+str(N_S)+" ACC: " + str(np.mean(results_s[0,:])) + " STD: "+ str(np.std(results_s[0,:])))
    print("····")
    
    results_s_final = np.hstack([results_s_rand,results_s])
        
    if N_aux==0:
        results_final=results_s_final
    else:
        results_final = np.hstack([results_final,results_s_final])
    N_aux = N_aux+1
    

# In[]: Plot
results_final = results_final.T

df=pd.DataFrame(results_final,columns=["Test Accuracy [%]","Subject","Random Label"])

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax = sns.violinplot(data = df, x= round(df["Subject"]) , y = "Test Accuracy [%]", hue = "Random Label",
                    split = True, inner = None, ax = ax, scale = "width",)

ax.grid(b = True, color = 'black', linestyle = ':', linewidth = 1, alpha = 0.4)

ax.legend([], [], framealpha = 0)

fig.legend(["Real", "Random"], bbox_to_anchor=(0.412, 0.515, 0.5, 0.5))

ax.set_xticklabels(N_S_list)

# Comparisons for statistical test
box_list = [((N_S, 0), (N_S,1)) for N_S in N_S_list]

add_stat_annotation(ax, data = df, x = "Subject", y = "Test Accuracy [%]",  hue = "Random Label",
                    box_pairs = box_list, test = 'Mann-Whitney', text_format = 'star', loc = 'inside', 
                    verbose=1, pvalue_thresholds = [[1,"ns"], [0.01,"*"]])

if save_plot:
    # File name for saving
    file_name= root_dir + fig_name + Cond_0 + "_vs_" + Cond_1 + ".pdf"
    # Save figure
    plt.savefig(file_name,pad_inches=0.5) 


