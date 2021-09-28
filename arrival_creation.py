# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 19:17:15 2021

@author: Usuario
"""

import pandas as pd
import copy

def arrival_creation():

    all_bids = pd.read_excel(open('Bids_energy_33.xlsx','rb'),sheet_name='T1_random',index_col=0) 
    BB_separation = pd.DataFrame(columns = ['ID','Bid','Bus','Direction','Quantity','Price','Time_target','Time_stamp','Block'])
    BB_separation.set_index('ID',inplace=True)
    all_bids_BB = pd.DataFrame(columns = ['ID','Bid','Bus','Direction','Quantity','Price','Time_target','Time_stamp','Block'])
    all_bids_BB.set_index('ID',inplace=True)
    all_bids_SB = pd.DataFrame(columns = ['ID','Bid','Bus','Direction','Quantity','Price','Time_target','Time_stamp','Block'])
    all_bids_SB.set_index('ID',inplace=True)

    # Random arrival with block bids
    block = 0
    for ID in all_bids.index:
        if all_bids.at[ID,'Block'] != 'No':
            if block == all_bids.at[ID,'Block']:
                BB_separation.loc[ID] = [all_bids.at[ID,'Bid'],all_bids.at[ID,'Bus'],all_bids.at[ID,'Direction'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                all_bids = all_bids.drop([ID], axis=0)
            else:
                block = all_bids.at[ID,'Block']
           
    all_bids_random = all_bids.sample(frac=1)
    
    time_stamp = 0
    for ID in all_bids_random.index:
        all_bids_BB.loc[ID] = [all_bids_random.at[ID,'Bid'],all_bids_random.at[ID,'Bus'],all_bids_random.at[ID,'Direction'],all_bids_random.at[ID,'Quantity'],all_bids_random.at[ID,'Price'],all_bids_random.at[ID,'Time_target'],time_stamp,all_bids_random.at[ID,'Block']]
        if all_bids_random.at[ID,'Block'] != 'No':
            for BB in BB_separation.index: 
                if BB_separation.at[BB,'Block'] == all_bids_random.at[ID,'Block']:
                    all_bids_BB.loc[BB] = [BB_separation.at[BB,'Bid'],BB_separation.at[BB,'Bus'],BB_separation.at[BB,'Direction'],BB_separation.at[BB,'Quantity'],BB_separation.at[BB,'Price'],BB_separation.at[BB,'Time_target'],time_stamp,BB_separation.at[BB,'Block']]
        time_stamp += 1
    
    # Random arrival with single bids
    all_bids_SB = copy.copy(all_bids_BB)
    for SB in all_bids_SB.index:
        all_bids_SB.at[SB,'Block'] = 'No'
    
    return (all_bids_BB,all_bids_SB)








