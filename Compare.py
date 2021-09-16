# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 14:40:50 2021

@author: Usuario
"""

import pandas as pd
import numpy as np

from itertools import combinations
from operator import add
import ast


CC_dispatch = pd.read_excel(open('Dispatch.xlsx', 'rb'),sheet_name='CC_best',index_col=0)
AC_dispatch = pd.read_excel(open('Dispatch.xlsx', 'rb'),sheet_name='AC',index_col=0)
accepted_offers = pd.DataFrame(columns = ['Offer','Quantity CC','Quantity AC']) 
accepted_offers.set_index('Offer',inplace=True)
accepted_requests = pd.DataFrame(columns = ['Request','Quantity CC','Quantity AC']) 
accepted_requests.set_index('Request',inplace=True)

# print (CC_dispatch)
# print (CC_accepted_offers)

for i in CC_dispatch.index:
    offer = CC_dispatch.at[i,'Offer']
    req = CC_dispatch.at[i,'Request'] 
    if CC_dispatch.at[i,'Offer'] in accepted_offers.index:
        # print ('The offer is already there')
        
        # print (offer)
        # print (CC_accepted_offers)    
        accepted_offers.at[offer,'Quantity CC'] += CC_dispatch.at[i,'Quantity']
    else:
        #print (CC_accepted_offers)
        accepted_offers.loc[offer]=[CC_dispatch.at[i,'Quantity'],0]
    if CC_dispatch.at[i,'Request'] in accepted_requests.index:
        accepted_requests.at[req,'Quantity CC'] += CC_dispatch.at[i,'Quantity']
    else:
        accepted_requests.loc[req]=[CC_dispatch.at[i,'Quantity'],0]
        
        
# print (CC_accepted_offers)
# print (CC_dispatch)

for n in AC_dispatch.index:
    if AC_dispatch.at[n,'Bid']=='Request':
        req = AC_dispatch.at[n,'ID']
        if req in accepted_requests.index:
            accepted_requests.at[req,'Quantity AC'] += AC_dispatch.at[n,'Quantity']
        else: 
            accepted_requests.loc[req]=[0,AC_dispatch.at[n,'Quantity']]
    else:
        offer = AC_dispatch.at[n,'ID']
        if offer in accepted_offers.index:
            accepted_offers.at[offer,'Quantity AC'] += AC_dispatch.at[n,'Quantity']
        else: 
            accepted_offers.loc[offer]=[0,AC_dispatch.at[n,'Quantity']]

print (accepted_offers,accepted_requests)