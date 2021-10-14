
# -*- coding: utf-8 -*-
"""
Created on May 2021

@author: aalarcon
"""

from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo
import pyomo.gdp as gdp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import time

def auction_BB_NC(all_bids):

    start_time = time.time()
    Setpoint = pd.read_excel(open('Setpoint.xlsx', 'rb'),index_col=0) # Baseline injections at each nodes (negative for retrieval)
    
    # Initial Social Welfare and Flexibility Procurement Cost
    Social_Welfare = 0
    Flex_procurement = 0
    
    # Index for nodes
    bus = pd.read_excel(open('network33bus.xlsx','rb'),sheet_name='Bus',index_col=0)
    nodes = list(bus.index)
    
    # Index for branches
    branch= pd.read_excel(open('network33bus.xlsx','rb'),sheet_name='Branch',index_col=0)
    lines = list(branch.index)
    
    # Upload bids
                                                                    #'T1_best','T1_worst','T1_random'
    #all_bids = pd.read_excel(open('Bids_energy_33 - copia.xlsx','rb'),sheet_name='T1_random',index_col=0) 
    
    # Create empty dataframes to contain the bids according to their type and direction
    offers = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    offers.set_index('ID',inplace=True)
    offers_up = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    offers_up.set_index('ID',inplace=True)
    offers_down = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    offers_down.set_index('ID',inplace=True)
    
    all_offers_up = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    all_offers_up.set_index('ID',inplace=True)
    all_offers_down = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    all_offers_down.set_index('ID',inplace=True)
    
    
    offers_BB = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    offers_BB.set_index('ID',inplace=True)
    offers_BB_up = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    offers_BB_up.set_index('ID',inplace=True)
    offers_BB_down = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    offers_BB_down.set_index('ID',inplace=True)
    
    requests = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    requests.set_index('ID',inplace=True)
    requests_up = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    requests_up.set_index('ID',inplace=True)
    requests_down = pd.DataFrame(columns = ['ID','Bus','Quantity','Price','Time_target','Time_stamp','Block'])
    requests_down.set_index('ID',inplace=True)
    
    AR_results = pd.DataFrame(columns = ['Block','AR'])
    AR_results.set_index('Block',inplace=True)
    
    # Bids clasification
    for ID in all_bids.index:
        
        if all_bids.at[ID,'Bid'] == 'Offer':  
            if all_bids.at[ID,'Block'] == 'No': # Classify simple offers
                offers.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                offers.sort_values(by=['Time_target'], ascending=[True], inplace=True)
                if all_bids.at[ID,'Direction'] == 'Up':
                    offers_up.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    offers_up.sort_values(by=['Time_target'], ascending=[True], inplace=True)
                    all_offers_up.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    all_offers_up.sort_values(by=['Time_target','Price'], ascending=[True,True], inplace=True)
    
                
                elif all_bids.at[ID,'Direction'] == 'Down':
                    offers_down.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    offers_down.sort_values(by=['Time_target'], ascending=[True], inplace=True) 
                    all_offers_down.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    all_offers_down.sort_values(by=['Time_target','Price'], ascending=[True,True], inplace=True)
    
            else:
                offers_BB.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                offers_BB.sort_values(by=['Time_target'], ascending=[True], inplace=True)
                if all_bids.at[ID,'Direction'] == 'Up':
                    offers_BB_up.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    offers_BB_up.sort_values(by=['Time_target'], ascending=[True], inplace=True)
                    all_offers_up.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    all_offers_up.sort_values(by=['Time_target','Price'], ascending=[True,True], inplace=True)
                    
                elif all_bids.at[ID,'Direction'] == 'Down':
                    offers_BB_down.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    offers_BB_down.sort_values(by=['Time_target'], ascending=[True], inplace=True)
                    all_offers_down.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                    all_offers_down.sort_values(by=['Time_target','Price'], ascending=[True,True], inplace=True)
    
             
        elif all_bids.at[ID,'Bid'] == 'Request':
            requests.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
            requests.sort_values(by=['Time_target'], ascending=[True], inplace=True)
            if all_bids.at[ID,'Direction'] == 'Up':
                requests_up.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                requests_up.sort_values(by=['Time_target','Price'], ascending=[True,False], inplace=True)
            elif all_bids.at[ID,'Direction'] == 'Down':
                requests_down.loc[ID] = [all_bids.at[ID,'Bus'],all_bids.at[ID,'Quantity'],all_bids.at[ID,'Price'],all_bids.at[ID,'Time_target'],all_bids.at[ID,'Time_stamp'],all_bids.at[ID,'Block']]
                requests_down.sort_values(by=['Time_target','Price'], ascending=[True,False], inplace=True)
     
    
    branch['B'] = 1/branch['X']
    n_ref = 'n1'
    
    epsilon = 0.00001 # Tolerance
        
    #%% 1- Optimization problem MILP
    def dc_opf():
        
        m = pyo.ConcreteModel()
       
        # for access to dual solution for constraints
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        
        # Sets creation
        m.T = pyo.Set(initialize = Setpoint.index, doc='Time_periods')
        m.N = pyo.Set(initialize = nodes, doc = 'Nodes')
        m.L = pyo.Set(initialize = lines)
        
        m.RU = pyo.Set(initialize = requests_up.index)
        m.RD = pyo.Set(initialize = requests_down.index)
        m.OU = pyo.Set(initialize = offers_up.index)
        m.OD = pyo.Set(initialize = offers_down.index)
        m.K = pyo.Set(initialize = np.unique(offers_BB['Block']))
        m.BB = pyo.Set(initialize = offers_BB.index)
        m.BBU = pyo.Set(initialize = offers_BB_up.index)
        m.BBD = pyo.Set(initialize = offers_BB_down.index)
        
        # Variables creation
        m.pru = pyo.Var(m.RU, domain=pyo.NonNegativeReals)
        m.prd = pyo.Var(m.RD, domain=pyo.NonNegativeReals)
        m.pou = pyo.Var(m.OU, domain=pyo.NonNegativeReals)
        m.pod = pyo.Var(m.OD, domain=pyo.NonNegativeReals)  
        #m.theta = pyo.Var(m.N,m.T, bounds=(-1.5, 1.5), domain=pyo.Reals) 
        m.theta = pyo.Var(m.N,m.T, domain=pyo.Reals) 
        m.f = pyo.Var(m.L,m.T, domain=pyo.Reals)   #directed graph
        m.AR = pyo.Var(m.K, domain=pyo.Binary)
    
        # Constraints
        m.ang_ref = pyo.ConstraintList() # Constraint 1: Reference angle
        m.flow_eq = pyo.ConstraintList() # Constraint 2: Distribution lines limits
        m.flow_bounds = pyo.ConstraintList() # Constraint 2: Distribution lines limits
        m.nod_bal = pyo.ConstraintList() # Constraint 3: Power balance per node
        #m.nod_bal.set_index('Node',inplace=True)
        m.bal_up = pyo.ConstraintList() # Constraint 4: Power balance between requests and offers up and down
        m.bal_down = pyo.ConstraintList() # Constraint 4: Power balance between requests and offers up and down
        m.pru_limit = pyo.ConstraintList() # Constraint 5: Flexibility limits for requests and offers
        m.prd_limit = pyo.ConstraintList() # Constraint 5: Flexibility limits for requests and offers
        m.pou_limit = pyo.ConstraintList() # Constraint 5: Flexibility limits for requests and offers
        m.pod_limit = pyo.ConstraintList() # Constraint 5: Flexibility limits for requests and offers
       
        m.up = pyo.ConstraintList()
        m.down = pyo.ConstraintList()
        
        # objective function 
        m.obj_cost = pyo.Objective(expr= sum(m.pou[OU]*offers_up.at[OU,'Price'] for OU in m.OU) 
                                   + sum(m.pod[OD]*offers_down.at[OD,'Price'] for OD in m.OD)
                                   #block bids
                                   + sum(m.AR[K]*(sum(offers_BB.at[BB,'Quantity']*offers_BB.at[BB,'Price'] for BB in m.BB if offers_BB.at[BB,'Block'] == K)) for K in m.K)
                                   - sum(m.pru[RU]*requests_up.at[RU,'Price'] for RU in m.RU) 
                                   - sum(m.prd[RD]*requests_down.at[RD,'Price'] for RD in m.RD), sense=pyo.minimize)
       
        for T in m.T:
            
            # Constraint 1: Reference angle
            m.ang_ref.add(m.theta[n_ref,T] == 0) 
            
            # Constraint 2: Distribution lines limits
            for L in m.L:
                m.flow_eq.add(m.f[L,T] - branch.loc[L,'B']*(m.theta[branch.loc[L,'From'],T] - m.theta[branch.loc[L,'To'],T]) == 0)
                m.flow_bounds.add(pyo.inequality(-branch.loc[L,'Pmax'],m.f[L,T],branch.loc[L,'Pmax']))
            
            # Constraint 3: Power balance per node # Setpoint + Pou - Pod - Pru + Prd - losses = 0
            for N in m.N:
         
                m.nod_bal.add(Setpoint.at[T,N] 
                              # single bids
                              + sum(m.pou[OU] for OU in m.OU if (offers_up.at[OU,'Time_target'] == T and offers_up.at[OU,'Bus'] == N)) 
                              - sum(m.pod[OD] for OD in m.OD if (offers_down.at[OD,'Time_target'] == T and offers_down.at[OD,'Bus'] == N)) 
                              - sum(m.pru[RU] for RU in m.RU if (requests_up.at[RU,'Time_target'] == T and requests_up.at[RU,'Bus'] == N)) 
                              + sum(m.prd[RD] for RD in m.RD if (requests_down.at[RD,'Time_target'] == T and requests_down.at[RD,'Bus'] == N)) 
                              # block bids
                              + sum(m.AR[K]*(sum(offers_BB_up.at[BBU,'Quantity'] for BBU in m.BBU if offers_BB_up.at[BBU,'Block'] == K and offers_BB_up.at[BBU,'Bus'] == N and offers_BB_up.at[BBU,'Time_target'] == T)
                                             - sum(offers_BB_down.at[BBD,'Quantity'] for BBD in m.BBD if offers_BB_down.at[BBD,'Block'] == K and offers_BB_down.at[BBD,'Bus'] == N and offers_BB_down.at[BBD,'Time_target'] == T))for K in m.K)
                              # line flow
                              - sum(branch.at[L,'B']*(m.theta[branch.at[L,'From'],T] - m.theta[branch.at[L,'To'],T]) for L in branch.index if branch.at[L,'From'] == N)
                              - sum(branch.at[L,'B']*(m.theta[branch.at[L,'To'],T] - m.theta[branch.at[L,'From'],T]) for L in branch.index if branch.at[L,'To'] == N) == 0)
            
            #Balance between offers and requests:
            m.up.add(sum(m.pou[OU] for OU in m.OU if offers_up.at[OU,'Time_target'] == T)
                      + sum(m.AR[K]*(sum(offers_BB_up.at[BBU,'Quantity'] for BBU in m.BBU if offers_BB_up.at[BBU,'Block'] == K and offers_BB_up.at[BBU,'Time_target'] == T)) for K in m.K)
                      == sum(m.pru[RU] for RU in m.RU if requests_up.at[RU,'Time_target'] == T))
            m.down.add(sum(m.pod[OD] for OD in m.OD if offers_down.at[OD,'Time_target'] == T)
                      + sum(m.AR[K]*(sum(offers_BB_down.at[BBD,'Quantity'] for BBD in m.BBD if offers_BB_down.at[BBD,'Block'] == K and offers_BB_down.at[BBD,'Time_target'] == T)) for K in m.K)
                      == sum(m.prd[RD] for RD in m.RD if requests_down.at[RD,'Time_target'] == T))
            
            
        # Constraint 5: Flexibility limits for requests and offers
        for ru in m.RU:
            m.pru_limit.add(pyo.inequality(0,m.pru[ru],requests_up.at[ru,'Quantity']))
        for rd in m.RD:
            m.prd_limit.add(pyo.inequality(0,m.prd[rd],requests_down.at[rd,'Quantity']))
        for ou in m.OU:
            m.pou_limit.add(pyo.inequality(0,m.pou[ou],offers_up.at[ou,'Quantity']))
        for od in m.OD:
            m.pod_limit.add(pyo.inequality(0,m.pod[od],offers_down.at[od,'Quantity']))  
        
        return m

    m = dc_opf()
    
    #choose the solver
    opt = pyo.SolverFactory('gurobi')
    results = opt.solve(m)
    #m.display()
    m.dual.display()

    for K in np.unique(offers_BB['Block']):
        AR_results.loc[K] = m.AR[K].value
    
    #%% solutions
    
    printsol = 1
    # solution
    if printsol == 1:
        energy_volume_up_total = 0
        energy_volume_down_total = 0
        social_welfare_total = 0
        market_result = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
        bids_out = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
        req_out = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
        off_out = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
        bids_in = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
        req_in = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
        off_in = pd.DataFrame(columns = ['Time_target','Bid','ID','Bus','Direction','Quantity','Price','Time_stamp','Block']) 
    
        line_flow = pd.DataFrame(columns = ['Time_target','Line','Power_flow','Line_limit']) 
        line_flow_per = pd.DataFrame(columns = ['Time Period','l1','l2','l3','l4','l5','l6','l7','l8','l9','l10','l11','l12','l13','l14','l15','l16','l17','l18','l19','l20','l21','l22','l23','l24','l25','l26','l27','l28','l29','l30','l31','l32'])
        line_flow_per.set_index('Time Period',inplace=True)
        energy_volume_up = pd.DataFrame(columns = ['Time_target','Energy_volume'])
        energy_volume_up.set_index('Time_target',inplace=True)
        energy_volume_down = pd.DataFrame(columns = ['Time_target','Energy_volume'])
        energy_volume_down.set_index('Time_target',inplace=True)
        SocialW = pd.DataFrame(columns = ['Time_target','Social Welfare'])
        SocialW.set_index('Time_target',inplace=True)
    
    
        for T in Setpoint.index:
            print ('----Time period----',T)
            print ('Offers Up')
            energy_volume_up_offers = 0
            energy_volume_down_offers = 0
            energy_volume_up_req = 0
            energy_volume_down_req = 0
            for OU in offers_up.index:
                pou = m.pou[OU].value
                if offers_up.at[OU,'Time_target'] == T:       
                    if m.pou[OU].value > epsilon:
                        market_result = market_result.append({'Time_target':offers_up.at[OU,'Time_target'],'ID':OU,'Bid':'Offer','Bus':offers_up.at[OU,'Bus'],'Direction':'Up','Quantity':round(m.pou[OU].value,2),'Price':offers_up.at[OU,'Price'],'Time_stamp':offers_up.at[OU,'Time_stamp'],'Block':'No'},ignore_index=True)
                        #print (OU,pou,offers_up.at[OU,'Price'],offers_up.at[OU,'Time_target'])
                        energy_volume_up_offers += m.pou[OU].value
                        bids_in = bids_in.append({'Time_target':offers_up.at[OU,'Time_target'],'ID':OU,'Bid':'Offer','Bus':offers_up.at[OU,'Bus'],'Direction':'Up','Quantity':offers_up.at[OU,'Quantity'],'Price':offers_up.at[OU,'Price'],'Time_stamp':offers_up.at[OU,'Time_stamp'],'Block':'No'},ignore_index=True)                 
                    else:
                        bids_out = bids_out.append({'Time_target':offers_up.at[OU,'Time_target'],'ID':OU,'Bid':'Offer','Bus':offers_up.at[OU,'Bus'],'Direction':'Up','Quantity':offers_up.at[OU,'Quantity'],'Price':offers_up.at[OU,'Price'],'Time_stamp':offers_up.at[OU,'Time_stamp'],'Block':'No'},ignore_index=True)
                    
            for BBU in offers_BB_up.index:
                if offers_BB_up.at[BBU,'Time_target'] == T:
                    K = offers_BB_up.at[BBU,'Block']
                    if round(m.AR[K].value*offers_BB_up.at[BBU,'Quantity'],2) > 0.00:
                        market_result = market_result.append({'Time_target':offers_BB_up.at[BBU,'Time_target'],'ID':BBU,'Bid':'Offer','Bus':offers_BB_up.at[BBU,'Bus'],'Direction':'Up','Quantity':round(m.AR[K].value*offers_BB_up.at[BBU,'Quantity'],2),'Price':offers_BB_up.at[BBU,'Price'],'Time_stamp':offers_BB_up.at[BBU,'Time_stamp'],'Block':K},ignore_index=True)
                        print (BBU,round(m.AR[K].value*offers_BB_up.at[BBU,'Quantity'],2),offers_BB_up.at[BBU,'Price'],offers_BB_up.at[BBU,'Time_target'],'BB')
                        energy_volume_up_offers += m.AR[K].value*offers_BB_up.at[BBU,'Quantity']
                    if m.AR[K].value*offers_BB_up.at[BBU,'Quantity'] > epsilon:
                        bids_in = bids_in.append({'Time_target':offers_BB_up.at[BBU,'Time_target'],'ID':BBU,'Bid':'Offer','Bus':offers_BB_up.at[BBU,'Bus'],'Direction':'Up','Quantity':offers_BB_up.at[BBU,'Quantity'],'Price':offers_BB_up.at[BBU,'Price'],'Time_stamp':offers_BB_up.at[BBU,'Time_stamp'],'Block':K},ignore_index=True)
                    else: 
                        bids_out = bids_out.append({'Time_target':offers_BB_up.at[BBU,'Time_target'],'ID':BBU,'Bid':'Offer','Bus':offers_BB_up.at[BBU,'Bus'],'Direction':'Up','Quantity':offers_BB_up.at[BBU,'Quantity'],'Price':offers_BB_up.at[BBU,'Price'],'Time_stamp':offers_BB_up.at[BBU,'Time_stamp'],'Block':K},ignore_index=True)
                       
                        
            print ('Offers Down')
            for OD in offers_down.index:
                pod = m.pod[OD].value
                if offers_down.at[OD,'Time_target'] == T:
                    if m.pod[OD].value > epsilon:
                        market_result = market_result.append({'Time_target':offers_down.at[OD,'Time_target'],'ID':OD,'Bid':'Offer','Bus':offers_down.at[OD,'Bus'],'Direction':'Down','Quantity':pod,'Price':offers_down.at[OD,'Price'],'Time_stamp':offers_down.at[OD,'Time_stamp'],'Block':'No'},ignore_index=True)
                        print (OD,pod,offers_down.at[OD,'Price'],offers_down.at[OD,'Time_target'])
                        energy_volume_down_offers += m.pod[OD].value
                        bids_in = bids_in.append({'Time_target':offers_down.at[OD,'Time_target'],'ID':OD,'Bid':'Offer','Bus':offers_down.at[OD,'Bus'],'Direction':'Down','Quantity':offers_down.at[OD,'Quantity'],'Price':offers_down.at[OD,'Price'],'Time_stamp':offers_down.at[OD,'Time_stamp'],'Block':'No'},ignore_index=True)                    
                    else:
                        bids_out = bids_out.append({'Time_target':offers_down.at[OD,'Time_target'],'ID':OD,'Bid':'Offer','Bus':offers_down.at[OD,'Bus'],'Direction':'Down','Quantity':offers_down.at[OD,'Quantity'],'Price':offers_down.at[OD,'Price'],'Time_stamp':offers_down.at[OD,'Time_stamp'],'Block':'No'},ignore_index=True)
                   
                    
            for BBD in offers_BB_down.index:
                if offers_BB_down.at[BBD,'Time_target'] == T:
                    K = offers_BB_down.at[BBD,'Block']
                    pbbd = m.AR[K].value*offers_BB_down.at[BBD,'Quantity']
                    if m.AR[K].value*offers_BB_down.at[BBD,'Quantity'] > epsilon:
                        bids_in = bids_in.append({'Time_target':offers_BB_down.at[BBD,'Time_target'],'ID':BBD,'Bid':'Offer','Bus':offers_BB_down.at[BBD,'Bus'],'Direction':'Down','Quantity':offers_BB_down.at[BBD,'Quantity'],'Price':offers_BB_down.at[BBD,'Price'],'Time_stamp':offers_BB_down.at[BBD,'Time_stamp'],'Block':K},ignore_index=True)             
                        market_result = market_result.append({'Time_target':offers_BB_down.at[BBD,'Time_target'],'ID':BBD,'Bid':'Offer','Bus':offers_BB_down.at[BBD,'Bus'],'Direction':'Down','Quantity':pbbd,'Price':offers_BB_down.at[BBD,'Price'],'Time_stamp':offers_BB_down.at[BBD,'Time_stamp'],'Block':K},ignore_index=True)             
                        print (BBD,pbbd,offers_BB_down.at[BBD,'Price'],offers_BB_down.at[BBD,'Time_target'],'BB')
                        energy_volume_down_offers += m.AR[K].value*offers_BB_down.at[BBD,'Quantity']
                    else:
                        bids_out = bids_out.append({'Time_target':offers_BB_down.at[BBD,'Time_target'],'ID':BBD,'Bid':'Offer','Bus':offers_BB_down.at[BBD,'Bus'],'Direction':'Down','Quantity':offers_BB_down.at[BBD,'Quantity'],'Price':offers_BB_down.at[BBD,'Price'],'Time_stamp':offers_BB_down.at[BBD,'Time_stamp'],'Block':K},ignore_index=True)             
      
                    
            print ('Requests Up')
            for RU in requests_up.index:
                pru = m.pru[RU].value
                if requests_up.at[RU,'Time_target'] == T:
                    if m.pru[RU].value > epsilon:
                        market_result = market_result.append({'Time_target':requests_up.at[RU,'Time_target'],'ID':RU,'Bid':'Request','Bus':requests_up.at[RU,'Bus'],'Direction':'Up','Quantity':pru,'Price':requests_up.at[RU,'Price'],'Time_stamp':requests_up.at[RU,'Time_stamp'],'Block':'No'},ignore_index=True)                
                        print (RU,pru,requests_up.at[RU,'Price'],requests_up.at[RU,'Time_target'])
                        energy_volume_up_req += m.pru[RU].value
                        bids_in = bids_in.append({'Time_target':requests_up.at[RU,'Time_target'],'ID':RU,'Bid':'Request','Bus':requests_up.at[RU,'Bus'],'Direction':'Up','Quantity':requests_up.at[RU,'Quantity'],'Price':requests_up.at[RU,'Price'],'Time_stamp':requests_up.at[RU,'Time_stamp'],'Block':'No'},ignore_index=True)                
                    else:
                        bids_out = bids_out.append({'Time_target':requests_up.at[RU,'Time_target'],'ID':RU,'Bid':'Request','Bus':requests_up.at[RU,'Bus'],'Direction':'Up','Quantity':requests_up.at[RU,'Quantity'],'Price':requests_up.at[RU,'Price'],'Time_stamp':requests_up.at[RU,'Time_stamp'],'Block':'No'},ignore_index=True)                
                       
                    
            print ('Requests Down')
            for RD in requests_down.index:
                prd = m.prd[RD].value
                if requests_down.at[RD,'Time_target'] == T:
                    if m.prd[RD].value > epsilon:
                        market_result = market_result.append({'Time_target':requests_down.at[RD,'Time_target'],'ID':RD,'Bid':'Request','Bus':requests_down.at[RD,'Bus'],'Direction':'Down','Quantity':prd,'Price':requests_down.at[RD,'Price'],'Time_stamp':requests_down.at[RD,'Time_stamp'],'Block':'No'},ignore_index=True)                
                        print (RD,prd,requests_down.at[RD,'Price'],requests_down.at[RD,'Time_target'])
                        energy_volume_down_req += m.prd[RD].value
                        bids_in = bids_in.append({'Time_target':requests_down.at[RD,'Time_target'],'ID':RD,'Bid':'Request','Bus':requests_down.at[RD,'Bus'],'Direction':'Down','Quantity':requests_down.at[RD,'Quantity'],'Price':requests_down.at[RD,'Price'],'Time_stamp':requests_down.at[RD,'Time_stamp'],'Block':'No'},ignore_index=True)                
                    else:
                        bids_out = bids_out.append({'Time_target':requests_down.at[RD,'Time_target'],'ID':RD,'Bid':'Request','Bus':requests_down.at[RD,'Bus'],'Direction':'Down','Quantity':requests_down.at[RD,'Quantity'],'Price':requests_down.at[RD,'Price'],'Time_stamp':requests_down.at[RD,'Time_stamp'],'Block':'No'},ignore_index=True)                
                    
            # Indicators:
            energy_volume_down_total += energy_volume_down_offers + energy_volume_down_req
            energy_volume_up_total += energy_volume_up_offers + energy_volume_up_req
            energy_volume_up.at[T,'Energy_volume'] = energy_volume_up_offers
            energy_volume_down.at[T,'Energy_volume'] = energy_volume_down_offers
            
            Social_Welfare = - (sum(m.pou[OU].value*offers_up.at[OU,'Price'] for OU in m.OU if offers_up.at[OU,'Time_target'] == T) 
            + sum(m.pod[OD].value*offers_down.at[OD,'Price'] for OD in m.OD if offers_down.at[OD,'Time_target'] == T)
            + sum(m.AR[K].value*(sum(offers_BB.at[BB,'Quantity']*offers_BB.at[BB,'Price'] for BB in m.BB if offers_BB.at[BB,'Block'] == K and offers_BB.at[BB,'Time_target'] == T)) for K in m.K)
            - sum(m.pru[RU].value*requests_up.at[RU,'Price'] for RU in m.RU if requests_up.at[RU,'Time_target'] == T) 
            - sum(m.prd[RD].value*requests_down.at[RD,'Price'] for RD in m.RD if requests_down.at[RD,'Time_target'] == T))
            SocialW.at[T,'Social Welfare'] = Social_Welfare

            social_welfare_total += Social_Welfare
            
            # Line flow calculations:
    
            for L in lines: 
                line_flow = line_flow.append({'Time_target':T,'Line':L,'Power_flow': abs(m.f[L,T].value),'Line_limit':branch.at[L,'Pmax'],'Line_capacity':100*abs(m.f[L,T].value)/branch.at[L,'Pmax']},ignore_index=True)                    
                line_flow_per.at[T,L] = round(abs(m.f[L,T].value)/branch.at[L,'Pmax'],2)
                
        print ('Total Social Welfare',round(social_welfare_total,4))
        energy_volume = (energy_volume_up_total+energy_volume_down_total)/2
        print ('Total energy volume',round((energy_volume_up_total+energy_volume_down_total)/2,2)) 
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Time: ", total_time)

    return (social_welfare_total,energy_volume,total_time)

