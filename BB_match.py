# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:23:55 2021

@author: aalarcon
"""

from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo
import pyomo.gdp as gdp
from PTDF_check import PTDF_check

import matplotlib.pyplot as plt
import pandas as pd
import ast
import sys

#%% Testing inputs

# Index for nodes
bus = pd.read_excel(open('network33bus.xlsx','rb'),sheet_name='Bus',index_col=0)
nodes = list(bus.index)

# Index for branches
branch= pd.read_excel(open('network33bus.xlsx','rb'),sheet_name='Branch',index_col=0)
lines = list(branch.index)

#%%

def BB_match(setpoint,orderbook_temporary,offer_index,offer_quantity, lines, nodes, direction, offer_bus, offer_price, offer_block):
    
    epsilon = 0.00001 # Tolerance
    
    #Adjust Setpoint format
    Setpoint = pd.DataFrame(columns = nodes)
    setpoint = pd.Series(setpoint, index=Setpoint.columns) 
    Setpoint = Setpoint.append(setpoint, ignore_index=True)
    
    # Create empty dataframes
    requests_up = pd.DataFrame(columns = ['ID','Bus','Quantity','Price'])
    requests_up.set_index('ID',inplace=True)
    requests_down = pd.DataFrame(columns = ['ID','Bus','Quantity','Price'])
    requests_down.set_index('ID',inplace=True)
    BB_matches = pd.DataFrame(columns = ['Offer','Offer Bus','Offer Price','Offer Block','Request','Request Bus','Request Price','Direction','Quantity','Matching Price','Time_target','Social Welfare'])

    # Consider only the temporary matches that include the part of the BB
    for match in orderbook_temporary.index:
        if orderbook_temporary.at[match,'Offer'] != offer_index:
            orderbook_temporary = orderbook_temporary.drop([match], axis=0) # Remove the temporary match  
        else:
            offer_bus = orderbook_temporary.at[match,'Offer Bus']
            if direction == 'Up':
                requests_up.loc[orderbook_temporary.at[match,'Request']] = [orderbook_temporary.at[match,'Request Bus'],orderbook_temporary.at[match,'Quantity'],orderbook_temporary.at[match,'Request Price']]
            else:
                requests_down.loc[orderbook_temporary.at[match,'Request']] = [orderbook_temporary.at[match,'Request Bus'],orderbook_temporary.at[match,'Quantity'],orderbook_temporary.at[match,'Request Price']]
    
    if orderbook_temporary.shape[0]==1:
        print ('Only one request to match with')
        for only in orderbook_temporary.index:
            print ('No optimization problem needed')
            offer_bus = nodes.index(orderbook_temporary.at[only,'Offer Bus'])
            request_bus = nodes.index(orderbook_temporary.at[only,'Request Bus'])
            Q = PTDF_check(setpoint,offer_quantity,offer_bus,request_bus,direction) # Returns the maximum quantity that can be exchanged without leading to congestions
            print ('Quantity to be exchanged according to PTDF {}'.format(Q))
            if Q > epsilon:
                SW = (orderbook_temporary.at[only,'Matching Price'])*offer_quantity
                BB_matches = BB_matches.append({'Offer':offer_index,'Offer Bus':orderbook_temporary.at[only,'Offer Bus'],'Offer Price':orderbook_temporary.at[only,'Offer Price'],'Offer Block':offer_block,'Request':orderbook_temporary.at[only,'Request'],'Request Bus':orderbook_temporary.at[only,'Request Bus'],'Request Price':orderbook_temporary.at[only,'Request Price'],'Matching Price':orderbook_temporary.at[only,'Matching Price'],'Direction':direction,'Quantity':offer_quantity, 'Time_target':orderbook_temporary.at[only,'Time_target'], 'Social Welfare':SW},ignore_index=True)               
        
        return BB_matches
            
    else:
        if direction == 'Up':
            pou = offer_quantity
            pod = 0
        else:
            pou = 0
            pod = offer_quantity
        
        branch['B'] = 1/branch['X']
        n_ref = 'n1'
        
     #%% Optimization problem 
        def req_selection():
            
            print ('Calling optimization problem')
            
            m = pyo.ConcreteModel()
            
            # Sets creation
            m.N = pyo.Set(initialize = nodes)
            m.L = pyo.Set(initialize = lines) 
            m.RU = pyo.Set(initialize = requests_up.index)
            m.RD = pyo.Set(initialize = requests_down.index)
            
            # Variables creation
            m.pru = pyo.Var(m.RU, domain=pyo.NonNegativeReals)
            m.prd = pyo.Var(m.RD, domain=pyo.NonNegativeReals)
            m.slack = pyo.Var(m.L, domain=pyo.NonNegativeReals)
            m.theta = pyo.Var(m.N, domain=pyo.Reals) 
            m.f = pyo.Var(m.L, domain=pyo.Reals) 
        
            def flow_lim(model,L):
                return m.f[L]+m.slack[L] >= -branch.loc[L,'Pmax']
            def flow_lim2(model,L):
                return m.f[L]-m.slack[L] <= branch.loc[L,'Pmax']
            
            def pru_lim(model,RU):
                return m.pru[RU] <= requests_up.at[RU,'Quantity']
            def prd_lim(model,RD):
                return m.prd[RD] <= requests_down.at[RD,'Quantity']

            # Constraints
            m.ang_ref = pyo.Constraint(expr = m.theta[n_ref] == 0) # Reference angle
            m.flow_eq = pyo.ConstraintList() # Distribution lines limits
            m.flow_bounds = pyo.Constraint(m.L, rule = flow_lim) 
            m.flow_bounds2 = pyo.Constraint(m.L, rule = flow_lim2)
            m.nod_bal = pyo.ConstraintList() # Power balance per node

            if direction == 'Up':
                m.up = pyo.Constraint(expr = sum(m.pru[RU] for RU in m.RU) == pou)
            else:
                m.down = pyo.Constraint(expr = sum(m.prd[RD] for RD in m.RD) == pod)

            # objective function 
            m.obj_cost = pyo.Objective(expr= (pou + pod)*offer_price
                                       + sum(m.slack[L]*1000 for L in m.L) # One per line 
                                       - sum(m.pru[RU]*requests_up.at[RU,'Price'] for RU in m.RU) 
                                       - sum(m.prd[RD]*requests_down.at[RD,'Price'] for RD in m.RD), sense=pyo.minimize)
    
            # Constraint: Distribution lines limits
            for L in m.L:
                m.flow_eq.add(m.f[L] - branch.loc[L,'B']*(m.theta[branch.loc[L,'From']] - m.theta[branch.loc[L,'To']]) == 0)
                
            # Constraint: Power balance per node # Setpoint + Pou - Pod - Pru + Prd - losses = 0
            for N in m.N: 
                if N == offer_bus:
                    m.nod_bal.add(Setpoint.at[0,N] + pou - pod
                              - sum(m.pru[RU] for RU in m.RU if (requests_up.at[RU,'Bus'] == N)) 
                              + sum(m.prd[RD] for RD in m.RD if (requests_down.at[RD,'Bus'] == N)) 
                              # line flow
                              - sum(branch.at[L,'B']*(m.theta[branch.at[L,'From']] - m.theta[branch.at[L,'To']]) for L in branch.index if branch.at[L,'From'] == N)
                              - sum(branch.at[L,'B']*(m.theta[branch.at[L,'To']] - m.theta[branch.at[L,'From']]) for L in branch.index if branch.at[L,'To'] == N) == 0)
                else:
                    m.nod_bal.add(Setpoint.at[0,N]
                              - sum(m.pru[RU] for RU in m.RU if (requests_up.at[RU,'Bus'] == N)) 
                              + sum(m.prd[RD] for RD in m.RD if (requests_down.at[RD,'Bus'] == N)) 
                              # line flow
                              - sum(branch.at[L,'B']*(m.theta[branch.at[L,'From']] - m.theta[branch.at[L,'To']]) for L in branch.index if branch.at[L,'From'] == N)
                              - sum(branch.at[L,'B']*(m.theta[branch.at[L,'To']] - m.theta[branch.at[L,'From']]) for L in branch.index if branch.at[L,'To'] == N) == 0)
    
            # Constraint: Flexibility limits for requests and offers
            print ('Offer quantity')
            print ('Request quantity available')
            for ru in m.RU:
                print (ru,requests_up.at[ru,'Quantity'])
            for rd in m.RD:
                print (rd,requests_down.at[rd,'Quantity'])
            return m

        m = req_selection()
        
        #choose the solver
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(m)

        #Check if the problem if feasible
        print ('Check feasibility')
        slack = 0
        for L in m.L:
            slack += m.slack[L].value
        
        if slack < epsilon:
            print ('Feasible')

            for RU in m.RU:
                print (RU, m.pru[RU].value)
                if not m.pru[RU].value:
                    continue
                elif m.pru[RU].value > epsilon:
                    a = orderbook_temporary[orderbook_temporary['Request']==RU]
                    # Define matching price according to the arrival order
                    if a.iloc[-1,8] < a.iloc[-1,4]:
                        matching_price = a.iloc[-1,7]
                    elif a.iloc[-1,8] > a.iloc[-1,4]:
                        matching_price = a.iloc[-1,3]
                        
                    SW = (requests_up.at[RU,'Price']-offer_price)*m.pru[RU].value
                    BB_matches = BB_matches.append({'Offer':offer_index,'Offer Bus':offer_bus,'Offer Price':offer_price,'Offer Block':offer_block,'Request':RU,'Request Bus':requests_up.at[RU,'Bus'],'Request Price':requests_up.at[RU,'Price'],'Matching Price':matching_price,'Direction':direction,'Quantity':m.pru[RU].value, 'Time_target':a.iloc[-1,-1], 'Social Welfare':SW},ignore_index=True)               
            
            for RD in m.RD:
                print (RD,m.prd[RD].value)
                if not m.prd[RD].value:
                    print ('No values')
                elif m.prd[RD].value > epsilon:
                    a = orderbook_temporary[orderbook_temporary['Request']==RD]
                    
                    # Define matching price according to the arrival order
                    if a.iloc[-1,8] < a.iloc[-1,4]:
                        matching_price = a.iloc[-1,7]
                    elif a.iloc[-1,8] > a.iloc[-1,4]:
                        matching_price = a.iloc[-1,3]
                    
                    SW = (requests_down.at[RD,'Price']-offer_price)*m.prd[RD].value
                    BB_matches = BB_matches.append({'Offer':offer_index,'Offer Bus':offer_bus,'Offer Price':offer_price,'Offer Block':offer_block,'Request':RD,'Request Bus':requests_down.at[RD,'Bus'],'Request Price':requests_down.at[RD,'Price'],'Matching Price':matching_price,'Direction':direction,'Quantity':m.prd[RD].value, 'Time_target':a.iloc[-1,-1], 'Social Welfare':SW},ignore_index=True)               
            
        else:
            print ('Infeasible')
        
        return BB_matches
