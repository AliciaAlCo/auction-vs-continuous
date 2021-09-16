# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:13:54 2020

@author: emapr
"""
import pandas as pd

# Defines a function that returns the minimum quantity of power that can be exchanged between an offer bus and a request bus, without leading to congestions

def PTDF_check(SetPoint,Quantity,offer_bus,request_bus,direction):
    
    epsilon=0.00001 # Tolerance

    lines_df = pd.read_excel(open('network33bus.xlsx', 'rb'),sheet_name='Branch',index_col=0)
    lines = list(lines_df.index)
    
    PTDF = pd.read_excel(open('network33bus.xlsx', 'rb'),sheet_name='PTDF',index_col=0)
    nodes = list(PTDF.columns)

    # Initial state
    Pl_flow=[] # List for the line flows
    Pl_max_pos = [] # List for the maximum variation of the line flows in the same direction
    Pl_max_neg = [] # List for the maximum variation of the line flows in the other direction
    
    for l in lines: # Calculate power flow in each line
        Pl = 0
        for i in range(len(nodes)):
            Pl += PTDF.loc[l,nodes[i]] * (SetPoint[i]) # Calculate the power flow in the line by adding the contribution of each bus
        # if abs(Pl) > (data[l]['lineCapacity'] + epsilon): # Make sure that the initial power flows are feasible
            # print('The initial dispatch is not feasible ({}): {} > {}'.format(l,abs(Pl),(data[l]['lineCapacity'] + epsilon)))
        Pl_max_pos.append(lines_df.at[l,'Pmax']-Pl) # Calculate the maximum variation of the power flow in the same direction for this line
        Pl_max_neg.append(-lines_df.at[l,'Pmax']-Pl) # Calculate the maximum variation of the power flow with a change of direction for this line
        Pl_flow.append(Pl)
    # Define the proper buses depending on the direction of the bids
    if direction == 'Up':
        k = nodes[offer_bus]    
        m = nodes[request_bus]
    if direction == 'Down':
        m = nodes[offer_bus]
        k = nodes[request_bus]
    
    # Update the quantity to make sure that the line flows are all feasible
    for l in range(len(lines)):
        PTDF_diff = - (PTDF.loc[lines[l],m] - PTDF.loc[lines[l],k])
        # First calculate the maximum power flow change in the line Pl_max
        if PTDF_diff > epsilon: # If the power is flowing in the same direction
            Pl_max = max(Pl_max_pos[l],0)
        elif PTDF_diff < -epsilon: # If the power is flowing in the other direction
            Pl_max = min(Pl_max_neg[l],0)
        # Then update the quantity
        if PTDF_diff > epsilon or PTDF_diff < -epsilon: # The difference between the two PTDFs is not equal to zero
            if Pl_max/PTDF_diff < Quantity: # If the quantity is bigger than the max for this line, update it to be equal to the max for this line
                Quantity = Pl_max/PTDF_diff
    
            
                
    return Quantity
